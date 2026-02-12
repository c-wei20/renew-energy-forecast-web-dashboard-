import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import tf_keras as keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

st.set_page_config(
    page_title="Renewable Energy Production",
    page_icon="🌱",
    layout="wide",
)

# Load the renewable energy data
@st.cache_data  # Enable caching for improved performance
def load_data():
    # Define your filtered start dates
    start_date = '2017-01-01'

    hydro_data = pd.read_csv('./dataset/hydro_daily_production_dataset.csv')
    hydro_data['Energy Source'] = 'Hydro'
    hydro_data['Date'] = pd.to_datetime(hydro_data['Date'])
    hydro_data = hydro_data[(hydro_data['Date'] >= start_date)]

    wind_data = pd.read_csv('./dataset/wind_daily_production_dataset.csv')
    wind_data['Energy Source'] = 'Wind'
    wind_data['Date'] = pd.to_datetime(wind_data['Date'])
    wind_data = wind_data[(wind_data['Date'] >= start_date)]

    solar_data = pd.read_csv('./dataset/solar_daily_production_dataset.csv')
    solar_data['Energy Source'] = 'Solar'
    solar_data['Date'] = pd.to_datetime(solar_data['Date'])
    solar_data = solar_data[(solar_data['Date'] >= start_date)]

    # data = pd.concat([hydro_data, wind_data, solar_data], ignore_index=True)

    return hydro_data, wind_data, solar_data


def forecast_data(data, forecast_models, eva_matrix_df):
    countries = data['Country'].unique()

    data_dict = {}

    # use loc() function to select data from a DataFrame based on specified row and column labels or conditions.
    # Iterate over the countries
    for country in countries:
        # Filter the data for the current country and energy
        country_energy_production = data.loc[
            (data['Country'] == country)
        ]
        # Store the filtered data in the dictionary
        data_dict[country] = country_energy_production.reset_index(drop=True)

    split_date = '2021-01-01'
    time_steps = 365

    for country in data_dict:

        data = data_dict[country].copy()
        data.set_index('Date', inplace=True)
        train, test = data[:split_date], data[split_date:]

        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(train[['Value']].values)

        inputs = data[len(data) - len(test) - time_steps:]['Value'].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(time_steps, len(test) + time_steps):
            X_test.append(inputs[i - time_steps:i, 0])
        X_test = np.array(X_test)

        forecast_value = forecast_models[country].predict(X_test)
        forecast_value = sc.inverse_transform(forecast_value)

        mae = mean_absolute_error(forecast_value, test['Value'].values)
        mse = mean_squared_error(forecast_value, test['Value'].values)
        rmse = math.sqrt(mse)

        new_eva_matrix = {'Country': country, 'Energy Source': data['Energy Source'][0], 'MAE': mae, 'MSE': mse, 'RMSE': rmse}
        # Convert the dictionary to a DataFrame and use concat
        new_row_df = pd.DataFrame([new_eva_matrix])

        # Adding a new evaluation matrix
        eva_matrix_df = pd.concat([eva_matrix_df, new_row_df], ignore_index=True)

        forecast = pd.DataFrame(forecast_value, columns=['Forecast Value'], index=pd.date_range(start=split_date, periods=len(forecast_value), freq='D'))
        forecast.reset_index(inplace=True)
        forecast.rename(columns={'index': 'Date'}, inplace=True)
        forecast['Date'] = pd.to_datetime(forecast['Date'])

        data_dict[country] = pd.merge(data_dict[country], forecast, on='Date', how='left')

    final_dataset = pd.concat(data_dict.values())
    final_dataset.reset_index(drop=True, inplace=True)

    return final_dataset, eva_matrix_df

# Sidebar filters
def sidebar_filters(data, eva_matrix_df):
    st.sidebar.header('Filters')
    countries = data['Country'].unique()
    selected_country = st.sidebar.multiselect(
        "Select countries", countries, ['United States', 'Canada', 'Germany', 'Brazil', 'Australia']
    )

    energy_sources = data['Energy Source'].unique()
    selected_source = st.sidebar.selectbox('Select Energy Source', energy_sources)

    years = data['Date'].dt.year.unique()
    min_year, max_year = int(min(years)), int(max(years))
    selected_min_year, selected_max_year = st.sidebar.slider(
        'Select Year Range', min_value=min_year, max_value=max_year, value=(min_year, max_year)
    )

    plot_frequency = st.sidebar.radio("Select Granularity", ["Daily", "Monthly", "Yearly"], index=1)

    filtered_data = data[(data['Country'].isin(selected_country)) &
                         (data['Energy Source'] == selected_source) &
                         (data['Date'].dt.year.between(selected_min_year, selected_max_year))]
    filtered_eva_matrix = eva_matrix_df[(eva_matrix_df['Country'].isin(selected_country)) &
                                        (eva_matrix_df['Energy Source'] == selected_source)]

    if plot_frequency == "Daily":
        filtered_data = filtered_data.reset_index()
        return filtered_data, filtered_eva_matrix.reset_index()
    elif plot_frequency == "Monthly":
        filtered_data = filtered_data.groupby(['Country', pd.Grouper(key='Date', freq='MS')]).sum().reset_index()
        filtered_data['Forecast Value'] = filtered_data['Forecast Value'].replace(0, np.nan)
    else:
        filtered_data = filtered_data.groupby(['Country', pd.Grouper(key='Date', freq='YS')]).sum().reset_index()
        filtered_data['Forecast Value'] = filtered_data['Forecast Value'].replace(0, np.nan)

    return filtered_data, filtered_eva_matrix.reset_index()


# Line chart with forecast
def line_chart(data):
    fig = go.Figure()

    # Plot the actual values with 'linear' shape
    for country in data['Country'].unique():
        country_data = data[data['Country'] == country]
        fig.add_trace(go.Scatter(x=country_data['Date'], y=country_data['Value'],
                                 mode='lines', name=f'Actual - {country}', line_shape='linear'))

    # Plot the forecast values with 'spline' shape
    for country in data['Country'].unique():
        country_data = data[data['Country'] == country]
        fig.add_trace(go.Scatter(x=country_data['Date'], y=country_data['Forecast Value'],
                                 mode='lines', name=f'Forecast - {country}', line_shape='spline'))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Production',
        title='Renewable Energy Production Over Time & Prediction',
        width=1000,
    )
    st.plotly_chart(fig)


# Bar chart
def bar_chart(data):
    fig = px.bar(data, x='Country', y='Value', color='Country', title='Renewable Energy Production by Country')
    fig.update_layout(
        width=1000,
    )
    st.plotly_chart(fig)

# Map
def map_chart(data):
    fig = px.choropleth(
        data_frame=data,
        locations='Country',
        locationmode='country names',
        color='Value',
        hover_name='Country',
        color_continuous_scale='Viridis',
        projection='natural earth'
    )

    fig.update_layout(
        title='Renewable Energy Production by Country',
        width=1000,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig)

@st.cache_resource
def load_all_models():
    # Load forecast models for each country and renewable source
    hydro_forecast_models = {'United States': keras.models.load_model('./forecast_models/Hydro/us_hydro_best_uni_lstm.h5'),
                             'Canada': keras.models.load_model('./forecast_models/Hydro/can_hydro_best_uni_lstm.h5'),
                             'Germany': keras.models.load_model('./forecast_models/Hydro/ger_hydro_best_uni_lstm.h5'),
                             'Brazil': keras.models.load_model('./forecast_models/Hydro/brz_hydro_best_uni_lstm.h5'),
                             "People's Republic of China": keras.models.load_model('./forecast_models/Hydro/chn_hydro_best_uni_lstm.h5'),
                             'Australia': keras.models.load_model('./forecast_models/Hydro/aus_hydro_best_uni_lstm.h5'),
                             }

    wind_forecast_models = {'United States': keras.models.load_model('./forecast_models/Wind/us_wind_best_gru.h5'),
                            'Canada': keras.models.load_model('./forecast_models/Wind/can_wind_best_gru.h5'),
                            'Germany': keras.models.load_model('./forecast_models/Wind/ger_wind_best_gru.h5'),
                            'Brazil': keras.models.load_model('./forecast_models/Wind/brz_wind_best_gru.h5'),
                            "People's Republic of China": keras.models.load_model('./forecast_models/Wind/chn_wind_best_gru.h5'),
                            'Australia': keras.models.load_model('./forecast_models/Wind/aus_wind_best_gru.h5'),
                            }

    solar_forecast_models = {'United States': keras.models.load_model('./forecast_models/Solar/us_solar_best_gru.h5'),
                             'Canada': keras.models.load_model('./forecast_models/Solar/can_solar_best_gru.h5'),
                             'Germany': keras.models.load_model('./forecast_models/Solar/ger_solar_best_gru.h5'),
                             'Brazil': keras.models.load_model('./forecast_models/Solar/brz_solar_best_gru.h5'),
                             "People's Republic of China": keras.models.load_model('./forecast_models/Solar/chn_solar_best_gru.h5'),
                             'Australia': keras.models.load_model('./forecast_models/Solar/aus_solar_best_gru.h5'),
                             }

    return hydro_forecast_models, wind_forecast_models, solar_forecast_models

@st.cache_data
def get_processed_data(_hydro_models, _wind_models, _solar_models):
    # Load and process data only when models or source files change
    hydro_data, wind_data, solar_data = load_data()
    
    # Creating an evaluation matrix DataFrame
    columns = ['Country', 'Energy Source', 'MAE', 'MSE', 'RMSE']
    eva_matrix_df = pd.DataFrame(columns=columns)

    #forecast data
    hydro_data, eva_matrix_df  = forecast_data(hydro_data, _hydro_models, eva_matrix_df)
    wind_data, eva_matrix_df = forecast_data(wind_data, _wind_models, eva_matrix_df)
    solar_data, eva_matrix_df = forecast_data(solar_data, _solar_models, eva_matrix_df)
    
    combined_data = pd.concat([hydro_data, wind_data, solar_data], ignore_index=True)
    return combined_data, eva_matrix_df


# Main function
def main():
    st.title('Global Renewable Energy Dashboard')

    # These will only run ONCE (the first time the app starts)
    h_models, w_models, s_models = load_all_models()
    all_data, full_eva_matrix = get_processed_data(h_models, w_models, s_models)

    # Sidebar filters
    filtered_data, filtered_eva_matrix = sidebar_filters(all_data, full_eva_matrix)

    # Line chart
    st.subheader('Renewable Energy Production Over Time & Prediction')
    line_chart(filtered_data)

    # Display filtered data
    st.subheader('Evaluation Matrix')
    st.write(filtered_eva_matrix[['Country', 'Energy Source', 'MAE', 'MSE', 'RMSE']])

    # Bar chart
    st.subheader('Renewable Energy Production by Country')
    bar_chart(filtered_data)

    # Map
    st.subheader('Renewable Energy Production by Country (Map)')
    map_chart(filtered_data)

    # Display filtered data
    st.subheader('Filtered Data')
    st.write(filtered_data)

if __name__ == '__main__':
    main()
