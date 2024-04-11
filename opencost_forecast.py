#############################################################################
## Models cross validation for containers CPU and Memory measurements retrieved 
## from Prometheus.
## 
#############################################################################
## Author: Simone Rosignoli
## 
##  
#############################################################################
import sqlite3
import sys
import requests
import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime, timedelta

from statsforecast.models import WindowAverage
from statsforecast.models import AutoARIMA
from statsforecast.models import AutoTheta
from statsforecast.models import AutoETS
from statsforecast.models import AutoCES
from statsforecast.models import MSTL
from statsforecast.models import SeasonalNaive
from statsforecast.models import SeasonalWindowAverage
from statsforecast.models import Naive
from prophet import Prophet

# this makes it so that the outputs of the predict methods have the id as a column 
# instead of as the index
os.environ['NIXTLA_ID_AS_COL'] = '1'

# Prometheus server details
PROMETHEUS = 'http://localhost:9001'

def get_container_memory(pod):
    QUERY = f'container_memory_allocation_bytes{{pod="{pod}"}}/1024/1024/1024'
    # Calculate start and end times for the last hour
    END = datetime.now()
    START = END - timedelta(hours=168)
    # Convert times to UNIX timestamps
    start_time = START.timestamp()
    end_time = END.timestamp()

    # Construct the query
    query_range_url = f"{PROMETHEUS}/api/v1/query_range"
    params = {
        'query': QUERY,
        'start': start_time,
        'end': end_time,
        'step': '60s'  # 60 seconds intervals
    }

    # Make the HTTP request to Prometheus
    response = requests.get(query_range_url, params=params)
    # Parse the JSON response
    data = response.json()['data']['result']

    # Initialize an empty list to hold the data points
    data_points = []

    # Extract the data points
    for result in data:
        for value in result['values']:
            # Convert timestamp to readable date
            timestamp = datetime.fromtimestamp(float(value[0]))
            memory_usage = float(value[1])
            data_points.append((timestamp, memory_usage))

    # Create a DataFrame
    df = pd.DataFrame(data_points, columns=['ds', 'y'])

    return df

def get_container_cpu(pod):
    QUERY = f'container_cpu_allocation{{pod="{pod}"}}'
    # Calculate start and end times for the last hour
    END = datetime.now()
    START = END - timedelta(hours=168)
    # Convert times to UNIX timestamps
    start_time = START.timestamp()
    end_time = END.timestamp()

    # Construct the query
    query_range_url = f"{PROMETHEUS}/api/v1/query_range"
    params = {
        'query': QUERY,
        'start': start_time,
        'end': end_time,
        'step': '60s'  # 60 seconds intervals
    }

    # Make the HTTP request to Prometheus
    response = requests.get(query_range_url, params=params)
    # Parse the JSON response
    data = response.json()['data']['result']

    # Initialize an empty list to hold the data points
    data_points = []

    # Extract the data points
    for result in data:
        for value in result['values']:
            # Convert timestamp to readable date
            timestamp = datetime.fromtimestamp(float(value[0]))
            memory_usage = float(value[1])
            data_points.append((timestamp, memory_usage))

    # Create a DataFrame
    df = pd.DataFrame(data_points, columns=['ds', 'y'])

    return df

def get_cpu_hourly_cost(node):
    query = f'node_cpu_hourly_cost{{node="{node}"}}'

    # Construct the API URL to query Prometheus
    query_url = f'{PROMETHEUS}/api/v1/query'
    params = {
        'query': query,
    }

    # Make the HTTP GET request to Prometheus
    response = requests.get(query_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        cpu_cost = result['data']['result'][0]['value'][1]
        return cpu_cost
    else:
        print("Failed to fetch data:", response.text)

def get_ram_hourly_cost(node):
    query = f'node_ram_hourly_cost{{node="{node}"}}'

    # Construct the API URL to query Prometheus
    query_url = f'{PROMETHEUS}/api/v1/query'
    params = {
        'query': query,
    }

    # Make the HTTP GET request to Prometheus
    response = requests.get(query_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        ram_cost = result['data']['result'][0]['value'][1]
        return ram_cost
    else:
        print("Failed to fetch data:", response.text)


def future_date(last_date, forecast_horizon):
    # Generate future timestamps with hourly frequency
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=forecast_horizon, freq='H')
    return future_dates

def add_future_dates_prediction_confidence_level(prediction, future_dates, model):
    #re-index with the timestamp
    mean_series = pd.DataFrame(data=prediction['mean'], index=future_dates)
    lo_series = pd.Series(prediction['lo-80'].values, index=future_dates)
    hi_series = pd.Series(prediction['hi-80'].values, index=future_dates)
    predictions_df = pd.concat([mean_series, lo_series, hi_series], axis=1)
    predictions_df.columns = [model, model + '-lo-80', model + '-hi-80']
    predictions_df = predictions_df.reset_index().rename(columns={'index': 'ds'})
    return predictions_df

def add_future_dates_prediction_confidence_level_df(prediction, future_dates, model):
    #re-index with the timestamp
    mean_series = pd.DataFrame(data=prediction['mean'], index=future_dates)
    lo_series = pd.DataFrame(prediction['lo-80'], index=future_dates)
    hi_series = pd.DataFrame(prediction['hi-80'], index=future_dates)
    predictions_df = pd.concat([mean_series, lo_series, hi_series], axis=1)
    predictions_df.columns = [model, model + '-lo-80', model + '-hi-80']
    predictions_df = predictions_df.reset_index().rename(columns={'index': 'ds'})
    return predictions_df

def add_future_dates_prediction(prediction, future_dates, model):
    #re-index with the timestamp
    mean_series = pd.DataFrame(data=prediction['mean'], index=future_dates)
    mean_series.columns = [model]
    mean_series_df = mean_series.reset_index().rename(columns={'index': 'ds'})
    return mean_series_df

def plot_save_chart_with_confidence_level(df, model, metric):
    # Plotting
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical Data', line=dict(color='blue')))

    # Predicted values
    fig.add_trace(go.Scatter(x=df['ds'], y=df[model], mode='lines', name='Predicted', line=dict(color='red')))

    # Confidence interval
    fig.add_trace(go.Scatter(x=df['ds'], y=df[model + '-lo-80'], mode='lines', name='Confidence Interval Lower Bound', line=dict(width=0)))
    fig.add_trace(go.Scatter(x=df['ds'], y=df[model + '-hi-80'], mode='lines', name='Confidence Interval Upper Bound', line=dict(width=0),
                            fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)'))  # Using fill to create the shaded confidence interval area

    # Layout adjustments
    fig.update_layout(title='Historical Data and 24H Predictions with Confidence Interval for ' + metric + ' predicted with ' + model,
                    xaxis_title='Date',
                    yaxis_title='Value',
                    showlegend=True)

    # Show plot
    fig.show()

    # Save plot to file
    fig.write_image(metric+"-historical_predictions_plot.png", width=800, height=600)

def plot_save_chart_prophet(df, metric):
    # Plotting
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical Data', line=dict(color='blue')))

    # Predicted values
    fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat'], mode='lines', name='Predicted', line=dict(color='red')))

    # Confidence interval
    fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_lower'], mode='lines', name='Confidence Interval Lower Bound', line=dict(width=0)))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_upper'], mode='lines', name='Confidence Interval Upper Bound', line=dict(width=0),
                            fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)'))  # Using fill to create the shaded confidence interval area

    # Layout adjustments
    fig.update_layout(title='Historical Data and 24H Predictions with Confidence Interval for ' + metric + ' predicted with Prophet',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    showlegend=True)

    # Show plot
    fig.show()

    # Save plot to file
    fig.write_image(metric+"-historical_predictions_plot.png", width=800, height=600)


def plot_save_chart(df, model, metric):
    # Plotting
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical Data', line=dict(color='blue')))

    # Predicted values
    fig.add_trace(go.Scatter(x=df['ds'], y=df[model], mode='lines', name='Predicted', line=dict(color='red')))

   
    # Layout adjustments
    fig.update_layout(title='Historical Data and 24H Predictions with Confidence Interval for ' + metric + ' predicted with ' + model,
                    xaxis_title='Date',
                    yaxis_title='Value',
                    showlegend=True)

    # Show plot
    fig.show()

    # Save plot to file
    fig.write_image(metric+"-historical_predictions_plot.png", width=800, height=600)

def plot_save_chart_cost_with_confidence_level(df, cpu_model, ram_model):
    # Plotting
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical Data', line=dict(color='blue')))

    # Predicted values
    fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast'], mode='lines', name='Predicted', line=dict(color='red')))

    # Confidence interval
    fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast-lo-80'], mode='lines', name='Confidence Interval Lower Bound', line=dict(width=0)))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast-hi-80'], mode='lines', name='Confidence Interval Upper Bound', line=dict(width=0),
                            fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)'))  # Using fill to create the shaded confidence interval area

    # Layout adjustments
    fig.update_layout(title='24H Cost Predictions with Confidence Interval for CPU predicted with ' + cpu_model + ' and RAM predicted with ' + ram_model,
                    xaxis_title='Date',
                    yaxis_title='Value',
                    showlegend=True)

    # Show plot
    fig.show()

    # Save plot to file
    fig.write_image("cost_predictions_plot.png", width=800, height=600)

# Function definitions for each forecasting model
def AutoARIMA_forecast(metric, ts):
    arima = AutoARIMA(season_length=24)
    arima = arima.fit(y=ts['y'].to_numpy())
    y_hat_dict = arima.predict(h=24, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = 24 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level(y_hat_dict, future_dates, "AutoARIMA")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    print(df_combined)
    plot_save_chart_with_confidence_level(df_combined, "AutoARIMA", metric)
    print(f"Forecasting with AutoARIMA for {metric}")
    return df_combined

def AutoTheta_forecast(metric, ts):
    theta = AutoTheta(season_length=24)
    theta = theta.fit(y=ts['y'].to_numpy())
    y_hat_dict = theta.predict(h=24, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = 24 
    future_dates = future_date(last_date, forecast_horizon)
    mean_series = pd.DataFrame(data=y_hat_dict['mean'], index=future_dates)
    lo_series = pd.DataFrame(y_hat_dict['lo-80'], index=future_dates)
    hi_series = pd.DataFrame(y_hat_dict['hi-80'], index=future_dates)
    predictions_df = pd.concat([mean_series, lo_series, hi_series], axis=1)
    predictions_df.columns = ["AutoTheta", 'AutoTheta-lo-80', 'AutoTheta-hi-80']
    predictions_df = predictions_df.reset_index().rename(columns={'index': 'ds'})
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    print(df_combined)
    plot_save_chart_with_confidence_level(df_combined, "AutoTheta", metric)
    print(f"Forecasting with AutoTheta for {metric}")
    return df_combined

def AutoETS_forecast(metric, ts):
    autoets = AutoETS(season_length=24)
    autoets = autoets.fit(y=ts['y'].to_numpy())
    y_hat_dict = autoets.predict(h=24, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = 24 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level_df(y_hat_dict, future_dates, "AutoETS")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    plot_save_chart_with_confidence_level(df_combined, "AutoETS", metric)
    print(df_combined)
    print(f"Forecasting with AutoETS for {metric}")
    return df_combined

def CES_forecast(metric, ts):
    ces = AutoCES(season_length=24)
    ces = ces.fit(y=ts['y'].to_numpy())
    y_hat_dict = ces.predict(h=24, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = 24 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level_df(y_hat_dict, future_dates, "CES")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    plot_save_chart_with_confidence_level(df_combined, "CES", metric)
    print(df_combined)
    print(f"Forecasting with CES for {metric}")
    return df_combined

def MSTL_forecast(metric, ts):
    mstl_model = MSTL(season_length=24)
    mstl_model = mstl_model.fit(y=ts['y'].to_numpy())
    y_hat_dict = mstl_model.predict(h=24, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = 24 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level_df(y_hat_dict, future_dates, "MSTL")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    plot_save_chart_with_confidence_level(df_combined, "MSTL", metric)
    print(df_combined)
    print(f"Forecasting with MSTL for {metric}")
    return df_combined

def SeasonalNaive_forecast(metric, ts):
    model = SeasonalNaive(season_length=24)
    model = model.fit(y=ts['y'].to_numpy())
    y_hat_dict = model.predict(h=24, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = 24
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level_df(y_hat_dict, future_dates, "SeasonalNaive")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    plot_save_chart_with_confidence_level(df_combined, "SeasonalNaive", metric)
    print(df_combined)
    print(f"Forecasting with SeasonalNaive for {metric}")
    return df_combined

def WindowAverage_forecast(metric, ts):
    model = WindowAverage(window_size=24)
    model = model.fit(y=ts['y'])
    y_hat_dict = model.predict(h=24)
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = 24 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction(y_hat_dict, future_dates, "WindowAverage")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    plot_save_chart(df_combined, "WindowAverage", metric)
    print(f"Forecasting with WindowAverage for {metric}")
    return df_combined

def SeasWA_forecast(metric, ts):
    model = SeasonalWindowAverage(window_size=1, season_length=24)
    model = model.fit(y=ts['y'].to_numpy())
    y_hat_dict = model.predict(h=24)
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = 24 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction(y_hat_dict, future_dates, "SeasWA")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    plot_save_chart(df_combined, "SeasWA", metric)
    print(df_combined)
    print(f"Forecasting with SeasWA for {metric}")
    return df_combined

def Naive_forecast(metric, ts):
    model = Naive()
    model = model.fit(y=ts['y'].to_numpy())
    y_hat_dict = model.predict(h=24, level=[80])
    print(y_hat_dict)
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = 24 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level_df(y_hat_dict, future_dates, "Naive")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    plot_save_chart_with_confidence_level(df_combined, "Naive", metric)
    print(df_combined)
    print(f"Forecasting with Naive for {metric}")
    return df_combined

def prophet_forecast(metric, ts):
    model = Prophet()
    model.fit(ts)
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    print(forecast)
    # Concatenate the fit and preicted datasets
    df_combined = pd.merge(ts, forecast, on='ds', how='outer')
    plot_save_chart_prophet(df_combined, metric)
    print(df_combined)
    print(f"Forecasting with Prophet for {metric}")
    return df_combined

# Main script
def main(pod_name):
    metrics = ["CPU", "RAM"]  # List of metrics to append to the pod name

    # Path to your SQLite database
    db_path = 'CrossValidationJobs/cross-validation.db'
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for metric in metrics:
        ts = pd.DataFrame()
        if (metric == "CPU"):
            ts = get_container_cpu(pod_name)
        elif (metric == "RAM"):
            ts = get_container_memory(pod_name)

        #Resample to hours
        print("Resampling dataset to hours\n")
        ts['ds'] = pd.to_datetime(ts ['ds'])
        ts.set_index('ds', inplace=True)
        ts = ts.resample('H').mean()
        ts = ts.fillna(0)  # Fills NaN with 0
        ts = ts.reset_index()

        ts = ts.sort_values(by='ds').tail(7 * 24)

        print(ts['y'])

        # Modify pod_name to include the metric
        modified_pod_name = f"{pod_name}-{metric}"

        # Query the database for the modified pod name
        cursor.execute("SELECT best_model FROM evaluate_cross_validation WHERE unique_id=?", (modified_pod_name,))
        result = cursor.fetchone()
        result = ["prophet"]

        if result:
            best_model = result[0]
            # Switch case equivalent using dictionary mapping
            forecast_function = {
                "AutoARIMA": lambda: AutoARIMA_forecast(metric, ts),
                "AutoTheta": lambda: AutoTheta_forecast(metric, ts),
                "AutoETS": lambda: AutoETS_forecast(metric, ts),
                "CES": lambda: CES_forecast(metric, ts),
                "MSTL": lambda: MSTL_forecast(metric, ts),
                "SeasonalNaive": lambda: SeasonalNaive_forecast(metric, ts),
                "WindowAverage": lambda: WindowAverage_forecast(metric, ts),
                "SeasWA": lambda: SeasWA_forecast(metric, ts),
                "Naive": lambda: Naive_forecast(metric, ts),
                "prophet": lambda: prophet_forecast(metric, ts)
            }.get(best_model, lambda: f"Error: No forecasting function for model {best_model}")

            # Call the forecasting function
            df_forecast = forecast_function()
            #Check if is a prophet dataset
            if(df_forecast.columns.tolist()[2] == 'trend'):
                #if yes normalize the dataset column names and number
                rename_dict = {
                    'yhat': 'forecast',
                    'yhat_lower': 'forecast-lo-80',
                    'yhat_upper': 'forecast-hi-80'
                }
                df_forecast = df_forecast.rename(columns=rename_dict)
                df_forecast = df_forecast[['ds', 'y', 'forecast', 'forecast-lo-80', 'forecast-hi-80']]
                print(df_forecast)
            if (metric == "CPU"):
                cpu_cost_hour = get_cpu_hourly_cost("kube2")
                print("CPU hourly cost: ", cpu_cost_hour)
                # Multiply non-NaN values by cpu_price for specified columns
                df_forecast_cpu_cost = df_forecast.apply(lambda x: x * float(cpu_cost_hour) if x.name != 'ds' else x)
                cpu_forecast_columns = df_forecast.columns.tolist()
            elif (metric == "RAM"):
                ram_cost_hour = get_ram_hourly_cost("kube2")
                print("RAM hourly cost: ", ram_cost_hour)
                # Multiply non-NaN values by cpu_price for specified columns
                df_forecast_ram_cost = df_forecast.apply(lambda x: x * float(ram_cost_hour) if x.name != 'ds' else x)
                ram_forecast_columns = df_forecast.columns.tolist()
        else:
            print(f"Error: No matching pod found for {modified_pod_name}")

    cpu_forecast_model = cpu_forecast_columns[2]
    ram_forecast_model = ram_forecast_columns[2]
    models_no_confidence_interval = ['WindowAverage', 'SeasWA']

    if (cpu_forecast_model in models_no_confidence_interval and ram_forecast_model in models_no_confidence_interval):
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        print(df_sum_updated)
        plot_save_chart_cost_with_confidence_level(df_sum_updated,  cpu_forecast_model, ram_forecast_model)
    elif (cpu_forecast_model not in models_no_confidence_interval and ram_forecast_model in models_no_confidence_interval):
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        df_sum_updated['forecast-lo-80'] = df_forecast_cpu_cost.iloc[:, 3]  + df_forecast_ram_cost.iloc[:, 2] 
        df_sum_updated['forecast-hi-80'] = df_forecast_cpu_cost.iloc[:, 4]  + df_forecast_ram_cost.iloc[:, 2] 
        print(df_sum_updated)
        plot_save_chart_cost_with_confidence_level(df_sum_updated,  cpu_forecast_model, ram_forecast_model)
    elif (cpu_forecast_model in models_no_confidence_interval and ram_forecast_model not in models_no_confidence_interval):
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        df_sum_updated['forecast-lo-80'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 3] 
        df_sum_updated['forecast-hi-80'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 4]
        print(df_sum_updated)
        plot_save_chart_cost_with_confidence_level(df_sum_updated,  cpu_forecast_model, ram_forecast_model)
    else:
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        df_sum_updated['forecast-lo-80'] = df_forecast_cpu_cost.iloc[:, 3]  + df_forecast_ram_cost.iloc[:, 3] 
        df_sum_updated['forecast-hi-80'] = df_forecast_cpu_cost.iloc[:, 4]  + df_forecast_ram_cost.iloc[:, 4] 
        print(df_sum_updated)
        plot_save_chart_cost_with_confidence_level(df_sum_updated,  cpu_forecast_model, ram_forecast_model)

    # Calculate the sum of each column after multiplication, excluding NaN values
    # Note: This retains 'ds' for plotting, but it's not included in the sum calculation
    column_sums = df_sum_updated[df_sum_updated.columns[df_sum_updated.columns != 'ds']].sum()

    

    print("CPU forecast model: ", cpu_forecast_model)
    print("RAM forecast model: ", ram_forecast_model)
    print("\nColumn sums after multiplication:\n", column_sums)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <pod_name>")
        sys.exit(1)
    
    pod_name = sys.argv[1]
    main(pod_name)


