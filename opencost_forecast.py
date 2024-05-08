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
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

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

from kubernetes import client, config



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



def get_running_kubernetes_pods(namespace):
    # Load the kubeconfig file
    config.load_kube_config()

    # Create a v1 Core API object
    v1 = client.CoreV1Api()

    print(f"Listing running pods in namespace '{namespace}':")

    # List pods in the specified namespace
    pod_list = v1.list_namespaced_pod(namespace)

    running_pods = []
    for pod in pod_list.items:
        # Check if the pod's status is 'Running'
        if pod.status.phase == "Running":
            # Append a tuple of the pod name and the node it is running on to the list
            running_pods.append((pod.metadata.name, pod.spec.node_name))
            print(f"Pod: {pod.metadata.name}, Node: {pod.spec.node_name}")

    return running_pods



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

def plot_save_chart_with_confidence_level(df, model, metric, forecast_period):
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
    fig.update_layout(title='Historical Data and ' + str(forecast_period) + 'H Predictions with Confidence Interval for ' + metric + ' predicted with ' + model,
                    xaxis_title='Date',
                    yaxis_title='Value',
                    showlegend=True)

    # Show plot
    #fig.show()

    # Save plot to file
    #fig.write_image(metric+"-historical_predictions_plot.png", width=800, height=600)
    return fig

def plot_save_chart_prophet(df, metric, forecast_period):
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
    fig.update_layout(title='Historical Data and ' + str(forecast_period) + 'H Predictions with Confidence Interval for ' + metric + ' predicted with Prophet',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    showlegend=True)

    # Show plot
    #fig.show()

    # Save plot to file
    #fig.write_image(metric+"-historical_predictions_plot.png", width=800, height=600)
    return fig


def plot_save_chart(df, model, metric, forecast_period):
    # Plotting
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical Data', line=dict(color='blue')))

    # Predicted values
    fig.add_trace(go.Scatter(x=df['ds'], y=df[model], mode='lines', name='Predicted', line=dict(color='red')))

   
    # Layout adjustments
    fig.update_layout(title='Historical Data and ' + str(forecast_period) + 'H Predictions with Confidence Interval for ' + metric + ' predicted with ' + model,
                    xaxis_title='Date',
                    yaxis_title='Value',
                    showlegend=True)

    # Show plot
    #fig.show()

    # Save plot to file
    #fig.write_image(metric+"-historical_predictions_plot.png", width=800, height=600)
    return fig

def plot_save_chart_cost_with_confidence_level(df, cpu_model, ram_model, forecast_period):
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
    fig.update_layout(title=str(forecast_period) + 'H Cost Predictions with Confidence Interval for CPU predicted with ' + cpu_model + ' and RAM predicted with ' + ram_model,
                    xaxis_title='Date',
                    yaxis_title='Value',
                    showlegend=True)

    # Show plot
    #fig.show()

    # Save plot to file
    #fig.write_image("cost_predictions_plot.png", width=800, height=600)
    return fig

def plot_save_chart_cost_withouth_confidence_level(df, cpu_model, ram_model, forecast_period):
    # Plotting
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical Data', line=dict(color='blue')))

    # Predicted values
    fig.add_trace(go.Scatter(x=df['ds'], y=df['forecast'], mode='lines', name='Predicted', line=dict(color='red')))


    # Layout adjustments
    fig.update_layout(title=str(forecast_period) + 'H Cost Predictions with Confidence Interval for CPU predicted with ' + cpu_model + ' and RAM predicted with ' + ram_model,
                    xaxis_title='Date',
                    yaxis_title='Value',
                    showlegend=True)

    # Show plot
    #fig.show()

    # Save plot to file
    #fig.write_image("cost_predictions_plot.png", width=800, height=600)
    return fig

# Function definitions for each forecasting model
def AutoARIMA_forecast(metric, ts, forecast_period):
    arima = AutoARIMA(season_length=24)
    arima = arima.fit(y=ts['y'].to_numpy())
    y_hat_dict = arima.predict(h=forecast_period, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = forecast_period 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level(y_hat_dict, future_dates, "AutoARIMA")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    print(df_combined)
    autoarima_plot = plot_save_chart_with_confidence_level(df_combined, "AutoARIMA", metric, forecast_period)
    print(f"Forecasting with AutoARIMA for {metric}")
    return df_combined, autoarima_plot

def AutoTheta_forecast(metric, ts, forecast_period):
    theta = AutoTheta(season_length=24)
    theta = theta.fit(y=ts['y'].to_numpy())
    y_hat_dict = theta.predict(h=forecast_period, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = forecast_period 
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
    autotheta_plot = plot_save_chart_with_confidence_level(df_combined, "AutoTheta", metric, forecast_period)
    print(f"Forecasting with AutoTheta for {metric}")
    return df_combined, autotheta_plot

def AutoETS_forecast(metric, ts, forecast_period):
    autoets = AutoETS(season_length=24) 
    autoets = autoets.fit(y=ts['y'].to_numpy())
    y_hat_dict = autoets.predict(h=forecast_period, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = forecast_period 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level_df(y_hat_dict, future_dates, "AutoETS")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    auto_ets_plot = plot_save_chart_with_confidence_level(df_combined, "AutoETS", metric, forecast_period)
    print(df_combined)
    print(f"Forecasting with AutoETS for {metric}")
    return df_combined, auto_ets_plot

def CES_forecast(metric, ts, forecast_period):
    ces = AutoCES(season_length=24)
    ces = ces.fit(y=ts['y'].to_numpy())
    y_hat_dict = ces.predict(h=forecast_period, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = forecast_period 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level_df(y_hat_dict, future_dates, "CES")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    ces_plot = plot_save_chart_with_confidence_level(df_combined, "CES", metric, forecast_period)
    print(df_combined)
    print(f"Forecasting with CES for {metric}")
    return df_combined, ces_plot

def MSTL_forecast(metric, ts, forecast_period):
    mstl_model = MSTL(season_length=24)
    mstl_model = mstl_model.fit(y=ts['y'].to_numpy())
    y_hat_dict = mstl_model.predict(h=forecast_period, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = forecast_period 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level_df(y_hat_dict, future_dates, "MSTL")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    mstl_plot = plot_save_chart_with_confidence_level(df_combined, "MSTL", metric, forecast_period)
    print(df_combined)
    print(f"Forecasting with MSTL for {metric}")
    return df_combined, mstl_plot

def SeasonalNaive_forecast(metric, ts, forecast_period):
    model = SeasonalNaive(season_length=24)
    model = model.fit(y=ts['y'].to_numpy())
    y_hat_dict = model.predict(h=forecast_period, level=[80])
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = forecast_period
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level_df(y_hat_dict, future_dates, "SeasonalNaive")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    seasonalnaive_plot = plot_save_chart_with_confidence_level(df_combined, "SeasonalNaive", metric, forecast_period)
    print(df_combined)
    print(f"Forecasting with SeasonalNaive for {metric}")
    return df_combined, seasonalnaive_plot

def WindowAverage_forecast(metric, ts, forecast_period):
    model = WindowAverage(window_size=24)
    model = model.fit(y=ts['y'])
    y_hat_dict = model.predict(h=forecast_period)
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = forecast_period 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction(y_hat_dict, future_dates, "WindowAverage")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    windowaverage_plot = plot_save_chart(df_combined, "WindowAverage", metric, forecast_period)
    print(f"Forecasting with WindowAverage for {metric}")
    return df_combined, windowaverage_plot

def SeasWA_forecast(metric, ts, forecast_period):
    model = SeasonalWindowAverage(window_size=1, season_length=24)
    model = model.fit(y=ts['y'].to_numpy())
    y_hat_dict = model.predict(h=forecast_period)
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = forecast_period 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction(y_hat_dict, future_dates, "SeasWA")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    seaswa_plot = plot_save_chart(df_combined, "SeasWA", metric, forecast_period)
    print(df_combined)
    print(f"Forecasting with SeasWA for {metric}")
    return df_combined, seaswa_plot

def Naive_forecast(metric, ts, forecast_period):
    model = Naive()
    model = model.fit(y=ts['y'].to_numpy())
    y_hat_dict = model.predict(h=forecast_period, level=[80])
    print(y_hat_dict)
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = forecast_period 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction_confidence_level_df(y_hat_dict, future_dates, "Naive")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    naive_plot = plot_save_chart_with_confidence_level(df_combined, "Naive", metric, forecast_period)
    print(df_combined)
    print(f"Forecasting with Naive for {metric}")
    return df_combined, naive_plot

def prophet_forecast(metric, ts, forecast_period):
    model = Prophet()
    model.fit(ts)
    future = model.make_future_dataframe(periods=forecast_period, freq='H')
    forecast = model.predict(future)
    print(forecast)
    # Determine the maximum date in your training data
    last_history_date = ts['ds'].max()
    # Assign NaN to predictions on historical dates
    forecast.loc[forecast['ds'] <= last_history_date, ['yhat', 'yhat_lower', 'yhat_upper']] = np.nan
    # Concatenate the fit and preicted datasets
    df_combined = pd.merge(ts, forecast, on='ds', how='outer')
    prophet_plot = plot_save_chart_prophet(df_combined, metric, forecast_period)
    print(df_combined)
    print(f"Forecasting with Prophet for {metric}")
    return df_combined, prophet_plot

# 
def forecast_one_day(pod_name, node):
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
        #if metric == "CPU":
        #    result = ["SeasonalNaive"]
        #else:
        #    result = ["SeasonalNaive"]

        if result:
            best_model = result[0]
            # Switch case equivalent using dictionary mapping
            forecast_function = {
                "AutoARIMA": lambda: AutoARIMA_forecast(metric, ts, 24),
                "AutoTheta": lambda: AutoTheta_forecast(metric, ts, 24),
                "AutoETS": lambda: AutoETS_forecast(metric, ts, 24),
                "CES": lambda: CES_forecast(metric, ts, 24),
                "MSTL": lambda: MSTL_forecast(metric, ts, 24),
                "SeasonalNaive": lambda: SeasonalNaive_forecast(metric, ts, 24),
                "WindowAverage": lambda: WindowAverage_forecast(metric, ts, 24),
                "SeasWA": lambda: SeasWA_forecast(metric, ts, 24),
                "Naive": lambda: Naive_forecast(metric, ts, 24),
                "prophet": lambda: prophet_forecast(metric, ts, 24)
            }.get(best_model, lambda: f"Error: No forecasting function for model {best_model}")

            # Call the forecasting function
            df_forecast, plot = forecast_function()
            #Check if is a prophet dataset
            if(df_forecast.columns.tolist()[2] == 'trend'):
                #if yes normalize the dataset column names and number
                rename_dict = {
                    'yhat': 'forecastP',
                    'yhat_lower': 'forecastP-lo-80',
                    'yhat_upper': 'forecastP-hi-80'
                }
                df_forecast = df_forecast.rename(columns=rename_dict)
                df_forecast = df_forecast[['ds', 'y', 'forecastP', 'forecastP-lo-80', 'forecastP-hi-80']]
                print(df_forecast)
            if (metric == "CPU"):
                cpu_cost_hour = get_cpu_hourly_cost(node)
                print("CPU hourly cost: ", cpu_cost_hour)
                # Multiply non-NaN values by cpu_price for specified columns
                df_forecast_cpu_cost = df_forecast.apply(lambda x: x * float(cpu_cost_hour) if x.name != 'ds' else x)
                cpu_forecast_columns = df_forecast.columns.tolist()
                cpu_plot = plot
            elif (metric == "RAM"):
                ram_cost_hour = get_ram_hourly_cost(node)
                print("RAM hourly cost: ", ram_cost_hour)
                # Multiply non-NaN values by cpu_price for specified columns
                df_forecast_ram_cost = df_forecast.apply(lambda x: x * float(ram_cost_hour) if x.name != 'ds' else x)
                ram_forecast_columns = df_forecast.columns.tolist()
                ram_plot = plot
        else:
            print(f"Error: No matching pod found for {modified_pod_name}")

    cpu_forecast_model = cpu_forecast_columns[2] if cpu_forecast_columns[2] != "forecastP" else "Prophet"
    ram_forecast_model = ram_forecast_columns[2] if ram_forecast_columns[2] != "forecastP" else "Prophet"
    models_no_confidence_interval = ['WindowAverage', 'SeasWA']

    if (cpu_forecast_model in models_no_confidence_interval and ram_forecast_model in models_no_confidence_interval):
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        print("Sum CPU and Mem cost 1")
        print(df_sum_updated)
        cost_forecast_plot = plot_save_chart_cost_withouth_confidence_level(df_sum_updated,  cpu_forecast_model, ram_forecast_model, 24)
    elif (cpu_forecast_model not in models_no_confidence_interval and ram_forecast_model in models_no_confidence_interval):
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        df_sum_updated['forecast-lo-80'] = df_forecast_cpu_cost.iloc[:, 3]  + df_forecast_ram_cost.iloc[:, 2] 
        df_sum_updated['forecast-hi-80'] = df_forecast_cpu_cost.iloc[:, 4]  + df_forecast_ram_cost.iloc[:, 2] 
        print("Sum CPU and Mem cost 2")
        print(df_sum_updated)
        cost_forecast_plot = plot_save_chart_cost_withouth_confidence_level(df_sum_updated,  cpu_forecast_model, ram_forecast_model, 24)
    elif (cpu_forecast_model in models_no_confidence_interval and ram_forecast_model not in models_no_confidence_interval):
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        df_sum_updated['forecast-lo-80'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 3] 
        df_sum_updated['forecast-hi-80'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 4]
        print("Sum CPU and Mem cost 3")
        print(df_sum_updated)
        cost_forecast_plot = plot_save_chart_cost_withouth_confidence_level(df_sum_updated,  cpu_forecast_model, ram_forecast_model, 24)
    else:
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        df_sum_updated['forecast-lo-80'] = df_forecast_cpu_cost.iloc[:, 3]  + df_forecast_ram_cost.iloc[:, 3] 
        df_sum_updated['forecast-hi-80'] = df_forecast_cpu_cost.iloc[:, 4]  + df_forecast_ram_cost.iloc[:, 4] 
        print("Sum CPU and Mem cost 4")
        print(df_sum_updated)
        cost_forecast_plot = plot_save_chart_cost_with_confidence_level(df_sum_updated, cpu_forecast_model, ram_forecast_model, 24)

    # Calculate the sum of each column after multiplication, excluding NaN values
    # Note: This retains 'ds' for plotting, but it's not included in the sum calculation
    column_sums = df_sum_updated[df_sum_updated.columns[df_sum_updated.columns != 'ds']].sum()

    

    print("CPU forecast model: ", cpu_forecast_model)
    print("RAM forecast model: ", ram_forecast_model)
    print("\nColumn sums after multiplication:\n", column_sums)

    return cost_forecast_plot, cpu_plot, ram_plot, column_sums


def forecast_one_week(pod_name, node):
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

        ts = ts.sort_values(by='ds').tail(19 * 24)

        print(ts['y'])

        # Modify pod_name to include the metric
        modified_pod_name = f"{pod_name}-{metric}"

        # Query the database for the modified pod name
        cursor.execute("SELECT best_model FROM evaluate_cross_validation_one_week WHERE unique_id=?", (modified_pod_name,))
        result = cursor.fetchone()
        # if(metric == "CPU"):
        #     result = ["AutoETS"]
        # else:
        #     result = ["AutoARIMA"]

        if result:
            best_model = result[0]
            # Switch case equivalent using dictionary mapping
            forecast_function = {
                "AutoARIMA": lambda: AutoARIMA_forecast(metric, ts, 168),
                "AutoTheta": lambda: AutoTheta_forecast(metric, ts, 168),
                "AutoETS": lambda: AutoETS_forecast(metric, ts, 168),
                "CES": lambda: CES_forecast(metric, ts, 168),
                "MSTL": lambda: MSTL_forecast(metric, ts, 168),
                "SeasonalNaive": lambda: SeasonalNaive_forecast(metric, ts, 168),
                "WindowAverage": lambda: WindowAverage_forecast(metric, ts, 168),
                "SeasWA": lambda: SeasWA_forecast(metric, ts, 168),
                "Naive": lambda: Naive_forecast(metric, ts, 168),
                "prophet": lambda: prophet_forecast(metric, ts, 168)
            }.get(best_model, lambda: f"Error: No forecasting function for model {best_model}")

            # Call the forecasting function
            df_forecast, plot = forecast_function()
            #Check if is a prophet dataset
            if(df_forecast.columns.tolist()[2] == 'trend'):
                #if yes normalize the dataset column names and number
                rename_dict = {
                    'yhat': 'forecastP',
                    'yhat_lower': 'forecastP-lo-80',
                    'yhat_upper': 'forecastP-hi-80'
                }
                df_forecast = df_forecast.rename(columns=rename_dict)
                df_forecast = df_forecast[['ds', 'y', 'forecastP', 'forecastP-lo-80', 'forecastP-hi-80']]
                print(df_forecast)
            if (metric == "CPU"):
                cpu_cost_hour = get_cpu_hourly_cost(node)
                print("CPU hourly cost: ", cpu_cost_hour)
                # Multiply non-NaN values by cpu_price for specified columns
                df_forecast_cpu_cost = df_forecast.apply(lambda x: x * float(cpu_cost_hour) if x.name != 'ds' else x)
                cpu_forecast_columns = df_forecast.columns.tolist()
                cpu_plot = plot
            elif (metric == "RAM"):
                ram_cost_hour = get_ram_hourly_cost(node)
                print("RAM hourly cost: ", ram_cost_hour)
                # Multiply non-NaN values by cpu_price for specified columns
                df_forecast_ram_cost = df_forecast.apply(lambda x: x * float(ram_cost_hour) if x.name != 'ds' else x)
                ram_forecast_columns = df_forecast.columns.tolist()
                ram_plot = plot
        else:
            print(f"Error: No matching pod found for {modified_pod_name}")

    cpu_forecast_model = cpu_forecast_columns[2] if cpu_forecast_columns[2] != "forecastP" else "Prophet"
    ram_forecast_model = ram_forecast_columns[2] if ram_forecast_columns[2] != "forecastP" else "Prophet"
    models_no_confidence_interval = ['WindowAverage', 'SeasWA']

    if (cpu_forecast_model in models_no_confidence_interval and ram_forecast_model in models_no_confidence_interval):
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        print(df_sum_updated)
        cost_forecast_plot = plot_save_chart_cost_with_confidence_level(df_sum_updated,  cpu_forecast_model, ram_forecast_model, 168)
    elif (cpu_forecast_model not in models_no_confidence_interval and ram_forecast_model in models_no_confidence_interval):
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        df_sum_updated['forecast-lo-80'] = df_forecast_cpu_cost.iloc[:, 3]  + df_forecast_ram_cost.iloc[:, 2] 
        df_sum_updated['forecast-hi-80'] = df_forecast_cpu_cost.iloc[:, 4]  + df_forecast_ram_cost.iloc[:, 2] 
        print(df_sum_updated)
        cost_forecast_plot = plot_save_chart_cost_with_confidence_level(df_sum_updated,  cpu_forecast_model, ram_forecast_model, 168)
    elif (cpu_forecast_model in models_no_confidence_interval and ram_forecast_model not in models_no_confidence_interval):
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        df_sum_updated['forecast-lo-80'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 3] 
        df_sum_updated['forecast-hi-80'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 4]
        print(df_sum_updated)
        cost_forecast_plot = plot_save_chart_cost_with_confidence_level(df_sum_updated,  cpu_forecast_model, ram_forecast_model, 168)
    else:
        df_sum_updated = pd.DataFrame()
        df_sum_updated['ds'] = df_forecast_cpu_cost['ds']
        df_sum_updated['y'] = df_forecast_cpu_cost['y'] + df_forecast_ram_cost['y']
        df_sum_updated['forecast'] = df_forecast_cpu_cost.iloc[:, 2]  + df_forecast_ram_cost.iloc[:, 2]
        df_sum_updated['forecast-lo-80'] = df_forecast_cpu_cost.iloc[:, 3]  + df_forecast_ram_cost.iloc[:, 3] 
        df_sum_updated['forecast-hi-80'] = df_forecast_cpu_cost.iloc[:, 4]  + df_forecast_ram_cost.iloc[:, 4] 
        print(df_sum_updated)
        cost_forecast_plot = plot_save_chart_cost_with_confidence_level(df_sum_updated, cpu_forecast_model, ram_forecast_model, 168)

    # Calculate the sum of each column after multiplication, excluding NaN values
    # Note: This retains 'ds' for plotting, but it's not included in the sum calculation
    column_sums = df_sum_updated[df_sum_updated.columns[df_sum_updated.columns != 'ds']].sum()

    

    print("CPU forecast model: ", cpu_forecast_model)
    print("RAM forecast model: ", ram_forecast_model)
    print("\nColumn sums after multiplication:\n", column_sums)

    return cost_forecast_plot, cpu_plot, ram_plot, column_sums

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    # Dropdown for selecting a POD
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='pod-dropdown',
                placeholder='Select a POD'
            ), width=6
        ),
        dbc.Col(
            dcc.Dropdown(
                id='period-dropdown',
                options=[
                    {'label': 'One day', 'value': '1d'},
                    {'label': 'One week', 'value': '1w'}
                ],
                placeholder='Select a Period'
            ), width=6
        )
    ]),
    dbc.Button('Start Forecast', id='forecast-button', n_clicks=0, className='mt-3'),
    html.Div(id='charts-container'),
    html.Div(id='text-output', style={'white-space': 'pre-line'})
])

@app.callback(
    Output('pod-dropdown', 'options'),
    Input('pod-dropdown', 'placeholder')  # A trick to trigger callback on load
)
def update_pod_dropdown(_):
    namespace = 'stress-test'
    running_pods = get_running_kubernetes_pods(namespace)
    return [{'label': pod[0], 'value': pod[0] + '--' + pod[1]} for pod in running_pods]

@app.callback(
    [Output('charts-container', 'children'),
     Output('text-output', 'children')],
    Input('forecast-button', 'n_clicks'),
    State('pod-dropdown', 'value'),
    State('period-dropdown', 'value')
)
def update_output(n_clicks, selected_pod, selected_period):
    if n_clicks > 0 and selected_period == '1d':
        pod_and_node = selected_pod.split('--')
        cost_forecast_plot, cpu_plot, ram_plot, calcultated_cost_over_period = forecast_one_day(pod_and_node[0], pod_and_node[1])
        text_output = "Forecast started...\nPOD: {}\nPeriod: {}\n Total cost forecast for the upcoming period: {}\n".format(selected_pod, selected_period, calcultated_cost_over_period[['forecast', 'forecast-lo-80', 'forecast-hi-80']])

        return [
            dcc.Graph(figure=cost_forecast_plot),
            dcc.Graph(figure=cpu_plot),
            dcc.Graph(figure=ram_plot)
        ], text_output
    elif n_clicks > 0 and selected_period == '1w':
        pod_and_node = selected_pod.split('--')
        cost_forecast_plot, cpu_plot, ram_plot, calcultated_cost_over_period = forecast_one_week(pod_and_node[0], pod_and_node[1])
        text_output = "Forecast started...\nPOD: {}\nPeriod: {}\n Total cost forecast for the upcoming period: {}\n".format(selected_pod, selected_period, calcultated_cost_over_period[['forecast', 'forecast-lo-80', 'forecast-hi-80']])

        return [
            dcc.Graph(figure=cost_forecast_plot),
            dcc.Graph(figure=cpu_plot),
            dcc.Graph(figure=ram_plot)
        ], text_output

    return [], ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

