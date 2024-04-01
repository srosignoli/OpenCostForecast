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
import pandas as pd
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
    return mean_series

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
    return f"Forecasting with AutoARIMA for {metric}"

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
    return f"Forecasting with AutoTheta for {metric}"

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
    print(df_combined)
    
    return f"Forecasting with AutoETS for {metric}"

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
    print(df_combined)
    return f"Forecasting with CES for {metric}"

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
    print(df_combined)
    return f"Forecasting with MSTL for {metric}"

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
    print(df_combined)
    return f"Forecasting with SeasonalNaive for {metric}"

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
    print(df_combined)
    return f"Forecasting with WindowAverage for {metric}"

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
    print(df_combined)
    return f"Forecasting with SeasWA for {metric}"

def Naive_forecast(metric, ts):
    model = Naive()
    model = model.fit(y=ts['y'].to_numpy())
    y_hat_dict = model.predict(h=24)
    last_date = ts['ds'].iloc[-1]
    # Number of hours to forecast
    forecast_horizon = 24 
    future_dates = future_date(last_date, forecast_horizon)
    predictions_df = add_future_dates_prediction(y_hat_dict, future_dates, "Naive")
    print(predictions_df)
    # Concatenate the fit and preicted datasets
    df_combined = pd.concat([ts, predictions_df])
    df_combined.sort_values(by='ds', inplace=True)
    # Reset the index of the combined DataFrame
    df_combined.reset_index(drop=True, inplace=True)
    print(df_combined)
    return f"Forecasting with Naive for {metric}"

def prophet_forecast(metric, ts):
    model = Prophet()
    model.fit(ts)
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    print(forecast)
    # Concatenate the fit and preicted datasets
    df_combined = pd.merge(ts, forecast, on='ds', how='outer')
    print(df_combined)
    return f"Forecasting with Prophet for {metric}"

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
        #result = ["prophet"]

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
            print(forecast_function())
        else:
            print(f"Error: No matching pod found for {modified_pod_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <pod_name>")
        sys.exit(1)
    
    pod_name = sys.argv[1]
    main(pod_name)


