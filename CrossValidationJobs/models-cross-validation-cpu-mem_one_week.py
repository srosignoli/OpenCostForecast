#############################################################################
## Models cross validation for containers CPU and Memory measurements retrieved 
## from Prometheus.
## 
#############################################################################
## Author: Simone Rosignoli
## 
##  
#############################################################################

import requests
import pandas as pd
import os
import math
import sqlite3
from statsforecast import StatsForecast
from utilsforecast.losses import mse
from utilsforecast.evaluation import evaluate
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from datetime import datetime, timedelta
from kubernetes import client, config

# Prometheus server details
PROMETHEUS = 'http://localhost:9001'


def get_containers_memory(pod_list):
    df_containers = pd.DataFrame()
    for pod in pod_list:
        QUERY = f'container_memory_allocation_bytes{{pod="{pod}"}}/1024/1024/1024'
        # Calculate start and end times for the last hour
        END = datetime.now()
        START = END - timedelta(days=19)
        # Convert times to UNIX timestamps
        start_time = START.timestamp()
        end_time = END.timestamp()

        # Construct the query
        query_range_url = f"{PROMETHEUS}/api/v1/query_range"
        params = {
            'query': QUERY,
            'start': start_time,
            'end': end_time,
            'step': '60m'  # 60 seconds intervals
        }

        # Make the HTTP request to Prometheus
        response = requests.get(query_range_url, params=params)
        # Parse the JSON response
        print(response)
        data = response.json()['data']['result']

        # Initialize an empty list to hold the data points
        data_points = []

        pod_id = pod + "-RAM"
        # Extract the data points
        for result in data:
            for value in result['values']:
                # Convert timestamp to readable date
                timestamp = datetime.fromtimestamp(float(value[0]))
                memory_usage = float(value[1])
                data_points.append((timestamp, memory_usage, pod_id))

        # Create a DataFrame
        df = pd.DataFrame(data_points, columns=['Timestamp', 'MemoryUsageGB', 'unique_id'])
        df_containers = pd.concat([df_containers, df], ignore_index=True)

    return df_containers

def get_containers_cpu(pod_list):
    df_containers = pd.DataFrame()
    for pod in pod_list:
        QUERY = f'container_cpu_allocation{{pod="{pod}"}}'
        # Calculate start and end times for the last hour
        END = datetime.now()
        START = END - timedelta(days=19)
        # Convert times to UNIX timestamps
        start_time = START.timestamp()
        end_time = END.timestamp()

        # Construct the query
        query_range_url = f"{PROMETHEUS}/api/v1/query_range"
        params = {
            'query': QUERY,
            'start': start_time,
            'end': end_time,
            'step': '60m'  # 60 seconds intervals
        }

        # Make the HTTP request to Prometheus
        response = requests.get(query_range_url, params=params)
        # Parse the JSON response
        data = response.json()['data']['result']

        # Initialize an empty list to hold the data points
        data_points = []

        pod_id = pod + "-CPU"
        # Extract the data points
        for result in data:
            for value in result['values']:
                # Convert timestamp to readable date
                timestamp = datetime.fromtimestamp(float(value[0]))
                memory_usage = float(value[1])
                data_points.append((timestamp, memory_usage, pod_id))

        # Create a DataFrame
        df = pd.DataFrame(data_points, columns=['Timestamp', 'CPUUsage', 'unique_id'])
        df_containers = pd.concat([df_containers, df], ignore_index=True)

    return df_containers

def get_kubernetes_pod_list(namespace):
    # Load the kubeconfig file
    config.load_kube_config()

    # Create a v1 Core API object
    v1 = client.CoreV1Api()

    print(f"Listing pods in namespace '{namespace}':")

    # List pods in the specified namespace
    pod_list = v1.list_namespaced_pod(namespace)

    pod_name_list = []
    for pod in pod_list.items:
        if pod.status.phase == "Running":
            pod_name_list.append(pod.metadata.name)
            print(pod.metadata.name)

    return pod_name_list

print("Retrieving Kubernetes pod list\n")
pod_list = get_kubernetes_pod_list("stress-test")
print("Retrieving containers memory metrics from Prometheus\n")    
memory_df = get_containers_memory(pod_list)
print(memory_df)
print("Retrieving containers CPU metrics from Prometheus\n")  
cpu_df = get_containers_cpu(pod_list)
print(cpu_df)

memory_df = memory_df.rename(columns={'MemoryUsageGB': 'y', 'Timestamp': 'ds'})
cpu_df = cpu_df.rename(columns={'CPUUsage': 'y', 'Timestamp': 'ds'})

cpu_memory_df = pd.concat([memory_df, cpu_df], ignore_index=True)
print(cpu_memory_df)

#Resample to hours
print("Resampling dataset to hours\n")
cpu_memory_df['ds'] = pd.to_datetime(cpu_memory_df['ds'])
cpu_memory_df.set_index('ds', inplace=True)
cpu_memory_df = cpu_memory_df.groupby('unique_id').resample('H').mean()
cpu_memory_df = cpu_memory_df.fillna(0)  # Fills NaN with 0
cpu_memory_df = cpu_memory_df.reset_index()

print(cpu_memory_df)

cpu_memory_df = cpu_memory_df.sort_values(by='ds').groupby('unique_id').tail(30 * 24)

#Filter out the time series that are too short to be processed in the cross-validation process
cpu_memory_df = cpu_memory_df.groupby('unique_id').filter(lambda x: len(x) >= 400)

# Count the number of rows for each 'unique_id'
#row_counts = cpu_memory_df.groupby('unique_id').size()

# Display the distribution of row counts
#print(row_counts)



# this makes it so that the outputs of the predict methods have the id as a column 
# instead of as the index
os.environ['NIXTLA_ID_AS_COL'] = '1'

def evaluate_cross_validation(df, metric):
    models = df.drop(columns=['unique_id', 'ds', 'cutoff', 'y']).columns.tolist()
    evals = []
    # Calculate loss for every unique_id and cutoff.    
    for cutoff in df['cutoff'].unique():
        eval_ = evaluate(df[df['cutoff'] == cutoff], metrics=[metric], models=models)
        evals.append(eval_)
    evals = pd.concat(evals)
    evals = evals.groupby('unique_id').mean(numeric_only=True) # Averages the error metrics for all cutoffs for every combination of model and unique_id
    evals['best_model'] = evals.idxmin(axis=1)
    return evals

# Output file to append the results
output_file_path = 'models_kpi_cpu_memory_dataset.csv'

from statsforecast.models import (
    AutoARIMA,
    AutoTheta,
    AutoETS,
    AutoCES,
    MSTL,
    SeasonalNaive,
    WindowAverage,
    SeasonalWindowAverage,
    Naive
)

models = [
    AutoARIMA(season_length=24),
    AutoTheta(season_length=24),
    AutoETS(season_length=24),
    AutoCES(season_length=24),
    MSTL(season_length=24),
    SeasonalNaive(season_length=24), 
    WindowAverage(window_size=24), 
    SeasonalWindowAverage(window_size=1, season_length=24),
    Naive()
]

sf = StatsForecast( 
    models=models,
    freq='H',
    fallback_model = SeasonalNaive(season_length=24),
    n_jobs=-1,
)

crossvaldation_df = sf.cross_validation(
    df=cpu_memory_df,
    h=168,
    step_size=48,
    n_windows=1
)


grouped = cpu_memory_df.groupby('unique_id')

for unique_id, group_df in grouped:
    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(group_df)
    print(unique_id)

    #Setup prophet initial trainig period dynamically based on the number of days in the data frame

    df = df = pd.DataFrame()
    df['date'] = group_df['ds'].dt.date
    df['hour'] = group_df['ds'].dt.hour

    daily_hours = df.groupby('date')['hour'].nunique()
    daily_fraction = daily_hours / 24
    total_days = daily_fraction.sum()
    total_days_rounded_down = math.floor(total_days * 10) / 10

    horizon = 7
    initial = total_days_rounded_down - horizon - 1
    prophet_horizon = str(horizon) + ' days'
    prophet_initial = str(initial) + ' days'
    print(prophet_initial)

    try:
        df_cv = cross_validation(model, horizon=prophet_horizon, initial=prophet_initial)
    except Exception as e:
        print(f"An error occurred during cross-validation for {unique_id}: {e}")
        continue

    df_cv = df_cv.sort_values(by='ds')
    df_cv['unique_id'] = unique_id
    df_new = df_cv[['ds', 'unique_id', 'yhat']].rename(columns={'yhat': 'prophet'})
    print(df_cv)
    # If 'prophet' already exists in crossvaldation_df, prepare to merge and resolve the column values
    if 'prophet' in crossvaldation_df.columns:
        # Temporarily rename 'prophet' in crossvaldation_df to avoid automatic suffixing
        crossvaldation_df.rename(columns={'prophet': 'prophet_temp'}, inplace=True)

        # Merge df1 and df_new
        crossvaldation_df = pd.merge(crossvaldation_df, df_new, on=['ds','unique_id'], how='left')

        # Update 'prophet_temp' with 'prophet' from df_new where available
        crossvaldation_df['prophet'] = crossvaldation_df['prophet'].combine_first(crossvaldation_df['prophet_temp'])

        # Drop the temporary and '_new' columns
        crossvaldation_df.drop(columns=['prophet_temp'], inplace=True)
    else:
        # If 'prophet' does not exist yet, simply merge
        crossvaldation_df = pd.merge(crossvaldation_df, df_new, on=['ds','unique_id'], how='left')
    

evaluation_df = evaluate_cross_validation(crossvaldation_df, mse)

# Connect to SQLite database (this will create the database if it doesn't exist)
db_connection = sqlite3.connect('cross-validation.db')
cursor = db_connection.cursor()


evaluation_df.to_sql('evaluate_cross_validation_one_week', db_connection, if_exists='replace', index=True) 

# Commit changes and close the connection
db_connection.commit()
db_connection.close()