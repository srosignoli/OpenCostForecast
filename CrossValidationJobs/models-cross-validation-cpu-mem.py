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
from datetime import datetime, timedelta
from kubernetes import client, config

# Prometheus server details
PROMETHEUS = 'http://localhost:9090'


def get_containers_memory(pod_list):
    df_containers = pd.DataFrame()
    for pod in pod_list:
        QUERY = f'container_memory_allocation_bytes{{pod="{pod}"}}/1024/1024/1024'
        # Calculate start and end times for the last hour
        END = datetime.now()
        START = END - timedelta(hours=1)
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

        pod_id = pod + "-RAM"
        # Extract the data points
        for result in data:
            for value in result['values']:
                # Convert timestamp to readable date
                timestamp = datetime.fromtimestamp(float(value[0]))
                memory_usage = float(value[1])
                data_points.append((timestamp, memory_usage, pod_id))

        # Create a DataFrame
        df = pd.DataFrame(data_points, columns=['Timestamp', 'MemoryUsageGB', 'UniqueID'])
        df_containers = pd.concat([df_containers, df], ignore_index=True)

    return df_containers

def get_containers_cpu(pod_list):
    df_containers = pd.DataFrame()
    for pod in pod_list:
        QUERY = f'container_cpu_allocation{{pod="{pod}"}}'
        # Calculate start and end times for the last hour
        END = datetime.now()
        START = END - timedelta(hours=1)
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

        pod_id = pod + "-CPU"
        # Extract the data points
        for result in data:
            for value in result['values']:
                # Convert timestamp to readable date
                timestamp = datetime.fromtimestamp(float(value[0]))
                memory_usage = float(value[1])
                data_points.append((timestamp, memory_usage, pod_id))

        # Create a DataFrame
        df = pd.DataFrame(data_points, columns=['Timestamp', 'CPUUsage', 'UniqueID'])
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




