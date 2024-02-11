#############################################################################
## Run the Naive baseline model on GWA-T-12 Bitbrains fastStorage Traces 
## and calculate the following KPI: MAE, MSE, and RMSE.
## 
#############################################################################
## Author: Simone Rosignoli
## 
## 
## 
#############################################################################

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Directory containing the CSV files
file_paths = ['rnd/2013-7', 'rnd/2013-8', 'rnd/2013-9']

# Output file to append the results
output_file_path = 'kpi_results_rnd.csv'

# Function to process each CSV file
def process_file(filename):
    # Initialize an empty list to store each DataFrame
    dataframes = []

    # Loop through each file path, load the data, and append to the list
    for file_path in file_paths:
        df = pd.read_csv(file_path+'/'+filename, sep=';\t', engine='python')  # Load each file
        dataframes.append(df)  # Append the DataFrame to the list

    # Concatenate all DataFrames in the list into a single DataFrame
    data = pd.concat(dataframes, ignore_index=True)
    
    start_date = "2013-08-01 00:00:00"
    start_timestamp = pd.to_datetime(start_date)
    timestamps = [start_timestamp + pd.Timedelta(minutes=5*i) for i in range(len(data))]
    data['Corrected Timestamp'] = timestamps
    data['CPU usage [%]_lagged'] = data['CPU usage [%]'].shift(1).fillna(0)
    data_reordered = data[['Corrected Timestamp', 'CPU usage [%]_lagged', 'CPU usage [%]']].dropna()
    
    train, test = train_test_split(data_reordered, test_size=0.4, shuffle=False)
    y_true = test['CPU usage [%]']
    y_pred = test['CPU usage [%]_lagged']
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Calculate the mean and standard deviation of the CPU usage 
    mean_cpu_usage_perc = data['CPU usage [%]'].mean()
    std_cpu_usage_perc = data['CPU usage [%]'].std()

    # Calculate the coefficient of variation for CPU usage in MHZ
    cv_cpu_usage_perc = (std_cpu_usage_perc / mean_cpu_usage_perc)
    
    return mae, mse, rmse, cv_cpu_usage_perc

# Open the output file in append mode
with open(output_file_path, 'a') as output_file:
    # Write the header if the file is new
    if os.stat(output_file_path).st_size == 0:
        output_file.write("Filename,MAE,MSE,RMSE,CV\n")
    
    # Process each CSV file in the directory
    for filename in os.listdir(file_paths[0]):
        if filename.endswith('.csv'):
            mae, mse, rmse, cv_cpu_usage_perc = process_file(filename)
            
            # Append the results to the output file
            output_file.write(f"{filename},{mae},{mse},{rmse},{cv_cpu_usage_perc}\n")

print("Processing complete. Results appended to", output_file_path)
