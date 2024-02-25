#############################################################################
## The script generates a unique dataset of 500 GWA-T-12 Bitbrains RND VM traces, 
## each identified by a unique ID equivalent to the VM number. 
## The dataset has an average of three months of observations for each VM. 
## 
## 
#############################################################################
## Author: Simone Rosignoli
## 
## 
## 
#############################################################################


import pandas as pd
import os

# Directory containing the CSV files
file_paths = ['rnd/2013-7', 'rnd/2013-8', 'rnd/2013-9']

# Output file to append the results
output_file_path = 'unique_dataset_rnd.csv'

# Function to process and concatenate CSV files from different directories
def concatenate_and_process_files(file_paths, output_file_path):
    all_dataframes = []  # List to store each processed DataFrame

    # Loop through each directory
    for dir_path in file_paths:
        # List all CSV files in the directory
        for filename in os.listdir(dir_path):
            if filename.endswith('.csv'):
                # Construct the full file path
                full_path = os.path.join(dir_path, filename)
                # Load the CSV file
                df = pd.read_csv(full_path, sep=';\t', engine='python')
                # Create 'unique_id' column, extracting filename without '.csv' and appending '_CPU'
                unique_id = filename[:-4] + '_CPU'
                df['unique_id'] = unique_id
                # Append the DataFrame to the list
                all_dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    final_data = pd.concat(all_dataframes, ignore_index=True)
    # Convert from second to datetime tiemstamp
    final_data['timestamp'] = pd.to_datetime(final_data['Timestamp [ms]'], unit='s')
    # Save the final DataFrame to CSV
    final_data.to_csv(output_file_path, index=False)
    
    return final_data

# Call the function
data = concatenate_and_process_files(file_paths, output_file_path)
