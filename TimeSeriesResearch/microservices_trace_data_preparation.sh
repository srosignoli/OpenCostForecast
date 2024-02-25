#!/bin/bash

#############################################################################
## Alibaba Microservices trace data preparation script
## 
## Please parse and filter the MSResource trace files 
## that are present in the Alibaba cluster-trace-microservices-v2022 dataset.
#############################################################################
## Author: Simone Rosignoli
## 
## 
## 
#############################################################################

# Check if an instanceid parameter is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <instanceid1> <instanceid2>"
    exit 1
fi

# Instance ID passed as a parameter
instanceid="$1"
instanceid2="$2"

# Specify the directory where your .tar.gz files are located
directory="/media/simone/TOSHIBA EXT/Dataset/data/MSMetrics"

# Loop through each sorted .tar.gz file in the directory
sorted_files=$(find "$directory" -type f -printf "%f\n" | sort -t_ -k2,2n)

while IFS= read -r file; do
    if [ -e "$directory/$file" ]; then
        # Extract the contents to a temporary directory
        temp_dir=$(mktemp -d)
        tar -xzvf "$directory/$file" -C "$temp_dir"

        # Loop through each extracted CSV file
        for csv_file in "$temp_dir"/*.csv; do
            if [ -e "$csv_file" ]; then
                # Filter and print only the desired columns
                awk -F, -v instanceid="$instanceid" 'BEGIN {OFS=","} $3 == instanceid {print $1, $5}' "$csv_file" | sort -t, -k1,1 -n >> "${instanceid}_CPU.csv"
                awk -F, -v instanceid="$instanceid" 'BEGIN {OFS=","} $3 == instanceid {print $1, $6}' "$csv_file" | sort -t, -k1,1 -n >> "${instanceid}_Memory.csv"

                awk -F, -v instanceid="$instanceid2" 'BEGIN {OFS=","} $3 == instanceid {print $1, $5}' "$csv_file" | sort -t, -k1,1 -n >> "${instanceid2}_CPU.csv"
                awk -F, -v instanceid="$instanceid2" 'BEGIN {OFS=","} $3 == instanceid {print $1, $6}' "$csv_file" | sort -t, -k1,1 -n >> "${instanceid2}_Memory.csv"
                echo "Filtered content for $instanceid in $csv_file"
            else
                echo "CSV file not found: $csv_file"
            fi
        done

        # Clean up temporary directory
        rm -rf "$temp_dir"

        echo "Processed $file"
    else
        echo "File not found: $file"
    fi
done <<< "$sorted_files"
