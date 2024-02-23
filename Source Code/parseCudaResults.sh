#!/bin/bash

# Check if an input file was provided
if [[ $# -eq 0 ]] ; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file=$1
output_file="${input_file%.*}.csv"

# Prepare the CSV header
echo "m,k,NZ,xBlockSize,yBlockSize,CRS,ELLPACK,CRS performance,ELLPACK performance,Serial CRS performance,Serial ELLPACK performance" > "$output_file"

# Parse the input file and append to CSV
awk '{
    for (i = 1; i <= NF; i++) {
        if ($i ~ /:/) {
            gsub(/,/, "", $(i+1)); # Remove commas from numbers
            printf "%s", $(i+1);
            if (i < NF-1) {
                printf ",";
            }
        }
    }
    printf "\n";
}' "$input_file" >> "$output_file"

echo "CSV file created: $output_file"
