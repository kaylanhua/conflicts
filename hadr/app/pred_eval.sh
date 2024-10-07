#!/bin/bash

# Check if a country argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a country name as an argument."
    exit 1
fi

COUNTRY="$1"

cd /Users/kaylahuang/Desktop/conflicts/hadr/app
conda activate hadr
python cycles.py "$COUNTRY"

cd /Users/kaylahuang/Desktop/conflicts/hadr/views_testing
python calculate_crps.py "$COUNTRY"