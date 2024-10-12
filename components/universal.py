import pandas as pd
from fuzzywuzzy import process
import os
import sys
sys.path.append(os.getcwd())

# Check if the file exists before trying to read it
file_path = '/Users/kaylahuang/Desktop/conflicts/data/views/country_key.csv'
if os.path.exists(file_path):
    country_key_df = pd.read_csv(file_path)
else:
    print(os.getcwd())
    print(sys.path)
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path and ensure the file is present.")

def get_country_id(country_name):
    best_match = process.extractOne(country_name, country_key_df['name'])
    if best_match[1] >= 80:  # Threshold for a good match
        return country_key_df[country_key_df['name'] == best_match[0]]['id'].values[0]
    else:
        raise ValueError(f"No close match found for country name: {country_name}")
    
def get_country_gwid(country_name):
    best_match = process.extractOne(country_name, country_key_df['name'])
    if best_match[1] >= 80:  # Threshold for a good match
        return country_key_df[country_key_df['name'] == best_match[0]]['gwcode'].values[0]
    else:
        raise ValueError(f"No close match found for country name: {country_name}")

def get_country_name(country_id):
    return country_key_df[country_key_df['id'] == country_id]['name'].values[0]
