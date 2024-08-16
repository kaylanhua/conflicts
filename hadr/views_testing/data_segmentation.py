import pandas as pd
import os
import datetime

# Load the data
features_df = pd.read_parquet('../../data/views/cm_features.parquet')
month_key_df = pd.read_csv('../../data/views/month_key.csv')

def parse_date(date_str):
    month, year = date_str.split('-')
    year = int(year)
    if year >= 80:
        year += 1900
    else:
        year += 2000
    return datetime.datetime.strptime(f"{month}-{year}", "%b-%Y")

merged_df = pd.merge(features_df, month_key_df, on='month_id', how='left')

merged_df['Date'] = merged_df['Date'].apply(parse_date)


# Function to create dataset for a specific year
def create_dataset(year):
    cutoff_date = pd.to_datetime(f'{year-1}-10-31')
    dataset = merged_df[merged_df['Date'] <= cutoff_date]
    return dataset

# Create datasets for each year
years = [2018, 2019, 2020, 2021, 2024]
for year in years:
    dataset = create_dataset(year)
    output_path = f'../../data/views/input_data_{year}.csv'
    dataset.to_csv(output_path, index=False)
    print(f"Dataset for {year} created and saved to {output_path}")