import pandas as pd
import os
import datetime

# Load the data
features_df = pd.read_parquet('/Users/kaylahuang/Desktop/conflicts/data/views/cm_features.parquet')
month_key_df = pd.read_csv('/Users/kaylahuang/Desktop/conflicts/data/views/month_key.csv')

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

# Function to create target dataset for a specific year
def create_target_dataset(year):
    start_date = pd.to_datetime(f'{year}-01-01')
    end_date = pd.to_datetime(f'{year}-12-31')
    target_dataset = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)]
    return target_dataset

# Create datasets for each year
# years = [2018, 2019, 2020, 2021, 2024]
years = [2022, 2023]
for year in years:
    # Create and save input dataset
    # dataset = create_dataset(year)
    # input_path = f'../../data/views/input_data_{year}.csv'
    # dataset.to_csv(input_path, index=False)
    # print(f"Input dataset for {year} created and saved to {input_path}")

    # Create and save target dataset
    target_dataset = create_target_dataset(year)
    target_path = f'../../data/views/target_{year}.csv'
    target_dataset.to_csv(target_path, index=False)
    print(f"Target dataset for {year} created and saved to {target_path}")