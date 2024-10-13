import pandas as pd
import argparse
import sys
import os
sys.path.append('../../../')
from components.universal import get_country_id
from components.create_country_data import create_country_data

## MANUAL STEPS
# download the parquet files for each type and year from ViEWS dropbox

## THIS SCRIPT
# for a particular country, create the folder, actuals file, and benchmark files since 2010

def create_folder(country):
    os.makedirs(f"/Users/kaylahuang/Desktop/conflicts/hadr/results/{country}", exist_ok=True)
    global COUNTRY_FOLDER
    COUNTRY_FOLDER = f"/Users/kaylahuang/Desktop/conflicts/hadr/results/{country}"
    
    for year in range(2018, 2024):
        os.makedirs(f"{COUNTRY_FOLDER}/{year}", exist_ok=True)

def process_country(country, start_year=2010):
    if not os.path.exists(f"/Users/kaylahuang/Desktop/conflicts/data/views/{country}.csv"):
        create_country_data(country)
    
    country_data = pd.read_csv(f"/Users/kaylahuang/Desktop/conflicts/data/views/{country}.csv")

    def load_month_key():
        """
        Load the month_key.csv file and return a dictionary mapping month_id to (year, month).
        """
        month_key_df = pd.read_csv('/Users/kaylahuang/Desktop/conflicts/data/views/month_key.csv')
        return {row['month_id']: (row['Year'], row['Month']) for _, row in month_key_df.iterrows()}

    # Load the month key
    month_key = load_month_key()
    country_data_filtered = country_data[country_data['month_id'].isin([k for k, v in month_key.items() if v[0] >= start_year])]

    # Select only the specified columns
    country_data_filtered = country_data_filtered[['month_id', 'country_id', 'ged_sb']]
    country_data_filtered = country_data_filtered.rename(columns={'ged_sb': 'outcome'})

    print(f"Filtered data for year {start_year} until the present:")
    print(country_data_filtered)
    country_data_filtered.to_csv(f"{COUNTRY_FOLDER}/{country}_cm_actuals.csv", index=False)
    print(f"Saved filtered data to {COUNTRY_FOLDER}/{country}_cm_actuals.csv")


def process_benchmark(country, country_id, country_folder):
    type = ["Conflictology", "last", "zero", "boot"]
    
    for year in range(2018, 2024):
        for t in type:
            # check if already converted to csv
            if os.path.exists(f"/Users/kaylahuang/Desktop/conflicts/data/views/bm/{t}_{year}.csv"):
                outcome_df = pd.read_csv(f"/Users/kaylahuang/Desktop/conflicts/data/views/bm/{t}_{year}.csv")
            else:
                c18 = pd.read_parquet(f"/Users/kaylahuang/Desktop/conflicts/data/views/bm/{t} {year}.parquet")
                outcome_series = c18['outcome']
                print("Series info:")
                print(outcome_series.info())

                outcome_df = outcome_series.reset_index()
                print("\nConverted to DataFrame:")
                print(outcome_df.head())
                outcome_df.to_csv(f"/Users/kaylahuang/Desktop/conflicts/data/views/bm/{t}_{year}.csv")
            
            country_data = outcome_df[outcome_df['country_id'] == country_id]
            country_data.to_csv(f"{country_folder}/{year}/{country}_{t}_{year}.csv")

            print(f"Generated file for {t} {year}")
        

def main():
    parser = argparse.ArgumentParser(description="Process ViEWS country data")
    parser.add_argument("country", type=str, help="The country of the data")
    parser.add_argument("--start_year", type=int, help="The year of the data")
    args = parser.parse_args()  
    
    COUNTRY = args.country
    create_folder(COUNTRY)
    country_folder = f"/Users/kaylahuang/Desktop/conflicts/hadr/results/{COUNTRY}"
    
    country_id = get_country_id(COUNTRY)
    print(f"Country ID: {country_id}")
    process_country(COUNTRY, args.start_year if args.start_year else 2018)
    process_benchmark(COUNTRY, country_id, country_folder)
    

if __name__ == "__main__":
    main()