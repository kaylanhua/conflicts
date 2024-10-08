import sys
sys.path.append("/Users/kaylahuang/Desktop/conflicts/components/")

from universal import get_country_gwid
from views_cleaner import VIEWSCleaner
import argparse

def main(country):
    filename = '../data/views/cm_features.parquet'
    gw_id = get_country_gwid(country)

    cleaner = VIEWSCleaner(filename, gw_id, trim_full=False)
    original_features = cleaner.features # already aggregated by month
    X = original_features.copy()

    print(f"Shape of data for {country}: {X.shape}")
    cleaner.plot(n=4)

    output_file = f'../data/views/{country.lower().replace(" ", "_")}.csv'
    X.to_csv(output_file)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate country data from VIEWS dataset")
    parser.add_argument("country", type=str, help="Name of the country to process")
    args = parser.parse_args()

    main(args.country)

