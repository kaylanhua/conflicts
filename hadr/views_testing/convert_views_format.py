import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../../')
from components.universal import get_country_id

def convert_views_format(input_file, output_file, country_name):
    country_id = get_country_id(country_name)
    df = pd.read_csv(input_file)
    df['month_id'] = pd.to_datetime(df['Date'], format='%b/%Y').dt.to_period('M').astype(int) + 468
    percentiles = ['90th pct', '50th pct', '10th pct']
    
    new_data = []
    for _, row in df.iterrows():
        for draw, percentile in enumerate(percentiles):
            new_data.append({
                'month_id': row['month_id'],
                'country_id': country_id,
                'draw': draw,
                'outcome': row[percentile]
            })
    
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(output_file, index=False)
    print(f"Converted file saved as {output_file}")

def process_country(country_name):
    # Infer the folder path based on the country name
    year = "2019"  # Assuming the year is always 2019, adjust if needed
    folder_path = f'/Users/kaylahuang/Desktop/conflicts/hadr/views_testing/{country_name}_{year}'
    # folder_path = os.path.join("hadr", "views_testing", f"{country_name}_{year}")
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return

    for filename in os.listdir(folder_path):
        if "views" in filename.lower() and filename.endswith('.csv'):
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_converted.csv")
            convert_views_format(input_file, output_file, country_name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_views_format.py <country_name>")
        sys.exit(1)

    country_name = sys.argv[1].lower()
    process_country(country_name)