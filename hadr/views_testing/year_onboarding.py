import pandas as pd

## MANUAL STEPS
# download the parquet files for each type and year from ViEWS dropbox

year = 2023
type = ["Conflictology", "last", "zero", "boot"]
for t in type:
    c18 = pd.read_parquet(f"../../data/views/bm/{t} {year}.parquet")

    # Parse the 'outcome' series
    outcome_series = c18['outcome']

    print("Series info:")
    print(outcome_series.info())

    outcome_df = outcome_series.reset_index()
    print("\nConverted to DataFrame:")
    print(outcome_df.head())
    outcome_df.to_csv(f"{type}_{year}.csv")
    
    print(f"Saved {type} {year} to csv")

    # print("\nAccessing specific levels:")
    # print("month_id values:", outcome_series.index.get_level_values('month_id').unique())
    # print("country_id values:", outcome_series.index.get_level_values('country_id').unique())
    # print("draw values:", outcome_series.index.get_level_values('draw').unique())

