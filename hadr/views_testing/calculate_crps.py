import pandas as pd
import numpy as np
from crps import crps_ensemble

def calculate_crps(actuals_file, forecasts_file):
    # Read actuals
    actuals = pd.read_csv(actuals_file)
    observations = actuals['outcome'].tolist()

    # Read forecasts
    forecasts_df = pd.read_csv(forecasts_file)
    forecasts = forecasts_df.groupby('month_id')['outcome'].apply(list).reset_index(name='forecasts')
    forecasts_array = np.array(forecasts['forecasts'].tolist())

    results = []
    for i in range(len(forecasts_array)):
        score = crps_ensemble(forecasts_array[i], observations[i])
        results.append({
            'month_id': i,
            'forecast': forecasts_array[i],
            'observation': observations[i],
            'crps_score': score
        })
    
    return results

# Example usage:
if __name__ == "__main__":
    actuals_file = "DRC_cm_actuals_2019.csv"
    forecasts_file = "DRC_Conflictology_2019.csv"
    
    results = calculate_crps(actuals_file, forecasts_file)
    
    for result in results:
        print("--------------------------------")
        print(f"month_id: {result['month_id']}, forecast: {result['forecast']}, observation: {result['observation']}")
        print(f"CRPS Score: {result['crps_score']:.4f}")