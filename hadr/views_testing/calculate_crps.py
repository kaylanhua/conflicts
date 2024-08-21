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
    conflictology_file = "DRC_Conflictology_2019.csv"
    lstm_file = "DRC_lstm_forecasts_2019.csv"
    
    conflictology_results = calculate_crps(actuals_file, conflictology_file)
    lstm_results = calculate_crps(actuals_file, lstm_file)
    
    print("Comparison of Conflictology and LSTM Results:")
    for i in range(len(conflictology_results)):
        print("--------------------------------")
        print(f"month_id: {conflictology_results[i]['month_id']}, observation: {conflictology_results[i]['observation']}")
        print(f"Conflictology forecast: {conflictology_results[i]['forecast']}")
        print(f"Conflictology CRPS Score: {conflictology_results[i]['crps_score']:.4f}")
        print(f"LSTM forecast: {lstm_results[i]['forecast']}")
        print(f"LSTM CRPS Score: {lstm_results[i]['crps_score']:.4f}")