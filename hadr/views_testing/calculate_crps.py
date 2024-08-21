import pandas as pd
import numpy as np
from crps import crps_ensemble

def calculate_crps(actuals_file, forecasts_file):
    # Read actuals
    actuals = pd.read_csv(actuals_file)
    observations = actuals['outcome'].tolist()

    # Read forecasts
    forecasts_df = pd.read_csv(forecasts_file)
    # Check if the forecasts file has a 'draw' column (Conflictology format)
    if 'draw' in forecasts_df.columns:
        forecasts = forecasts_df.groupby('month_id')['outcome'].apply(list).reset_index(name='forecasts')
    else:
        # Handle LSTM format
        # forecast_columns = [col for col in forecasts_df.columns if col.startswith('forecast_')]
        forecast_columns = ['forecast_1', 'forecast_3']
        # sets negative
        forecasts = forecasts_df.groupby('month_id')[forecast_columns].apply(lambda x: np.maximum(x.values.flatten(), 0).tolist()).reset_index(name='forecasts')
    
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
        print(f"\033[1m--------------------------------\nmonth_id: {conflictology_results[i]['month_id']}, observation: {conflictology_results[i]['observation']}\nConflictology forecast: {conflictology_results[i]['forecast']}, CRPS: {conflictology_results[i]['crps_score']:.4f}\nLSTM forecast: {lstm_results[i]['forecast']}, CRPS: {lstm_results[i]['crps_score']:.4f}\033[0m")
        print("\033[91m" if conflictology_results[i]['crps_score'] > lstm_results[i]['crps_score'] else "\033[92m", end="")