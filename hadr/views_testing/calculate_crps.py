import pandas as pd
import numpy as np
from crps import crps_ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import math

# constants
preamble = "2019/"
actuals_file = preamble + "DRC_cm_actuals_2019.csv"
conflictology_file = preamble + "DRC_Conflictology_2019.csv"
# lstm_file = preamble + "DRC_lstm_forecasts_2019.csv"
rf_file = preamble + "DRC_rf_forecasts_2019.csv"
# rag_file = preamble + "drc_RAG_2019.csv"
pred_file = rf_file

# ignorance score
def log_score(f_y):
    if f_y <= 0:
        return float('inf')  # Return infinity for non-positive values
    return -math.log2(f_y)

def calculate_metrics(actuals_file, forecasts_file):
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
        forecast_columns = [col for col in forecasts_df.columns if col.startswith('forecast_')]
        # sets negative forecasts to 0
        forecasts = forecasts_df.groupby('month_id')[forecast_columns].apply(lambda x: np.maximum(x.values.flatten(), 0).tolist()).reset_index(name='forecasts')
    
    forecasts_array = np.array(forecasts['forecasts'].tolist())

    results = []
    for i in range(len(forecasts_array)):
        forecast = forecasts_array[i]
        observation = observations[i]
        crps_score = crps_ensemble(forecast, observation)
        
        # Calculate mean and standard deviation of forecast
        forecast_mean = np.mean(forecast)
        forecast_std = np.std(forecast)
        
        # Calculate additional metrics
        mse = mean_squared_error([observation], [forecast_mean])
        mae = mean_absolute_error([observation], [forecast_mean])
        r2 = r2_score([observation], [forecast_mean])
        
        # Calculate IGN (Ignorance Score)
        # not sure if this is correct (i.e. if calculated and binned the same way as in VIEWS)
        f_y = norm.pdf(observation, loc=forecast_mean, scale=forecast_std) 
        ign = log_score(f_y) 
        
        results.append({
            'month_id': i,
            'forecast': forecast,
            'observation': observation,
            'crps_score': crps_score,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'ign': ign
        })
    
    return results

# Example usage:
if __name__ == "__main__":

    conflictology_results = calculate_metrics(actuals_file, conflictology_file)
    predictions = calculate_metrics(actuals_file, pred_file)
    
    print("Comparison of Conflictology and LSTM Results:")
    beat = 0
    for i in range(len(conflictology_results)):
        conflictology_metrics = conflictology_results[i]
        pred_metrics = predictions[i]
        if pred_metrics['crps_score'] > conflictology_metrics['crps_score']:
            color = "\033[91m"
        else:
            color = "\033[92m"
            beat += 1
        print(f"{color}--------------------------------")
        print(f"month_id: {conflictology_metrics['month_id']}, observation: {conflictology_metrics['observation']}")
        print(f"Conflictology forecast: {conflictology_metrics['forecast']}")
        print(f"CRPS: {conflictology_metrics['crps_score']:.4f}, MSE: {conflictology_metrics['mse']:.4f}, MAE: {conflictology_metrics['mae']:.4f}, R²: {conflictology_metrics['r2']:.4f}, IGN: {conflictology_metrics['ign']:.4f}")
        print(f"My forecast: {pred_metrics['forecast']}")
        print(f"CRPS: {pred_metrics['crps_score']:.4f}, MSE: {pred_metrics['mse']:.4f}, MAE: {pred_metrics['mae']:.4f}, R²: {pred_metrics['r2']:.4f}, IGN: {pred_metrics['ign']:.4f}\033[0m")
        
    print(f"predictions beat baseline on: {beat} / {len(conflictology_results)} months")