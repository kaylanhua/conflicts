import pandas as pd
import numpy as np
from crps import crps_ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import math

# constants
pred_preamble = "DRC_2019/"
bm_preamble = "../../data/views/bm/"
country = "DRC"
year = "2019"
actuals_file = pred_preamble + "DRC_cm_actuals_2019.csv"

# BASELINE FORECAST FILES
def get_file_path(file_type, model=None, variant=None):
    base = f"{pred_preamble}{country}_{year}.csv"
    
    if file_type == "baseline":
        if model == "conflictology":
            return f"{pred_preamble}{{country}}_Conflictology_{{year}}.csv"
        return f"{pred_preamble}{country}_{model}_{year}.csv"
    
    elif file_type == "prediction":
        if model == "rag":
            if variant:
                return f"{pred_preamble}{country}_RAG_{variant}_{year}.csv"
            return f"{pred_preamble}{country}_RAG_{year}.csv"
        return f"{pred_preamble}{country}_{model}_forecasts_{year}.csv"

# BASELINE FORECAST FILES
conflictology_file = get_file_path("baseline", "conflictology")
last_file = get_file_path("baseline", "last")
boot_file = get_file_path("baseline", "boot")
zero_file = get_file_path("baseline", "zero")

# PREDICTION FILES
lstm_file = get_file_path("prediction", "lstm")
rf_file = get_file_path("prediction", "rf")
rag_file = get_file_path("prediction", "rag")
rag_with_dates_file = get_file_path("prediction", "rag", "with_dates")
rag_with_dates_and_country_file = get_file_path("prediction", "rag", "with_dates-country")
rag_claude_file = get_file_path("prediction", "rag", "claude")
rag_fixed_file = get_file_path("prediction", "rag", "fixed")

# set the evaluation files 
bm_file = zero_file
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

# List of all files to evaluate
files_to_evaluate = {
    'Conflictology': conflictology_file,
    'Last': last_file,
    'Boot': boot_file,
    'Zero': zero_file,
    'LSTM': lstm_file,
    'Random Forest': rf_file,
    'RAG': rag_file,
    'RAG with Dates': rag_with_dates_file,
    'RAG w/ Dates and Country': rag_with_dates_and_country_file,
    'RAG Claude': rag_claude_file,
    'RAG Fixed (GPT)': rag_fixed_file
}

def calculate_aggregate_metrics(results):
    metrics = {
        'CRPS': np.mean([r['crps_score'] for r in results]),
        'MSE': np.mean([r['mse'] for r in results]),
        'MAE': np.mean([r['mae'] for r in results]),
        'R²': np.mean([r['r2'] for r in results]),
        'IGN': np.mean([r['ign'] for r in results])
    }
    return metrics

def print_latex_table(all_results):
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l|ccccc}")
    print("\\hline")
    print("Model & CRPS & MSE & MAE & R² & IGN \\\\")
    print("\\hline")
    for model, metrics in all_results.items():
        print(f"{model} & {metrics['CRPS']:.4f} & {metrics['MSE']:.4f} & {metrics['MAE']:.4f} & {metrics['R²']:.4f} & {metrics['IGN']:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Comparison of Model Performance}")
    print("\\label{tab:model-comparison}")
    print("\\end{table}")

if __name__ == "__main__":
    all_results = {}

    for model, file in files_to_evaluate.items():
        results = calculate_metrics(actuals_file, file.format(country=country, year=year))
        aggregate_metrics = calculate_aggregate_metrics(results)
        all_results[model] = aggregate_metrics

    print_latex_table(all_results)