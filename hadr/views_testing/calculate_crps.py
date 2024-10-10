import os
import pandas as pd
import numpy as np
from crps import crps_ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import sys

# Constants
if len(sys.argv) < 2:
    print("Please provide a country name as an argument.")
    sys.exit(1)

country = sys.argv[1].lower()
year = "2019"
pred_preamble = f"{country}_{year}/"
actuals_file = pred_preamble + f"{country}_cm_actuals_{year}.csv"

def get_file_path(file_type, model=None, variant=None):
    if file_type == "baseline":
        if model == "conflictology":
            return f"{pred_preamble}{country}_Conflictology_{year}.csv"
        return f"{pred_preamble}{country}_{model}_{year}.csv"
    elif file_type == "prediction":
        if model == "rag" and variant:
            return f"{pred_preamble}{country}_RAG_{variant}_{year}.csv"
        return f"{pred_preamble}{country}_{model}_forecasts_{year}.csv"

def log_score(f_y):
    epsilon = 1e-10  # Small value to avoid log(0)
    return -np.log2(f_y + epsilon)

def calculate_metrics(actuals_file, forecasts_file):
    actuals = pd.read_csv(actuals_file)
    observations = actuals['outcome'].tolist()

    forecasts_df = pd.read_csv(forecasts_file)
    if 'draw' in forecasts_df.columns:
        forecasts = forecasts_df.groupby('month_id')['outcome'].apply(list).reset_index(name='forecasts')
    else:
        forecast_columns = [col for col in forecasts_df.columns if col.startswith('forecast_')]
        forecasts = forecasts_df.groupby('month_id')[forecast_columns].apply(lambda x: np.maximum(x.values.flatten(), 0).tolist()).reset_index(name='forecasts')
    
    forecasts_array = np.array(forecasts['forecasts'].tolist())

    results = []
    for i in range(len(forecasts_array)):
        forecast = forecasts_array[i]
        observation = observations[i]
        
        if len(forecast) == 0:
            print(f"Warning: Empty forecast for month_id {i}. Skipping.")
            continue
        
        crps_score = crps_ensemble(forecast, observation)
        forecast_mean = np.mean(forecast)
        forecast_std = np.std(forecast) + 1e-10  # Add small epsilon to avoid division by zero
        mse = mean_squared_error([observation], [forecast_mean])
        mae = mean_absolute_error([observation], [forecast_mean])
        f_y = norm.pdf(observation, loc=forecast_mean, scale=forecast_std)
        ign = log_score(f_y)
        
        results.append({
            'month_id': i,
            'forecast': forecast,
            'observation': observation,
            'crps_score': crps_score,
            'mse': mse,
            'mae': mae,
            'ign': ign
        })
    
    return results

def calculate_aggregate_metrics(results):
    if not results:
        return {metric: float('nan') for metric in ['CRPS', 'MSE', 'MAE', 'IGN']}
    
    metrics = {
        'CRPS': np.mean([r['crps_score'] for r in results]),
        'MSE': np.mean([r['mse'] for r in results]),
        'MAE': np.mean([r['mae'] for r in results]),
        'IGN': np.mean([r['ign'] for r in results])
    }
    return metrics

def print_latex_table(all_results):
    sorted_results = dict(sorted(all_results.items(), key=lambda item: item[1]['MSE']))
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l|cccc}")
    print("\\hline")
    print("Model & CRPS & MSE & MAE & IGN \\\\")
    print("\\hline")
    for model, metrics in sorted_results.items():
        model_name = model.replace('_', ' ')
        print(f"{model_name} & {metrics['CRPS']:.4f} & {metrics['MSE']:.4f} & {metrics['MAE']:.4f} & {metrics['IGN']:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Comparison of Model Performance}")
    print("\\label{tab:model-comparison}")
    print("\\end{table}")

def get_all_prediction_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv') and f != actuals_file.split('/')[-1]]

def visualize_table(all_results):
    sorted_results = dict(sorted(all_results.items(), key=lambda item: item[1]['CRPS']))

    fig, ax = plt.subplots(figsize=(12, len(sorted_results) * 0.5 + 1))
    ax.axis('off')
    ax.axis('tight')

    data = [[model] + [f"{metrics[m]:.4f}" for m in ['CRPS', 'MSE', 'MAE', 'IGN']] 
            for model, metrics in sorted_results.items()]
    
    table = ax.table(cellText=data,
                     colLabels=['Model', 'CRPS', 'MSE', 'MAE', 'IGN'],
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)

    # Highlight rows based on model name
    for i, model in enumerate(sorted_results.keys()):
        if "RAG" in model:
            for j in range(5):  # 5 columns
                table[(i+1, j)].set_facecolor('lightgreen')
        else:
            for j in range(5):  # 5 columns
                table[(i+1, j)].set_facecolor('lightyellow')

    plt.title(f"Comparison of Model Performance for {country} {year} (Sorted by CRPS)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    use_latex = False
    all_results = {}
    
    prediction_files = get_all_prediction_files(pred_preamble)
    
    for file in prediction_files:
        model_name = os.path.splitext(file)[0].replace(f"{country}_", "").replace(f"_{year}", "")
        file_path = os.path.join(pred_preamble, file)
        try:
            results = calculate_metrics(actuals_file, file_path)
            aggregate_metrics = calculate_aggregate_metrics(results)
            all_results[model_name] = aggregate_metrics
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            all_results[model_name] = {metric: float('nan') for metric in ['CRPS', 'MSE', 'MAE', 'IGN']}

    if use_latex:
        print_latex_table(all_results)
    else:
        visualize_table(all_results)