## GENERAL IMPORTS
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# Constants
if len(sys.argv) < 2:
    print("Please provide a country name as an argument.")
    sys.exit(1)

country = sys.argv[1].lower()
year = sys.argv[2]
pred_preamble = f"results/{country}/{year}/"
if not os.path.exists(pred_preamble):
    print(f"Folder {pred_preamble} does not exist. Please try again with a valid country and year.")
    sys.exit(1)
    
actuals_file = pred_preamble + f"{country}_cm_actuals_{year}.csv"

def heaviside_step(x):
    """Heaviside step function."""
    return np.where(x >= 0, 1.0, 0.0)

def crps_ensemble(forecasts, observation):
    """
    Calculate the Continuous Rank Probability Score (CRPS) for an ensemble forecast.
    
    :param forecasts: Array of forecast ensemble members
    :param observation: Actual observed value
    :return: CRPS score
    """
    n = len(forecasts)
    sorted_forecasts = np.sort(forecasts)
    
    # Calculate empirical CDF
    def empirical_cdf(x):
        return np.sum(heaviside_step(x - sorted_forecasts)) / n
    
    # Vectorize the empirical CDF function
    v_empirical_cdf = np.vectorize(empirical_cdf)
    
    # Calculate CRPS using numerical integration
    x = np.linspace(min(np.min(forecasts), observation) - 1, 
                    max(np.max(forecasts), observation) + 1, 
                    1000)
    integrand = (v_empirical_cdf(x) - heaviside_step(x - observation))**2
    crps = np.trapz(integrand, x)
    
    return crps

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

    # Create a Tkinter window
    root = tk.Tk()
    root.title(f"Comparison of Model Performance for {country} {year}")

    # Create a frame to hold the canvas
    frame = ttk.Frame(root)
    frame.grid(row=0, column=0, sticky="nsew")

    # Create a canvas that can scroll vertically
    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create the figure and axis
    fig = Figure(figsize=(12, len(sorted_results) * 0.5 + 1))
    ax = fig.add_subplot(111)
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

    # Create a FigureCanvasTkAgg object
    canvas_widget = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_widget.draw()

    # Add the canvas to the Tkinter window
    canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Pack the scrollbar and canvas
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Configure the frame to expand
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Set a fixed height for the window (adjust as needed)
    root.geometry(f"800x600")

    # Start the Tkinter event loop
    root.mainloop()

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