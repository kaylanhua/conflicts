import pandas as pd
import numpy as np
from crps import crps_ensemble

# read actuals
actuals = pd.read_csv("DRC_cm_actuals_2019.csv")
observation = actuals['outcome'].tolist()
print(f"observation: {observation}")

# read forecasts
benchmark = pd.read_csv("DRC_Conflictology_2019.csv")
print(f"benchmark: {benchmark}")
# Group by month_id and aggregate values into a list
forecasts = benchmark.groupby('month_id')['outcome'].apply(list).reset_index(name='forecasts')
print(f"forecasts: {forecasts}")

# Convert the 'forecasts' column to a numpy array
forecasts = np.array(forecasts['forecasts'].tolist())

for i in range(len(forecasts)):
    score = crps_ensemble(forecasts[i], observation[i])
    print("--------------------------------")
    print(f"month_id: {i}, forecast: {forecasts[i]}, observation: {observation[i]}")
    print(f"CRPS Score: {score:.4f}")