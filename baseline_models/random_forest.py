# STANDARD IMPORTS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
import xgboost as xgb
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.svm import SVR

# PERSONAL IMPORTS
import sys
sys.path.append("/Users/kaylahuang/Desktop/conflicts/components/")
sys.path.append("/Users/kaylahuang/Desktop/conflicts/yun_ff/")
from views_cleaner import VIEWSCleaner
import enums

# CONSTANTS
filename = '../data/views/cm_features.parquet'
gw_id = 490
country = "drc"
year = 2019

# FUNCTIONS 
def load_data(filename, gw_id):
    cleaner = VIEWSCleaner(filename, gw_id, trim_full=False)
    original_features = cleaner.features # already aggregated by month
    X = original_features.copy()
    print(X.shape)
    # X.to_csv(f'../data/views/{country}.csv')
    cleaner.plot(n=4)
    return X, cleaner

def autocorrelation_plot(X):
    acf_values = acf(X['ged_sb'], alpha=0.05)
    significant_lags = sum(1 for lower, upper in acf_values[1:] if lower > 0)
    print(f"Number of significant autocorrelation lags: {significant_lags}")
    return significant_lags
    # plot_acf(X['ged_sb'], lags=36)
    
def create_sliding_window(data, window_size):
    for i in range(1, window_size + 1):
        data[f'lag_{i}'] = data['ged_sb'].shift(i)
    return data.dropna()


X, cleaner = load_data(filename, gw_id)
window_size = 5 # dictated by the above
X = create_sliding_window(X, window_size)
X.head() # last WINDOW_SIZE columns are the lags

# ADDING VARS: set war var
war_dates = enums.WAR_DATES.get(country)
if war_dates:
    cleaner.set_war_var(X, war_dates)

X = cleaner.features_war_dates
print(X.shape)

# ADDING VARS: set peak var w poisson process

data = X['ged_sb'].copy()
data = pd.Series(data)


height_threshold = data.mean() + 2 * data.std()
peaks, _ = find_peaks(data, height=height_threshold)

peak_indices = list(peaks)
print(peak_indices)

peaks_df = pd.DataFrame({
    'Month': X['month_id'].iloc[peak_indices],
    'Peak Value': data.iloc[peak_indices]
})

print("peaks_df")
print(peaks_df.head())

# Plot the data and peaks
plt.figure(figsize=(10, 6))
plt.plot(X["month_id"], data, label='Monthly Fatalities')
plt.plot(peaks_df["Month"], peaks_df["Peak Value"], "x", label='Peaks')
plt.title('Monthly Fatalities with Peaks')
plt.xlabel('Month')
plt.ylabel('Number of Fatalities')
plt.legend()
plt.show()

## POISSON PROCESS CODE
# calculate inter-peak intervals
inter_peak_intervals = np.diff(peaks)

# estimate average time between peaks (lambda)
lambda_est = 1 / np.mean(inter_peak_intervals)

# model the poisson process
def poisson_process_prob(t, lambda_est):
    return 1 - np.exp(-lambda_est * t)

# est the probability of future peaks
future_months = np.arange(1, 25)  # Example: next 24 months
probabilities = poisson_process_prob(future_months, lambda_est)
print(probabilities)

# Create a new column "peak_prob" in X
X['peak_prob'] = X['since_war_start'].apply(lambda t: poisson_process_prob(t, lambda_est))

def remove_low_variance_features_cv(df, threshold=0.01):
    cv = df.std() / df.mean()
    features_kept = cv[cv > threshold].index
    df_reduced = df[features_kept]
    return df_reduced

threshold = 0.01
df_reduced = remove_low_variance_features_cv(X, threshold)

print(X.shape)
print(df_reduced.shape)
print("Columns removed:\n", set(X.columns) - set(df_reduced.columns))

X = df_reduced

def predict_date(input_month, model_name='RF'):
    ## TRAINING DATA
    pre_date = X[X['month_id'] < input_month].tail(36)
    X_train = pre_date.drop(columns=["ged_sb", "month_id"]) # last 36
    y_train = pre_date["ged_sb"] # corresponding fatalities

    ## MODEL SELECTION AND FIT
    model = None
    if model_name == 'RF':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'XGB':
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    elif model_name == 'EN':
        model = ElasticNet(alpha=1.0, l1_ratio=0.1, max_iter=1000, random_state=40)
    elif model_name == 'Lasso':
        model = Lasso(alpha=1.0, max_iter=1000, random_state=42)
    elif model_name == 'Ridge':
        model = Ridge(alpha=1.0, max_iter=1000, random_state=42)
        
    model.fit(X_train, y_train)
    
    ## MODEL PREDICTION
    row = X[X['month_id'] == input_month]
    prediction = model.predict(row.drop(columns=["ged_sb", "month_id"]))
    
    if model_name in ['EN', 'Lasso', 'Ridge']:
        prediction = np.clip(prediction, a_min=0, a_max=None)
        
    importances = []
    if model_name in ['RF', 'XGB']:
        importances = model.feature_importances_
    
    return prediction[0], row["ged_sb"].values[0], importances

# set the prediction function
prediction_function = lambda date: predict_date(date, model_name='RF')

START = 469
WIDTH = 12
dates_to_predict = [i for i in range(START, START+WIDTH)]

ground_truth = []
predictions = []

for date in dates_to_predict:
    p, t, feature_imp = prediction_function(date)
    ground_truth.append(t)
    predictions.append(p)

plt.figure(figsize=(10, 6))

plt.plot(dates_to_predict, ground_truth, label='Actual', marker='o')
plt.plot(dates_to_predict, predictions, label='Predicted', marker='x')
plt.legend(['Ground Truth', 'Prediction'])
plt.show()

df = {
    'month_id': dates_to_predict,
    'country_id': [gw_id] * WIDTH,
    'new_forecast': predictions
}

new_forecast_df = pd.DataFrame(df)

save_filename = f'../hadr/views_testing/{country}_rf_forecasts_{year}.csv'
import os 

# Check if the file already exists
if os.path.isfile(save_filename):
    # If it exists, read the existing file
    existing_df = pd.read_csv(save_filename)

    # Merge the existing dataframe with the new forecast
    merged_df = existing_df.merge(new_forecast_df[['month_id', 'new_forecast']], on='month_id', how='left')
    
    # Rename the new column to avoid conflicts (e.g., 'forecast_2', 'forecast_3', etc.)
    new_col_name = f'forecast_{len([col for col in merged_df.columns if col.startswith("forecast")])}'
    merged_df = merged_df.rename(columns={'new_forecast': new_col_name})
    
    # Save the merged dataframe
    merged_df.to_csv(save_filename, index=False)
else:
    # If the file doesn't exist, create it with the current forecast
    new_forecast_df = new_forecast_df.rename(columns={'new_forecast': 'forecast_1'})
    new_forecast_df.to_csv(save_filename, index=False)

print(f"Forecasts saved to {save_filename}")