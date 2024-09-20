import argparse
import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from fuzzywuzzy import process
# Add the project root directory to the Python path
import sys
sys.path.append('/Users/kaylahuang/Desktop/conflicts/hadr')

# Add argument parsing
parser = argparse.ArgumentParser(description='LSTM model for 12-month forecast')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
args = parser.parse_args()

# DATA SECTION ----------------------------------------------------------------
def create_sequences(data, seq_length, n_ahead):
    # discards data after the last mult of seq_length
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length - n_ahead):
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+seq_length+n_ahead]
        xs.append(x)
        ys.append(y)
    
    print("shape of xs: ", np.array(xs).shape)
    print("shape of ys: ", np.array(ys).shape)
    return np.array(xs), np.array(ys)

# TOGGLES
country = 'afghanistan'
lstm_type = 'base'

country_key_df = pd.read_csv('../../data/views/country_key.csv')

def get_country_id(country_name):
    best_match = process.extractOne(country_name, country_key_df['name'])
    if best_match[1] >= 80:  # Threshold for a good match
        return country_key_df[country_key_df['name'] == best_match[0]]['id'].values[0]
    else:
        raise ValueError(f"No close match found for country name: {country_name}")

COUNTRY_ID = get_country_id(country)

# Load and preprocess data
all_history = pd.read_csv('../../data/views/input_data_2019.csv')
all_history = all_history[all_history['country_id'] == COUNTRY_ID]

# Filter data up to October of the previous year (2018 in this case)
train_data = all_history['ged_sb'].tolist()

# Normalize the data
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(np.array(train_data).reshape(-1, 1)).flatten()

# Prepare sequences for training
seq_length = 24  
n_ahead = 15
X_train, y_train = create_sequences(train_data_normalized, seq_length, n_ahead)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# MODEL SECTION ----------------------------------------------------------------

# Base LSTM: https://medium.com/@mike.roweprediger/using-pytorch-to-train-an-lstm-forecasting-model-e5a04b6e0e67
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# INITIALIZING THE MODEL -----------------------------------------------------------
input_size = 1
hidden_size = 50
num_layers = 3
output_size = n_ahead

model = LSTM(input_size, hidden_size, num_layers, output_size)

# TRAINING THE MODEL ----------------------------------------------------------------
learning_rate = 0.01
num_epochs = args.epochs  # Use the command line argument here

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    train_outputs = model(X_train.unsqueeze(-1))
    optimizer.zero_grad()
    loss = criterion(train_outputs, y_train[:, :n_ahead]) 
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

print("!! Training complete !!")

# FORECASTING
model.eval()
forecast_horizon = n_ahead 
forecasts = []

# Use the last sequence from the training data as the initial input
input_seq = train_data_normalized[-seq_length:]

with torch.no_grad():
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1)
    forecasts = model(input_tensor).squeeze().numpy()

# Inverse transform the forecasts
forecasts_denormalized = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()

print(forecasts_denormalized)
print(len(forecasts_denormalized))
print(forecasts_denormalized[3:])
print(len(forecasts_denormalized[3:]))

starting_month = 469
forecast_data = {
    'month_id': range(starting_month, starting_month + 12),
    'country_id': [167] * 12,  # Assuming country_id 167 for DRC
    'new_forecast': forecasts_denormalized[3:]  # start in jan
}

new_forecast_df = pd.DataFrame(forecast_data)
save_filename = f'../views_testing/{country}_lstm_forecasts_2019.csv'

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