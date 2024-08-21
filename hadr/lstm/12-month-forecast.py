import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# DATA SECTION ----------------------------------------------------------------
def create_sequences(data, seq_length, n_ahead):
    # discards data after the last mult of seq_length
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length - n_ahead):
        x = data[i:i+seq_length]
        y = data[i+seq_length+n_ahead-1]
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)

# TOGGLES
country = 'drc'
lstm_type = 'base'

# Load and preprocess data
all_history = pd.read_csv('../../data/views/input_data_2019.csv')
all_history = all_history[all_history['country_id'] == 167]

# Filter data up to October of the previous year (2018 in this case)
train_data = all_history['ged_sb'].tolist()

# Normalize the data
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(np.array(train_data).reshape(-1, 1)).flatten()

# Prepare sequences for training
seq_length = 12  # Use 12 months of history
X_train, y_train = create_sequences(train_data_normalized, seq_length, 1)

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
        # fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B = x.size(0)
        
        # FOR BIGGER DATA: if we want to persist the hidden states (this is for when each epoch calls forward multiple times)
        # if self.hidden is None:
        #     self.hidden = (torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device),
        #                    torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        
        # hidden states
        h0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        # cell states
        c0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        
        # output of size B, S (seq len), H (hidden size)
        # _ contains final hidden state and cell state, not used
        out, _ = self.lstm(x, (h0, c0))
        
        # output of size B, S, H -> B, H
        out = self.fc(out[:, -1, :])
        
        # output is of size B, O (output size)
        return out
    
# INITIALIZING THE MODEL -----------------------------------------------------------
input_size = 1
hidden_size = 50
num_layers = 3
output_size = 1

model = LSTM(input_size, hidden_size, num_layers, output_size)

# TRAINING THE MODEL ----------------------------------------------------------------
learning_rate = 0.01
num_epochs = 300

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    train_outputs = model(X_train.unsqueeze(-1)).squeeze()
    optimizer.zero_grad()
    loss = criterion(train_outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

print("!! Training complete !!")

# FORECASTING
model.eval()
forecast_horizon = 12  # Forecast for 12 months
forecasts = []

# Use the last sequence from the training data as the initial input
input_seq = train_data_normalized[-seq_length:]

for _ in range(forecast_horizon):
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1)
        forecast = model(input_tensor).item()
        forecasts.append(forecast)
        input_seq = np.append(input_seq[1:], forecast)

# Inverse transform the forecasts
forecasts_denormalized = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()


starting_month = 469
forecast_data = {
    'month_id': range(starting_month, starting_month + 12),
    'country_id': [167] * 12,  # Assuming country_id 167 for DRC
    'outcome': forecasts_denormalized
}

forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_csv('DRC_lstm_forecasts_2019.csv', index=False)