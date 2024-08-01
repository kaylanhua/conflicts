import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
from microprediction import MicroReader
# from microconventions import api_url

# # TOY DATA, it does not work 
# reader = MicroReader()
# stream_names = reader.get_stream_names()
# stream = stream_names[50]
# history = reader.get_lagged_values(name=stream)

# plt.plot(history)
# plt.title(f'Historical Values for Stream Named {stream}')
# plt.xlabel('Lag')
# plt.ylabel('Value')
# plt.savefig('plot.png')


# DATA SECTION ----------------------------------------------------------------
def create_sequences(data, seq_length):
    # discards data after the last mult of seq_length
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length - 1):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)

history = None

seq_length = 10
X, y = create_sequences(history, seq_length)

# split train and test
train_size = int(len(y) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# convert to tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()


# MODEL SECTION ----------------------------------------------------------------
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
input_size = 1 # univariate
hidden_size = 50
num_layers = 1 
output_size = 1
model = LSTM(input_size, hidden_size, num_layers, output_size)



# TRAINING THE MODEL ----------------------------------------------------------------
learning_rate = 0.01
num_epochs = 100