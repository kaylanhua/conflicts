import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

all_history = pd.read_csv('../../data/views/sri_lanka.csv')
history = all_history['ged_sb'].tolist()

# plt.plot(history[:30])
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.title(f"Historical Data for sri lanka")

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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model(X_train.unsqueeze(-1)).squeeze()
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
        
print("!! Training complete !!")


# TESTING THE MODEL ----------------------------------------------------------------
with torch.no_grad():
    test_outputs = model(X_test.unsqueeze(-1)).squeeze()
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")
    print("!! Testing complete !!")