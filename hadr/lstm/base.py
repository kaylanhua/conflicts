import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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

def create_multivariate_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    
    return np.array(X), np.array(y)

country = 'drc'
# types: 'attention', 'base', 'multi'
lstm_type = 'multi'
# all_history = pd.read_csv('../../data/views/drc_no_rolling.csv')
all_history = pd.read_csv('drc_features.csv').dropna()
# all_history = all_history.head(140)

# NORMALIZATION SECTION 
if lstm_type == 'multi':
    X, y = all_history.drop(columns=['ged_sb']), all_history.ged_sb.values
    mm = MinMaxScaler() # can also use the one below that univariate models use
    ss = StandardScaler()
    X_trans = ss.fit_transform(X)
    y_trans = mm.fit_transform(y.reshape(-1, 1))
    
    X_ss, y_mm = create_multivariate_sequences(X_trans, y_trans, 12, 6) # tweak these hyperparameters
    print(f"Multivariate data shapes for X, y: {X_ss.shape}, y: {y_mm.shape}")
    
    total_samples = len(X)
    train_test_cutoff = round(0.90 * total_samples)
    X_train, X_test = X_ss[:train_test_cutoff], X_ss[train_test_cutoff:]
    y_train, y_test = y_mm[:train_test_cutoff], y_mm[train_test_cutoff:]
    
    print(f"X train and test shapes: {X_train.shape}, {X_test.shape}")
    print(f"y train and test shapes: {y_train.shape}, {y_test.shape}")
    
else:
    history = all_history['ged_sb'].tolist()
    
    # Perform Log Transformation on history
    # history = [np.log1p(x) for x in history]  # Using log1p to handle zero values

    # Perform Min-Max Normalization on history
    min_val = min(history)
    max_val = max(history)
    history_normalized = [(x - min_val) / (max_val - min_val) for x in history]
    history = history_normalized
    print(f"Normalized data - Min: {min(history):.4f}, Max: {max(history):.4f}")

    seq_length = 5
    X, y = create_sequences(history, seq_length)

    # split train and test
    train_size = int(len(y) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()


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
    
# Attention LSTM: https://medium.com/@aidant0001/revolutionizing-time-series-prediction-with-lstm-with-the-attention-mechanism-2bd126e9fdf1
class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(LSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), lstm_out)
        out = self.fc(context_vector.squeeze(1))
        return out 
    
# LSTM Multivariate: https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130
class LSTMMultivariate(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        
        output, (hn, cn) = self.lstm(x, (h0, c0))
        out = output[:, -1, :]
        # hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
        
    
# INITIALIZING THE MODEL -----------------------------------------------------------
input_size = 4 # univariate
hidden_size = 50
num_layers = 3
output_size = 6

if lstm_type == 'base':
    # LSTM model: hidden size 70, num layers 4, epochs 180, LR 0.01, seq_length 10
    model = LSTM(input_size, hidden_size, num_layers, output_size)
elif lstm_type == 'attention':
    model = LSTMAttention(input_size, hidden_size, output_size, num_layers)
elif lstm_type == 'multi':
    model = LSTMMultivariate(input_size, hidden_size, output_size, num_layers)
    
# TRAINING THE MODEL ----------------------------------------------------------------
learning_rate = 0.01
num_epochs = 200

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if lstm_type != 'multi':
    model.train()
    for epoch in range(num_epochs):
        train_outputs = model(X_train.unsqueeze(-1)).squeeze()
        optimizer.zero_grad()
        loss = criterion(train_outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
            
    print("!! Training complete !!")

def multivariate_training_loop(n_epochs, lstm, optimizer, loss_fn, X_train, y_train,
                  X_test, y_test):
    for epoch in range(n_epochs):
        lstm.train()
        outputs = lstm.forward(X_train)
        optimizer.zero_grad() # calculate the gradient, manually setting to 0
        loss = loss_fn(outputs, y_train)
        loss.backward() # calculates the loss of the loss function
        optimizer.step() # improve from loss, i.e backprop
        
        # test loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_fn(test_preds, y_test)
        
        if epoch % 10 == 0:
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, 
                                                                      loss.item(), 
                                                                      test_loss.item())) 


# TESTING THE MODEL ----------------------------------------------------------------

if lstm_type == 'multi':
    multivariate_training_loop(n_epochs=num_epochs,
              lstm=model,
              optimizer=optimizer,
              loss_fn=criterion,
              X_train=X_train,
              y_train=y_train,
              X_test=X_test,
              y_test=y_test)
    
    
    
else:
    with torch.no_grad():
        test_outputs = model(X_test.unsqueeze(-1)).squeeze()
        test_loss = criterion(test_outputs, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")
        print("!! Testing complete !!")
        

    all_outputs = np.concatenate((train_outputs.detach().numpy(), test_outputs.detach().numpy()))

    # Calculate the index where the test set starts
    test_start_index = len(history) - len(y_test) - seq_length

    # Plot the true values and the predictions
    plt.plot(history, label="True Values")
    plt.plot(range(seq_length, seq_length + len(all_outputs)), all_outputs, label="Predictions")
    plt.axvline(x=test_start_index, color='gray', linestyle='--', label="Test set start")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("LSTM Predictions vs True Values")
    plt.savefig(f'{country}_{lstm_type}__EP{num_epochs}__NL{num_layers}__HS{hidden_size}__SL{seq_length}.jpg')
    plt.close()