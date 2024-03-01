# Model Design
import agentpy as ap 
import numpy as np 
import pandas as pd
from boids_model import BoidsModel

# Visualisation
from visualisation import animation_plot, animation_plot_single
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import ast

results = pd.read_csv('boids_statistics_results.csv')

def process_data(X, Y, normalize=True):
    # Split the data into training+validation set and test set
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Split the training+validation set into separate training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=42)

    # Convert pandas dataframes to numpy arrays (if not already in numpy format)
    X_train_np, Y_train_np = X_train.values, Y_train.values
    X_val_np, Y_val_np = X_val.values, Y_val.values
    X_test_np, Y_test_np = X_test.values, Y_test.values

    if normalize:
        # Initialize scalers for X and Y
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()

        # Fit and transform the training data
        X_train, Y_train= scaler_X.fit_transform(X_train_np), scaler_Y.fit_transform(Y_train_np)

        # Transform the validation and test data
        X_val, Y_val = scaler_X.transform(X_val_np), scaler_Y.transform(Y_val_np)
        X_test, Y_test = scaler_X.transform(X_test_np), scaler_Y.transform(Y_test_np)

    
    # Convert the numpy arrays back to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

output_parameters = ['final_alignment', 'cohesion_separation_ratio', 'flock_density']
knobs = ['cohesion_strength', 'seperation_strength', 'alignment_strength', 'border_strength']

Y = results[output_parameters]
X =  results[knobs]
sns.pairplot(X).savefig('./cov_matrix.png')

X_train, Y_train, X_val, Y_val, X_test, Y_test = process_data(X, Y, normalize=True)

# Neural Network Architecture
class Net(nn.Module):
    def __init__(self, X, Y):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X.size(1), 128)  # Input layer to 128 neurons
        self.fc2 = nn.Linear(128, 64)         # Hidden layer (128 to 64 neurons)
        self.fc3 = nn.Linear(64, 32)   # Hidden layer (64 to 32 neurons)
        self.fc4 = nn.Linear(32, Y.size(1))   # Output layer (32 to number of target features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # No activation function in the output layer
        return x


class SuperEnhancedNet(nn.Module):
    def __init__(self, X, Y):
        super(SuperEnhancedNet, self).__init__()
        # Significantly increase the capacity of the network
        self.fc1 = nn.Linear(X.size(1), 1024)  # Input layer to 1024 neurons
        self.bn1 = nn.BatchNorm1d(1024)        # Batch normalization layer
        self.fc2 = nn.Linear(1024, 512)        # First hidden layer
        self.bn2 = nn.BatchNorm1d(512)         # Batch normalization layer
        self.fc3 = nn.Linear(512, 512)         # Second hidden layer
        self.dropout1 = nn.Dropout(0.5)        # Dropout layer for regularization
        self.fc4 = nn.Linear(512, 256)         # Third hidden layer
        self.bn3 = nn.BatchNorm1d(256)         # Batch normalization layer
        self.fc5 = nn.Linear(256, 256)         # Fourth hidden layer
        self.dropout2 = nn.Dropout(0.5)        # Another dropout layer
        self.fc6 = nn.Linear(256, 128)         # Fifth hidden layer
        self.bn4 = nn.BatchNorm1d(128)         # Batch normalization layer
        self.fc7 = nn.Linear(128, 64)          # Sixth hidden layer
        self.fc8 = nn.Linear(64, Y.size(1))    # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout1(x)
        x = F.relu(self.fc4(x))
        x = self.bn3(x)
        x = F.relu(self.fc5(x))
        x = self.dropout2(x)
        x = F.relu(self.fc6(x))
        x = self.bn4(x)
        x = F.relu(self.fc7(x))
        x = self.fc8(x)  # No activation function in the output layer
        return x



model = SuperEnhancedNet(X_train, Y_train)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
for epoch in range(1000):  # Number of epochs
    optimizer.zero_grad()   # Zero the gradient buffers
    output = model(X_train)         # Pass the batch through the network
    loss = criterion(output, Y_train)  # Compute the loss
    loss.backward()         # Backpropagation
    optimizer.step()        # Update weights

    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# Assuming 'model' is your trained model and 'loss_fn' is your loss function
model.eval()  # Set the model to evaluation mode

with torch.no_grad():  # Do not calculate gradients to save memory
    # Calculate the predictions for the test set
    Y_pred = model(X_test)

    # Calculate the loss for the test set
    test_loss = criterion(Y_pred, Y_test)

print(f'Test loss: {test_loss.item()}')

# Calculate the predictions for the test set
Y_pred = model(X_test)

# Convert the predictions and true values to NumPy arrays
Y_pred_np = Y_pred.detach().numpy()
Y_test_np = Y_test.numpy()

# Calculate the metrics
mae = mean_absolute_error(Y_test_np, Y_pred_np)
mse = mean_squared_error(Y_test_np, Y_pred_np)
r2 = r2_score(Y_test_np, Y_pred_np)

print(f'MAE: {mae}, MSE: {mse}, R^2: {r2}')


# Assuming Y_test_np and Y_pred_np are 2D arrays with 4 columns each
for i in range(Y_pred.size(1)):
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_pred_np[:, i], Y_test_np[:, i])
    plt.xlabel('Predictions')
    plt.ylabel('True Values')
    plt.title(f'Y Variable {i+1}')

    # Add a diagonal line
    limits = [np.min([plt.xlim(), plt.ylim()]),  # Find the lower limit
              np.max([plt.xlim(), plt.ylim()])]  # Find the upper limit
    plt.xlim(limits)
    plt.ylim(limits)
    plt.plot(limits, limits, color='black', alpha=0.5, linestyle='--')

    plt.savefig(f"./figure_{i}.png")