# %% Packages
import numpy as np                                  # Import NumPy for numerical operations
import pandas as pd                                 # Import pandas for data manipulation
import torch                                        # Import PyTorch for tensor computations
import torch.nn as nn                               # Import PyTorch neural network module
import seaborn as sns                               # Import Seaborn for data visualization
import requests

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Set device to GPU if available, else CPU
print(f"Using device: {DEVICE}")                    # Print the device being used



# url = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
# local_path = 'cars.csv'

# response = requests.get(url)
# with open(local_path, 'wb') as file:
#     file.write(response.content)



# %% Data Import
cars_file = 'cars.csv'                              # Set the file name
cars = pd.read_csv(cars_file)                       # Read the CSV file into a DataFrame
cars.head()                                         # Display the first few rows of the data

# %% Visualize the Model
sns.scatterplot(x='wt', y='mpg', data=cars)         # Create a scatter plot of weight vs. mpg
sns.regplot(x='wt', y='mpg', data=cars)             # Add a regression line to the plot

# %% Convert Data to Tensor
X_list = cars.wt.values                             # Extract 'wt' column values as a NumPy array
X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)  # Convert to float32 NumPy array and reshape
y_list = cars.mpg.values.tolist()                   # Extract 'mpg' values as a list
y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)  # Convert to float32 NumPy array and reshape

X = torch.from_numpy(X_np).to(DEVICE)               # Convert input data to a PyTorch tensor and move to device
y = torch.tensor(y_list, device=DEVICE)             # Convert target data to a PyTorch tensor and move to device

# %% Training
w = torch.rand(1, requires_grad=True, dtype=torch.float64, device=DEVICE)  # Initialize weight with gradient tracking
b = torch.rand(1, requires_grad=True, dtype=torch.float64, device=DEVICE)  # Initialize bias with gradient tracking

num_epochs = 10                                    # Set the number of training epochs
learning_rate = 1e-3                                # Set the learning rate

for epoch in range(num_epochs):                     # Loop over each epoch
    for i in range(len(X)):                         # Loop over each data point
        # Forward pass
        y_predict = X[i] * w + b                    # Compute predicted y value
        # Calculate loss
        loss_tensor = torch.pow(y_predict - y[i], 2)  # Compute squared error loss
        # Backward pass
        loss_tensor.backward()                      # Compute gradients
        # Extract losses
        loss_value = loss_tensor.data.item()        # Get the loss value as a Python number
        # Update weights and biases
        with torch.no_grad():                       # Temporarily disable gradient tracking
            w -= w.grad * learning_rate             # Update weight
            b -= b.grad * learning_rate             # Update bias
            w.grad.zero_()                          # Reset gradients for next iteration
            b.grad.zero_()
    print(loss_value)                               # Print loss after each epoch

# %% Check Results
print(f"Weight: {w.item()}, Bias: {b.item()}")      # Output the final learned weight and bias

# %% Predict and Plot the Results
y_pred = (torch.tensor(X_list, device=DEVICE) * w + b).detach().cpu().numpy()  # Compute predictions and move to CPU
sns.scatterplot(x=X_list, y=y_list)                  # Plot original data points
sns.lineplot(x=X_list, y=y_pred, color='red')        # Plot the regression line

# %% (Statistical) Linear Regression
from sklearn.linear_model import LinearRegression   # Import LinearRegression from scikit-learn
reg = LinearRegression().fit(X_np, y_list)          # Fit a linear regression model using scikit-learn
print(f"Slope: {reg.coef_}, Bias: {reg.intercept_}")  # Output the slope and intercept from the model




import matplotlib.pyplot as plt

# Generate predictions from your trained model
y_pred = (torch.tensor(X_list, device='cuda') * w + b).detach().cpu().numpy()

# Scatter plot of actual data
sns.scatterplot(x=X_list, y=y_list, label="Actual Data")

# Regression line from trained model
sns.lineplot(x=X_list, y=y_pred, color='red', label="Model Prediction")

# Add labels and title
plt.xlabel("Car Weight (wt)")
plt.ylabel("Miles Per Gallon (mpg)")
plt.title("Trained Model: Predicted vs Actual")
plt.legend()
plt.show()




# %% Create Graph Visualization
import os                                           # Import os for operating system dependent functionality
from torchviz import make_dot                       # Import make_dot for visualizing computation graphs
os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'  # Add GraphViz to system PATH
make_dot(loss_tensor)                               # Generate and display the computation graph
