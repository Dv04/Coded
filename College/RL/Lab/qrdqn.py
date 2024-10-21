import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from termcolor import colored

# Preprocessing steps (from your template)
print("Reading dataset...")
data = pd.read_csv("loan_approval_dataset.csv")
print("Dataset read successfully.")
data = data.drop(columns=["loan_status"], axis=1)

# Label encode 'education' and 'self_employed' columns
print("Label encoding columns...")
data["education"] = data["education"].map({" Not Graduate": 0, " Graduate": 1})
data["self_employed"] = data["self_employed"].map({" No": 0, " Yes": 1})
print("Label encoding completed.")

# Separate features and target variable
print("Separating features and target variable...")
X = data.drop(columns=["loan_id", "loan_amount"])
y = data["loan_amount"]
print("Features and target separated.")

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split completed.")

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature standardization completed.")

# Scale the target variable as well
print("Scaling target variable...")
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
print("Target scaling completed.")

# Parameters for QR-DQN
input_size = X_train.shape[1]
n_quantiles = 51
learning_rate = 0.001
gamma = 0.99

tau = torch.linspace(0.0, 1.0, n_quantiles + 1)[1:]  # Quantiles to estimate


# Define the QR-DQN Network
class QRDQNNetwork(nn.Module):
    def __init__(self, input_size, n_quantiles):
        super(QRDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, n_quantiles)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        quantiles = self.output_layer(x)
        return quantiles


# Initialize network
print("Initializing QR-DQN network...")
qr_dqn_network = QRDQNNetwork(input_size, n_quantiles)
optimizer = optim.Adam(qr_dqn_network.parameters(), lr=learning_rate)
print("Network initialized.")

# Train the QR-DQN model
print("Starting training...")
n_episodes = 5
for episode in range(n_episodes):
    print(f"Episode {episode + 1}/{n_episodes}")
    for i in range(len(X_train)):
        state = torch.FloatTensor(X_train[i]).unsqueeze(0)
        target_quantiles = torch.zeros((1, n_quantiles))

        with torch.no_grad():
            target = y_train_scaled[i]
            td_error = target - qr_dqn_network(state)
            target_quantiles = td_error * (tau - (td_error < 0).float())

        # Forward pass
        predicted_quantiles = qr_dqn_network(state)

        # Compute loss
        loss = torch.mean(target_quantiles * predicted_quantiles)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training completed.")

# Testing
print("Starting testing...")
y_pred = []
for state in X_test:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    predicted_quantiles = qr_dqn_network(state_tensor)
    expected_value = torch.mean(predicted_quantiles).item()
    predicted_loan_amount = y_scaler.inverse_transform([[expected_value]])[0][0]
    y_pred.append(predicted_loan_amount)
    print(f"Test state: Predicted loan amount: {predicted_loan_amount}")

# Testing on custom input (as per template)
print("Testing on custom input...")
custom_input = pd.DataFrame(
    {
        "no_of_dependents": [2, 5, 3, 0],
        "education": [" Graduate", " Not Graduate", " Graduate", " Graduate"],
        "self_employed": [" No", " Yes", " No", " No"],
        "income_annum": [3900000, 1200000, 5000000, 300000],
        "loan_amount": [12300000, 5000000, 1500000, 10000000],
        "loan_term": [18, 12, 24, 18],
        "cibil_score": [700, 600, 750, 800],
        "residential_assets_value": [7600000, 200000, 10000000, 5000000],
        "commercial_assets_value": [690000, 1000000, 500000, 3000000],
        "luxury_assets_value": [1300000, 200000, 10000, 5000000],
        "bank_asset_value": [2800000, 50000, 200000, 300000],
    }
)

# Preprocessing custom input
print("Preprocessing custom input...")
custom_input["education"] = custom_input["education"].map(
    {" Not Graduate": 0, " Graduate": 1}
)
custom_input["self_employed"] = custom_input["self_employed"].map({" No": 0, " Yes": 1})
X_custom = custom_input.drop(columns=["loan_amount"])
y_custom = custom_input["loan_amount"]
X_custom = scaler.transform(X_custom)
print("Custom input preprocessing completed.")

# Predicting using QR-DQN
print("Predicting on custom input...")
y_custom_pred = []
for state in X_custom:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    predicted_quantiles = qr_dqn_network(state_tensor)
    expected_value = torch.mean(predicted_quantiles).item()
    predicted_loan_amount = y_scaler.inverse_transform([[expected_value]])[0][0]
    y_custom_pred.append(predicted_loan_amount)
    print(f"Custom state: Predicted loan amount: {predicted_loan_amount}")

print(f"\n\nPredicted loan amounts: \n{y_custom_pred}")
print(f"\nActual applied loan amounts: \n{y_custom.tolist()}")

# Loan approval predictions
print("\n\nPredictions:")
for i in range(len(y_custom_pred)):
    if y_custom_pred[i] > y_custom.iloc[i]:
        print(colored(f"Test Case {i+1}: Loan will be approved", "green"))
    else:
        print(colored(f"Test Case {i+1}: Loan will not be approved", "red"))
