import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Preprocessing steps
data = pd.read_csv("loan_approval_dataset.csv")
loan_status = data["loan_status"]  # Save loan_status for evaluation
data = data.drop(columns=["loan_status"], axis=1)

# Label encode 'education' and 'self_employed' columns
data["education"] = data["education"].map({" Not Graduate": 0, " Graduate": 1})
data["self_employed"] = data["self_employed"].map({" No": 0, " Yes": 1})

# Separate features and target variable
X = data.drop(columns=["loan_id", "loan_amount"])
y = data["loan_amount"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, loan_status_train, loan_status_test = (
    train_test_split(X, y, loan_status, test_size=0.2, random_state=42)
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Scale the target variable as well
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Parameters for C51
input_size = X_train.shape[1]
output_size = 51  # Number of atoms in C51
learning_rate = 0.001
gamma = 0.99
v_min = -10
v_max = 10
n_atoms = 51
delta_z = (v_max - v_min) / (n_atoms - 1)
z = torch.linspace(v_min, v_max, n_atoms)


# Define the C51 Network
class C51Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(C51Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.output_layer(x)
        return torch.softmax(logits.view(-1, n_atoms), dim=1)


# Initialize network
c51_network = C51Network(input_size, output_size)
optimizer = optim.Adam(c51_network.parameters(), lr=learning_rate)

# Train the C51 model
n_episodes = 5
for episode in range(n_episodes):
    for i in range(len(X_train)):
        state = torch.FloatTensor(X_train[i]).unsqueeze(0)
        target_distribution = torch.zeros((1, n_atoms))

        with torch.no_grad():
            mean = y_train_scaled[i]
            b = torch.tensor((mean - v_min) / delta_z)
            l = int(torch.floor(b).item())
            u = int(torch.ceil(b).item())

            target_distribution[0, l] += u - b
            if u < n_atoms:
                target_distribution[0, u] += b - l

        # Forward pass
        predicted_distribution = c51_network(state)

        # Compute loss
        loss = -(target_distribution * torch.log(predicted_distribution)).sum()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Testing
y_pred = []
for state in X_test:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    predicted_distribution = c51_network(state_tensor)
    expected_value = torch.sum(predicted_distribution * z, dim=1).item()
    predicted_loan_amount = y_scaler.inverse_transform([[expected_value]])[0][0]
    y_pred.append(predicted_loan_amount)

# Generate predicted loan status based on predicted loan amount
y_pred_loan_status = [
    "Approved" if pred >= actual else "Rejected" for pred, actual in zip(y_pred, y_test)
]

# Generate classification report
print("\nClassification Report:")
print(classification_report(loan_status_test, y_pred_loan_status))

# Testing on custom input
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
custom_input["education"] = custom_input["education"].map(
    {" Not Graduate": 0, " Graduate": 1}
)
custom_input["self_employed"] = custom_input["self_employed"].map({" No": 0, " Yes": 1})
X_custom = custom_input.drop(columns=["loan_amount"])
y_custom = custom_input["loan_amount"]
X_custom = scaler.transform(X_custom)

# Predicting using C51
y_custom_pred = []
for state in X_custom:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    predicted_distribution = c51_network(state_tensor)
    expected_value = torch.sum(predicted_distribution * z, dim=1).item()
    predicted_loan_amount = y_scaler.inverse_transform([[expected_value]])[0][0]
    y_custom_pred.append(predicted_loan_amount)

print(f"\n\nPredicted loan amounts: \n{y_custom_pred}")
print(f"\nActual applied loan amounts: \n{y_custom.tolist()}")

# Loan approval predictions
print("\n\nLoan Approval Predictions:")
for i in range(len(y_custom_pred)):
    if y_custom_pred[i] > y_custom.iloc[i]:
        print(f"Test Case {i+1}: Loan will be approved")
    else:
        print(f"Test Case {i+1}: Loan will not be approved")
