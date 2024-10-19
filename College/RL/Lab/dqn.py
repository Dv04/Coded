import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
from termcolor import colored

# Preprocessing steps (from your template)
data = pd.read_csv("loan_approval_dataset.csv")
data = data.drop(columns=["loan_status"], axis=1)

# Label encode 'education' and 'self_employed' columns
data["education"] = data["education"].map({" Not Graduate": 0, " Graduate": 1})
data["self_employed"] = data["self_employed"].map({" No": 0, " Yes": 1})

# Separate features and target variable
X = data.drop(columns=["loan_id", "loan_amount"])
y = data["loan_amount"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Scale the target variable as well
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Discretize loan_amount (actions) into bins (for DQN)
n_actions = 10  # Define the number of discrete actions (loan amount bins)
loan_amount_bins = np.linspace(min(y_train), max(y_train), n_actions + 1)
y_train_discretized = (
    np.digitize(y_train, loan_amount_bins) - 1
)  # Discretize the loan amount


# Define the Deep Q-Network (DQN) architecture
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Parameters for DQN
input_size = X_train.shape[1]  # Number of features (states)
output_size = n_actions  # Number of actions (discretized loan amounts)
learning_rate = 0.001
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration-exploitation tradeoff
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory_size = 10000
target_update_frequency = 10  # Frequency of updating the target network

# Initialize the DQN model and optimizer
model = DQNetwork(input_size, output_size)
target_model = DQNetwork(input_size, output_size)
target_model.load_state_dict(model.state_dict())  # Copy the weights
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Replay memory
memory = deque(maxlen=memory_size)


# Store experiences in the replay buffer
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


# Sample a batch of experiences
def sample_experiences(batch_size):
    return random.sample(memory, batch_size)


# Epsilon-greedy action selection
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, n_actions)  # Exploration
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor)
        return torch.argmax(q_values).item()  # Exploitation


# Update the target network
def update_target_network():
    target_model.load_state_dict(model.state_dict())


# Train the DQN
n_episodes = 5
for episode in range(n_episodes):
    for i, state in enumerate(X_train):
        action = select_action(state, epsilon)
        next_state = X_train[i]
        reward = -np.abs(
            loan_amount_bins[action] - y_train.iloc[i]
        )  # Reward is negative error

        # Store experience in the replay buffer
        store_experience(state, action, reward, next_state, False)

        # Sample a batch of experiences from the memory
        if len(memory) > batch_size:
            experiences = sample_experiences(batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)

            # Compute target Q-values
            q_targets_next = target_model(next_states).max(1)[0].detach()
            q_targets = rewards + (gamma * q_targets_next)

            # Compute predicted Q-values
            q_values = model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Compute loss and update the model
            loss = loss_fn(q_values, q_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update the target network
    if episode % target_update_frequency == 0:
        update_target_network()

    # Decay epsilon (reduce exploration)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Testing
y_pred_discretized = []
for state in X_test:
    action = select_action(state, epsilon=0.0)  # Pure exploitation
    predicted_loan_amount = loan_amount_bins[action]
    y_pred_discretized.append(predicted_loan_amount)

# Inverse scaling the predicted values
y_pred = y_scaler.inverse_transform(
    np.array(y_pred_discretized).reshape(-1, 1)
).flatten()

# Testing on custom input (as per template)
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

# Predicting using DQN
y_custom_pred_discretized = []
for state in X_custom:
    action = select_action(state, epsilon=0.0)  # Pure exploitation
    predicted_loan_amount = loan_amount_bins[action]
    y_custom_pred_discretized.append(predicted_loan_amount)

# Inverse scaling custom predictions
y_custom_pred = y_scaler.inverse_transform(
    np.array(y_custom_pred_discretized).reshape(-1, 1)
).flatten()

print(f"\n\nPredicted loan amounts: \n{y_custom_pred}")
print(f"\nActual applied loan amounts: \n{y_custom}")

# Loan approval predictions
print("\n\nPredictions:")
for i in range(len(y_custom_pred)):
    if y_custom_pred[i] > y_custom[i]:
        print(colored(f"Test Case {i+1}: Loan will be approved", "green"))
    else:
        print(colored(f"Test Case {i+1}: Loan will not be approved", "red"))
