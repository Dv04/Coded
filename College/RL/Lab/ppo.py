import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch.distributions import MultivariateNormal

# Preprocessing steps
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

# Parameters for PPO
input_size = X_train.shape[1]
output_size = 1
learning_rate = 0.001
gamma = 0.99
eps_clip = 0.2
k_epochs = 10
n_episodes = 5


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, output_size)
        self.log_std = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        return mean, std


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_layer(x)
        return value


# Initialize networks
policy_network = PolicyNetwork(input_size, output_size)
value_network = ValueNetwork(input_size)
policy_optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
value_optimizer = optim.Adam(value_network.parameters(), lr=learning_rate)


# Compute PPO loss
def ppo_loss(old_log_probs, new_log_probs, advantages, eps_clip):
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    return -torch.min(surr1, surr2).mean()


# Train the PPO model
for episode in range(n_episodes):
    states = []
    actions = []
    rewards = []
    log_probs = []

    # Collect trajectories
    for i in range(len(X_train)):
        state = torch.FloatTensor(X_train[i]).unsqueeze(0)
        mean, std = policy_network(state)
        dist = MultivariateNormal(mean, torch.diag(std + 1e-6))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        reward = -np.abs(y_train_scaled[i] - action.item())  # Reward is negative error

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)

    # Compute value targets
    values = torch.cat([value_network(state) for state in states])
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    advantages = rewards - values.detach()

    # Update value network
    value_loss = torch.mean((values - rewards) ** 2)
    value_optimizer.zero_grad()
    value_loss.backward(retain_graph=True)
    value_optimizer.step()

    # Update policy network using PPO
    old_log_probs = torch.cat(log_probs).detach()
    for _ in range(k_epochs):
        new_log_probs = torch.cat(
            [
                MultivariateNormal(
                    policy_network(state)[0],
                    torch.diag(torch.exp(policy_network(state)[1]) + 1e-6),
                ).log_prob(action)
                for state, action in zip(states, actions)
            ]
        )
        loss = ppo_loss(old_log_probs, new_log_probs, advantages, eps_clip)

        policy_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()

# Testing
y_pred = []
for state in X_test:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    mean, _ = policy_network(state_tensor)
    predicted_loan_amount = y_scaler.inverse_transform(mean.detach().numpy())[0][0]
    y_pred.append(predicted_loan_amount)

# Generate predicted loan status based on predicted loan amount
y_pred_loan_status = [
    "Approved" if pred >= actual else "Rejected" for pred, actual in zip(y_pred, y_test)
]

# Generate classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test.apply(lambda x: "Approved" if x >= 0 else "Rejected"), y_pred_loan_status
    )
)

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

# Predicting using PPO
y_custom_pred = []
for state in X_custom:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    mean, _ = policy_network(state_tensor)
    predicted_loan_amount = y_scaler.inverse_transform(mean.detach().numpy())[0][0]
    y_custom_pred.append(predicted_loan_amount)

print(f"\n\nPredicted loan amounts: \n{y_custom_pred}")
print(f"\nActual applied loan amounts: \n{y_custom.tolist()}")

# Loan approval predictions
print("\n\nPredictions:")
for i in range(len(y_custom_pred)):
    if y_custom_pred[i] > y_custom.iloc[i]:
        print(f"Test Case {i+1}: Loan will be approved")
    else:
        print(f"Test Case {i+1}: Loan will not be approved")
