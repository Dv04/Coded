import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from collections import deque
import random
from termcolor import colored

# Preprocessing steps (from your template)
print("Reading dataset...")
data = pd.read_csv("loan_approval_dataset.csv")

loan_status = data["loan_status"]  # Save loan_status before dropping
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


# Parameters for TD3
input_size = X_train.shape[1]  # Number of features (states)
output_size = 1  # Continuous output (loan amount prediction)
learning_rate_actor = 0.001
learning_rate_critic = 0.001
gamma = 0.99  # Discount factor
tau = 0.005  # For soft target updates
batch_size = 64
memory_size = 10000
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2

# Replay memory
memory = deque(maxlen=memory_size)


# Define Actor and Critic networks
class ActorNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, a):
        a = (
            a if a.dim() == 2 else a.unsqueeze(1)
        )  # Ensure action tensor has correct dimensions
        x = torch.cat([x, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Initialize Actor, Critic, and their target networks

actor = ActorNetwork(input_size, output_size)
target_actor = ActorNetwork(input_size, output_size)
critic_1 = CriticNetwork(input_size, output_size)
target_critic_1 = CriticNetwork(input_size, output_size)
critic_2 = CriticNetwork(input_size, output_size)
target_critic_2 = CriticNetwork(input_size, output_size)

# Copy weights from the original networks to the target networks
target_actor.load_state_dict(actor.state_dict())
target_critic_1.load_state_dict(critic_1.state_dict())
target_critic_2.load_state_dict(critic_2.state_dict())


# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate_actor)
critic_1_optimizer = optim.Adam(critic_1.parameters(), lr=learning_rate_critic)
critic_2_optimizer = optim.Adam(critic_2.parameters(), lr=learning_rate_critic)
loss_fn = nn.MSELoss()


# Store experiences in the replay buffer
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


# Sample a batch of experiences
def sample_experiences(batch_size):
    return random.sample(memory, batch_size)


# Soft update target networks
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


# Train the TD3 model
print("Starting training...")
n_episodes = 5
for episode in range(n_episodes):
    print(f"Episode {episode + 1}/{n_episodes}")
    for i, state in enumerate(X_train):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = actor(state_tensor).detach().numpy().flatten()
        next_state = X_train[i]
        reward = -np.abs(y_train_scaled[i] - action[0])  # Reward is negative error

        # Store experience in the replay buffer
        store_experience(state, action, reward, next_state, False)

        # Sample a batch of experiences from the memory
        if len(memory) > batch_size:
            experiences = sample_experiences(batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)
            states = torch.FloatTensor(np.array(states))
            actions = (
                torch.FloatTensor(np.array(actions)).unsqueeze(1)
                if actions[0].ndim == 0
                else torch.FloatTensor(np.array(actions))
            )
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states))

            # Add noise to target actions
            noise = torch.clamp(
                torch.normal(0, policy_noise, size=actions.shape),
                -noise_clip,
                noise_clip,
            )
            next_actions = target_actor(next_states) + noise
            next_actions = torch.clamp(next_actions, -1, 1)

            # Compute target Q-values
            target_q1_values = target_critic_1(next_states, next_actions).detach()
            target_q2_values = target_critic_2(next_states, next_actions).detach()
            q_targets = rewards + (
                gamma * torch.min(target_q1_values, target_q2_values)
            )

            # Compute predicted Q-values and update Critic networks
            q1_values = critic_1(states, actions)
            critic_1_loss = loss_fn(q1_values, q_targets)
            critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            critic_1_optimizer.step()

            q2_values = critic_2(states, actions)
            critic_2_loss = loss_fn(q2_values, q_targets)
            critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            critic_2_optimizer.step()

            # Update Actor network
            if i % policy_freq == 0:
                actor_loss = -critic_1(states, actor(states)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Soft update target networks
                soft_update(target_actor, actor, tau)
                soft_update(target_critic_1, critic_1, tau)
                soft_update(target_critic_2, critic_2, tau)

print("Training completed.")

# Testing
print("Starting testing...")
y_pred = []
for state in X_test:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = actor(state_tensor).detach().numpy().flatten()[0]
    predicted_loan_amount = y_scaler.inverse_transform([[action]])[0][0]
    y_pred.append(predicted_loan_amount)

# Generate predicted loan status based on predicted loan amount
y_pred_loan_status = [
    "Approved" if pred >= actual else "Rejected" for pred, actual in zip(y_pred, y_test)
]

# Generate classification report
print("\nClassification Report:")
print(classification_report(loan_status_test, y_pred_loan_status))

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

# Predicting using TD3
y_custom_pred = []
for state in X_custom:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = actor(state_tensor).detach().numpy().flatten()[0]
    predicted_loan_amount = y_scaler.inverse_transform([[action]])[0][0]
    y_custom_pred.append(predicted_loan_amount)

print(f"\n\nPredicted loan amounts: \n{y_custom_pred}")
print(f"\nActual applied loan amounts: \n{y_custom.tolist()}")

# Loan approval predictions
print("\n\nPredictions:")
for i in range(len(y_custom_pred)):
    if y_custom_pred[i] > y_custom.iloc[i]:
        print(colored(f"Test Case {i+1}: Loan will be approved", "green"))
    else:
        print(colored(f"Test Case {i+1}: Loan will not be approved", "red"))
