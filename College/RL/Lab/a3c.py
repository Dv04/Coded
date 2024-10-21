import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.multiprocessing as mp
import runpy
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

# Parameters for A3C
input_size = X_train.shape[1]
output_size = 1
learning_rate = 0.001
gamma = 0.99


# Define Actor-Critic Networks
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor_layer = nn.Linear(128, output_size)
        self.critic_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        actor_output = torch.tanh(self.actor_layer(x))
        critic_output = self.critic_layer(x)
        return actor_output, critic_output


# Training function for A3C
def train(rank, global_model, optimizer, X_train, y_train_scaled, gamma):
    local_model = ActorCriticNetwork(input_size, output_size)
    local_model.load_state_dict(global_model.state_dict())

    for episode in range(5):
        print(f"Worker {rank} - Episode {episode + 1}/5")
        for i in range(len(X_train)):
            state = torch.FloatTensor(X_train[i]).unsqueeze(0)
            action, value = local_model(state)
            reward = -np.abs(
                y_train_scaled[i] - action.item()
            )  # Reward is negative error

            next_state = torch.FloatTensor(X_train[i]).unsqueeze(0)
            _, next_value = local_model(next_state)

            target = reward + gamma * next_value.item()
            advantage = target - value.item()

            # Compute loss
            actor_loss = -torch.log(action) * advantage
            critic_loss = advantage**2
            loss = actor_loss + critic_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            for global_param, local_param in zip(
                global_model.parameters(), local_model.parameters()
            ):
                global_param._grad = local_param.grad
            optimizer.step()

            # Update local model
            local_model.load_state_dict(global_model.state_dict())


# Initialize global model and optimizer
print("Initializing A3C model...")
global_model = ActorCriticNetwork(input_size, output_size)
global_model.share_memory()
optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)
print("Model initialized.")

if __name__ == "__main__":
    # Start training with multiprocessing
    print("Starting training...")
    processes = []
    num_workers = 4
    for rank in range(num_workers):
        p = mp.Process(
            target=train,
            args=(rank, global_model, optimizer, X_train, y_train_scaled, gamma),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("Training completed.")

    # Testing
    print("Starting testing...")
    y_pred = []
    for state in X_test:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = global_model(state_tensor)
        predicted_loan_amount = y_scaler.inverse_transform([[action.item()]])[0][0]
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
    custom_input["self_employed"] = custom_input["self_employed"].map(
        {" No": 0, " Yes": 1}
    )
    X_custom = custom_input.drop(columns=["loan_amount"])
    y_custom = custom_input["loan_amount"]
    X_custom = scaler.transform(X_custom)
    print("Custom input preprocessing completed.")

    # Predicting using A3C
    print("Predicting on custom input...")
    y_custom_pred = []
    for state in X_custom:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = global_model(state_tensor)
        predicted_loan_amount = y_scaler.inverse_transform([[action.item()]])[0][0]
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
