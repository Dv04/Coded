import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Discretize loan_amount (actions) into bins (for Q-Learning)
n_actions = 10  # Define the number of discrete actions (loan amount bins)
loan_amount_bins = np.linspace(min(y_train), max(y_train), n_actions + 1)
y_train_discretized = (
    np.digitize(y_train, loan_amount_bins) - 1
)  # Discretize the loan amount

# Initialize Q-table (states = feature combinations, actions = discretized loan amounts)
n_states = X_train.shape[1]
Q_table = np.zeros((n_states, n_actions))

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.3  # Exploration-exploitation tradeoff
n_episodes = 100  # Number of episodes

# Q-learning algorithm
for episode in range(n_episodes):
    for i, state in enumerate(X_train):  # Loop through each data point as a state
        if np.random.uniform(0, 1) < epsilon:  # Exploration
            action = np.random.randint(0, n_actions)
        else:  # Exploitation
            state_index = np.digitize(state, bins=np.linspace(-3, 3, n_states)) - 1
            action = np.argmax(Q_table[state_index])

        # Find the next state and reward (predicted loan amount vs actual)
        reward = -np.abs(
            loan_amount_bins[action] - y_train.iloc[i]
        )  # Reward is negative error

        # Update Q-value using the Q-learning update rule
        next_state = X_train[i]
        Q_table[:, action] = Q_table[:, action] + alpha * (
            reward + gamma * np.max(Q_table[:, action]) - Q_table[:, action]
        )

# Testing
y_pred_discretized = []
for state in X_test:
    action = np.argmax(Q_table[:, np.random.randint(0, n_states)])
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

# Predicting using Q-learning
y_custom_pred_discretized = []
for state in X_custom:
    action = np.argmax(Q_table[:, np.random.randint(0, n_states)])
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
