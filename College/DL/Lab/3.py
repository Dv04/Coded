import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
from termcolor import colored
import numpy as np

# Load the dataset
data = pd.read_csv("loan_approval_dataset.csv")

# Drop the loan_status column
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

# Scale the target variable as well (not needed for SOM, but keeping it to check performance)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Initialize the Self-Organizing Map (SOM)
som_grid_size = (10, 10)  # SOM grid size (can be adjusted)
som = MiniSom(
    x=som_grid_size[0],
    y=som_grid_size[1],
    input_len=X_train.shape[1],
    sigma=1.0,
    learning_rate=0.5,
)

# Train the SOM
som.random_weights_init(X_train)
som.train_random(X_train, num_iteration=100)  # Training with 100 iterations

# Find the winning node for each data point in the training set
win_map = som.win_map(X_train)

# Predict clusters for the test set
test_clusters = []
for x in X_test:
    winning_node = som.winner(x)  # Get the winning node on the SOM
    test_clusters.append(winning_node)

# Create a custom input to map to SOM and check loan clustering
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

# Label encode 'education' and 'self_employed' columns
custom_input["education"] = custom_input["education"].map(
    {" Not Graduate": 0, " Graduate": 1}
)
custom_input["self_employed"] = custom_input["self_employed"].map({" No": 0, " Yes": 1})

# Separate features and target variable
X_custom = custom_input.drop(columns=["loan_amount"])
y_custom = custom_input["loan_amount"]

# Standardize the features
X_custom = scaler.transform(X_custom)

# Map custom input to SOM grid and get winning nodes
custom_clusters = []
for x in X_custom:
    winning_node = som.winner(x)  # Find the winning node
    custom_clusters.append(winning_node)

# Display SOM predictions for custom input
print("\n\nSOM Predictions:")
for i, node in enumerate(custom_clusters):
    print(f"Custom Input {i+1}: Mapped to SOM Node {node}")

# Display colored loan approval predictions based on cluster proximity
print("\n\nLoan Approval Predictions:")
for i in range(len(custom_clusters)):
    if custom_clusters[i] in test_clusters:  # If mapped to a previously learned cluster
        print(
            colored(
                f"Custom Input {i+1}: Loan likely to be approved (Cluster {custom_clusters[i]})",
                "green",
            )
        )
    else:
        print(
            colored(
                f"Custom Input {i+1}: Loan likely not approved (Cluster {custom_clusters[i]})",
                "red",
            )
        )
