import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import make_pipeline
from keras.models import Sequential
from keras.layers import Dense
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

# Scale the target variable as well
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(
    y_train.values.reshape(-1, 1)
).flatten()  # Scaling the target variable
y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Build the RBM model
rbm = BernoulliRBM(
    n_components=5, random_state=42
)  # Increased the number of components

# Create a pipeline with StandardScaler and RBM
pipeline = make_pipeline(StandardScaler(), rbm)

# Fit the model to the training data
pipeline.fit(X_train)

# Transform the training data using RBM
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Build a Feedforward Neural Network
model = Sequential()
model.add(Dense(64, activation="relu", input_dim=X_train_transformed.shape[1]))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Fit the model
model.fit(X_train_transformed, y_train, epochs=50, batch_size=32, verbose=0)

# Predict on the test set
y_pred = model.predict(X_test_transformed)

# Inverse scale the results to compare to original scale
y_pred = y_scaler.inverse_transform(y_pred)

# Create a custom input to predict whether a loan will be approved or not
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

# Transform the custom input using the pipeline
X_custom_transformed = pipeline.transform(X_custom)

# Predict the loan amount using the neural network model
y_custom_pred = model.predict(X_custom_transformed)

# Inverse scale the results to compare to original scale
y_custom_pred = y_scaler.inverse_transform(y_custom_pred)

print(f"\n\nPredicted loan amounts: \n{y_custom_pred.flatten()}")
print(f"\nActual applied loan amounts: \n{y_custom.values}")


print("\n\nPredictions:")
for i in range(len(y_custom_pred)):
    if (
        y_custom_pred[i][0] > y_custom.values[i]
    ):  # Compare the first element of the prediction
        print(colored(f"Test Case {i + 1}: Loan will be approved", "green"))
    else:
        print(colored(f"Test Case {i + 1}: Loan will not be approved", "red"))
