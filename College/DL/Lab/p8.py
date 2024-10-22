import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
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

# ------------------ RBF Network Model ------------------------

# Define the Radial Basis Function Network using Kernel Ridge Regression
rbf_model = KernelRidge(kernel="rbf", gamma=0.1)  # gamma is the spread of the RBF

# Fit the RBF model to the training data
rbf_model.fit(X_train, y_train)

# Predict using the RBF network on test data
y_pred = rbf_model.predict(X_test)

# Inverse scale the predicted results to compare to the original loan amount scale
y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Data: {mse}")

# -------------- Custom Input Predictions Using RBF --------------

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

# Predict the loan amount using the RBF model
y_custom_pred = rbf_model.predict(X_custom)

# Inverse scale the results to compare to the original scale
y_custom_pred = y_scaler.inverse_transform(y_custom_pred.reshape(-1, 1)).flatten()

# Print out the results for custom input predictions
print(f"\n\nPredicted loan amounts by RBF Network: \n{y_custom_pred}")
print(f"\nActual applied loan amounts: \n{y_custom}")

print("\n\nPredictions:")
for i in range(len(y_custom_pred)):
    if y_custom_pred[i] > y_custom[i]:
        print(
            colored(
                f"Test Case {i+1}: Loan will be approved (Predicted: {y_custom_pred[i]:.2f}, Applied: {y_custom[i]})",
                "green",
            )
        )
    else:
        print(
            colored(
                f"Test Case {i+1}: Loan will not be approved (Predicted: {y_custom_pred[i]:.2f}, Applied: {y_custom[i]})",
                "red",
            )
        )
