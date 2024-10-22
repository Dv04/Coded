import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from termcolor import colored

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

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="linear"))  # Linear activation for regression

# Compile the model
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Optionally, predict and inverse scale the results to compare to original scale
y_pred = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred).flatten()

# Updated custom input for loan approval prediction
custom_input = pd.DataFrame(
    {
        "no_of_dependents": [1, 2, 4, 0],
        "education": ["Graduate", "Not Graduate", "Graduate", "Graduate"],
        "self_employed": ["No", "Yes", "No", "Yes"],
        "income_annum": [4500000, 2500000, 6000000, 350000],
        "loan_amount": [10000000, 3000000, 2500000, 8000000],
        "loan_term": [15, 10, 20, 12],
        "cibil_score": [720, 610, 780, 850],
        "residential_assets_value": [5000000, 1500000, 12000000, 4000000],
        "commercial_assets_value": [750000, 900000, 600000, 2500000],
        "luxury_assets_value": [1200000, 150000, 50000, 4000000],
        "bank_asset_value": [2200000, 60000, 300000, 250000],
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

# Predict the loan amount
y_custom_pred = model.predict(X_custom)

# Inverse scale the results to compare to original scale
y_custom_pred = y_scaler.inverse_transform(y_custom_pred).flatten()

print(f"\n\nPredicted loan amounts: \n{y_custom_pred}")
print(f"\nActual applied loan amounts: \n{y_custom}")


print("\n\nPredictions:")
for i in range(len(y_custom_pred)):
    if y_custom_pred[i] > y_custom[i]:
        print(colored(f"Test Case {i+1}: Loan will be approved", "green"))
    else:
        print(colored(f"Test Case {i+1}: Loan will not be approved", "red"))
