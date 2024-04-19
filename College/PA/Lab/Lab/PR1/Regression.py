from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Create a label encoder object
le = LabelEncoder()

# Load dataset
dataset = pd.read_csv("Advertising.csv")

# Identify categorical columns
categorical_cols = dataset.select_dtypes(include=["object"]).columns

# Apply label encoder on categorical feature columns
for col in categorical_cols:
    dataset[col] = le.fit_transform(dataset[col])

X = dataset["TV"].values.reshape(-1, 1)
y = dataset["Sales"].values.reshape(-1, 1)

X = pd.DataFrame(X)
y = pd.DataFrame(y)

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)  # training the algorithm

# To retrieve the intercept:
print(regressor.intercept_)
# For retrieving the slope:
print(regressor.coef_)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({"Actual": y_test.values.flatten(), "Predicted": y_pred.flatten()})
print(df)

# Calculate errors
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)


# Create a label encoder object
le = LabelEncoder()

# Load dataset
dataset = pd.read_csv("Advertising.csv")

# Identify categorical columns
categorical_cols = dataset.select_dtypes(include=["object"]).columns

# Apply label encoder on categorical feature columns
for col in categorical_cols:
    dataset[col] = le.fit_transform(dataset[col])

# Define X and y
X = dataset.drop("Sales", axis=1)
y = dataset["Sales"]

# Handle NaN values
X = X.fillna(X.mean())
y = y.fillna(y.mean())


# Load your data and split it into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a Ridge regression model
ridge = Ridge(alpha=1.0)

# Fit the model to the training data
ridge.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = ridge.predict(X_test)

# Calculate errors
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
