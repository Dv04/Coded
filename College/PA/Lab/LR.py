from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
dataset = pd.read_csv("Life.csv")

# Identify categorical columns
categorical_cols = dataset.select_dtypes(include=["object"]).columns

# Apply label encoder on categorical feature columns
for col in categorical_cols:
    dataset[col] = le.fit_transform(dataset[col])

# # Compute the correlation matrix
# corr = dataset.corr()

# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(
#     corr,
#     mask=mask,
#     cmap=cmap,
#     vmax=0.3,
#     center=0,
#     square=True,
#     linewidths=0.5,
#     cbar_kws={"shrink": 0.5},
#     annot=True,
# )

# plt.show()

X = dataset["Schooling"].values.reshape(-1, 1)
y = dataset["Life expectancy "].values.reshape(-1, 1)

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

# Plotting the actual vs predicted values
plt.scatter(X_test, y_test, color="gray")
plt.plot(X_test, y_pred, color="red", linewidth=2)
plt.text(
    0.5,
    0.5,
    f"MSE: {mse}\nMAE: {mae}\nRMSE: {rmse}",
    horizontalalignment="center",
    verticalalignment="center",
    transform=plt.gca().transAxes,
)
plt.show()
