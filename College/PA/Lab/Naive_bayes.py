# Code for Naive Bayes Classifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB


# Create a label encoder object
le = LabelEncoder()

# Load dataset
dataset = pd.read_csv("Life.csv")

# Identify categorical columns
categorical_cols = dataset.select_dtypes(include=["object"]).columns

# Apply label encoder on categorical feature columns
for col in categorical_cols:
    dataset[col] = le.fit_transform(dataset[col])


# Compute the correlation matrix
corr = dataset.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the correct aspect ratio
sns.heatmap(
    corr,
    cmap=cmap,
    vmax=0.3,
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    annot=True,
)

plt.show()


# Define X and y
X = dataset.drop("Status", axis=1)
y = dataset["Status"]

# Handle NaN values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Create a Gaussian Naive Bayes object
gnb = GaussianNB()

# Train the model
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
classification_report = metrics.classification_report(y_test, y_pred)


print("Accuracy:", accuracy)
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("Classification report:\n", classification_report)
