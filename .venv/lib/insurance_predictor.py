"""
Student Name: Sopheaktra Lean
Student ID: 40225014
COMP 432 Machine Learning - Fall 2024
Deadline: September 30th 2024
"""

# Import modules and library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load data from csv file
data = pd.read_csv('insurance.csv')

# Display the first few rows of dataset
print(data.head())
print('\n')

# Describe the datatype of dataset
data.info()

# Check if any data is missing
data.isnull().sum()

# Describe the statistics for age, bmi, children and charges
data.describe()

# Visualize the numerical data: age, children, bmi, and charges
# Set style for seaborn plots
sns.set(style="whitegrid")

# Create a list of features to plot
features = {
    'age': 'Age Distribution',
    'children': 'Children Distribution',
    'bmi': 'BMI Distribution',
    'charges': 'Charges Distribution'
}

# Plot numerical data
plt.figure(figsize=(10, 8))
for feature, title in features.items():
    plt.subplot(2, 2, list(features.keys()).index(feature) + 1)
    sns.histplot(data[feature], kde=False)
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Visualize categorical data: sex, region, and smoker
categorical_features = {
    'sex': 'Sex Count',
    'region': 'Region Count',
    'smoker': 'Smoker Count'
}

plt.figure(figsize=(10, 8))
for feature, title in categorical_features.items():
    plt.subplot(3, 1, list(categorical_features.keys()).index(feature) + 1)
    sns.countplot(x=data[feature])
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Convert categorical columns to numerical column
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])
data['region'] = le.fit_transform(data['region'])

# Assign input data and output data
X = data.drop(columns='charges')
y = data['charges']

# Split the dataset into training and test sets with 7:3 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('\nDataset Shape Information:')
print(f"Original dataset (X): {X.shape} -> This represents {X.shape[0]} rows and {X.shape[1]} features.")
print(f"Original target dataset (y): {y.shape} -> This represents {y.shape[0]} target values (lung cancer cases).")

print(f"\nTraining set (X_train): {X_train.shape} -> This is the feature set used for training with {X_train.shape[0]} samples and {X_train.shape[1]} features.")
print(f"Test set (X_test): {X_test.shape} -> This is the feature set used for testing with {X_test.shape[0]} samples and {X_test.shape[1]} features.")

print(f"\nTraining target set (y_train): {y_train.shape} -> This represents {y_train.shape[0]} target values used for training.")
print(f"Test target set (y_test): {y_test.shape} -> This represents {y_test.shape[0]} target values used for testing.\n")


# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# Evaluate the model on the training data
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Evaluate the model on the test data
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training set Mean Squared Error (MSE): {train_mse}")
print(f"Training set R-squared (R²): {train_r2}")
print(f"Test set Mean Squared Error (MSE): {test_mse}")
print(f"Test set R-squared (R²): {test_r2}")
