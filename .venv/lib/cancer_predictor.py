"""
Student Name: Sopheaktra Lean
Student ID: 40225014
COMP 432 Machine Learning - Fall 2024
Deadline: September 30th 2024
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Load original dataset
data = pd.read_csv('dataset.csv')
print(data.head())
print('\n')

# data type of each columns
data.info()

# check if there is any missing value
data.isnull().sum()

# Summarize the statistics for 'age'
print(data['AGE'].describe())

# Counts for categorical columns 'GENDER', 'ALLERGY', and 'LUNG_CANCER'
print(data['GENDER'].value_counts())
print(data['ALLERGY'].value_counts())
print(data['LUNG_CANCER'].value_counts())
print(data['ANXIETY'].value_counts())

# Visualize the data

# First group of features
plt.figure(figsize=(12, 8))

# Features for the first figure
features_group1 = [
    ('AGE', 'Age Distribution', 'Frequency'),
    ('YELLOW_FINGERS', 'Yellow Fingers Distribution', 'Count'),
    ('ANXIETY', 'Anxiety Distribution', 'Count'),
    ('PEER_PRESSURE', 'Peer Pressure Distribution', 'Count'),
    ('CHRONIC_DISEASE', 'Chronic Disease Distribution', 'Count')
]

# Loop through the first group of features and create subplots
for i, (feature, title, ylabel) in enumerate(features_group1):
    plt.subplot(3, 2, i + 1)  # 3 rows and 2 columns grid
    if feature == 'AGE':
        plt.hist(data[feature], bins=10, edgecolor='black')
    else:
        sns.countplot(x=feature, data=data)
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel(ylabel)
    plt.grid(True)

plt.tight_layout()
plt.show()

# Second group of features
plt.figure(figsize=(12, 8))

# Features for the second figure
features_group2 = [
    ('FATIGUE', 'Fatigue Distribution', 'Count'),
    ('ALLERGY', 'Allergy Distribution', 'Count'),
    ('WHEEZING', 'Wheezing Distribution', 'Count'),
    ('ALCOHOL_CONSUMING', 'Alcohol Consumption Distribution', 'Count'),
    ('COUGHING', 'Coughing Distribution', 'Count')
]

# Loop through the second group of features and create subplots
for i, (feature, title, ylabel) in enumerate(features_group2):
    plt.subplot(3, 2, i + 1)  # 3 rows and 2 columns grid
    sns.countplot(x=feature, data=data)
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel(ylabel)
    plt.grid(True)

plt.tight_layout()
plt.show()

# Third group of features
plt.figure(figsize=(12, 8))

# Features for the third figure
features_group3 = [
    ('SHORTNESS_OF_BREATH', 'Shortness Of Breath Distribution', 'Count'),
    ('SWALLOWING_DIFFICULTY', 'Swallowing Difficulty Distribution', 'Count'),
    ('CHEST_PAIN', 'Chest Pain Distribution', 'Count'),
    ('LUNG_CANCER', 'Lung Cancer Distribution', 'Count')
]

# Loop through the third group of features and create subplots
for i, (feature, title, ylabel) in enumerate(features_group3):
    plt.subplot(2, 2, i + 1)  # 2 rows and 2 columns grid
    sns.countplot(x=feature, data=data)
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel(ylabel)
    plt.grid(True)

plt.tight_layout()
plt.show()

le = LabelEncoder()
data['GENDER'] = le.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])

data.describe()

# define input dataset
X = data.iloc[:, :-1]
print("dataset of X",X)

# define desired output
y = data.iloc[:,-1]
print("dataset of y",y)

# Split the dataset into training and test sets with a 7:3 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Print the shapes of the resulting datasets
print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)

# Build and train the Logistics Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# predict on the testing data
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix for Test Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


print(classification_report(y_test, y_test_pred))

# Evaluate the model
print("Training Set Performance:")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.2f}")
print("Classification Report:")
print(classification_report(y_train, y_train_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

print("\nTest Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
