Student Information:
Student Name: Sopheaktra Lean
Student ID: 40225014
Course: COMP 432 Machine Learning - Fall 2024
Deadline: September 30th 2024

Major Assignment 1 - Implementation Question 1

Insurance Charges Prediction Using Linear Regression

Model Description:
The model aims to predict the insurance charges based on different types of
features such as age, gender, children, bmi, region, and if the person is a smoker.
The dataset used to train this model is the insurance.csv file, which has been
imported from a website called "Kaggle". It contains several numerical and categorical
variables to support our prediction.

The steps to create the model:
1. Data loading: We upload the dataset from the insurance.csv file.
2. Data visualization: We visualize the distribution of the datasets.
3. Data processing: We encode the features like sex, smoker, and region into numerical.
4. Model creation: We create the linear regression model to predict the insurance charges by
dividing 70% of the dataset into the training process and 30% into the testing process.
5. Model performance: We evaluate teh model's performance using metrics like
mean squared error (MSE) and r-squared (R^2) score for both the training and testing.

Files related to the model:
1. insurance_predictor.py: The main Python file that contains the codes for all the functions
2. insurance.csv: The CSV file that containing all the features nessescary for the model.
3. README.txt: The file contains instructions for running the program and an overview of the model.

The program requires the Python libraries such as:
- pandas
- matplotlib
- seaborn
- numpy
- scikit-learn

How to install these libraries:
- Have Python3 ready
- Install the packages by running the following command:

        pip install pandas matplotlib seaborn numpy scikit-learn

How to run the program:
- Download the program and make sure 'insurance.csv' is in the same directory as
'insurance_predictor.py'
- Open your terminal or command prompt and navigate to the directory where the files are located
- Run the program by using this command: python insurance_predictor.py
- What to predict:
       + Program load and display the head of the dataset
       + Show information about the dataset and any null values
       + Display the visualization for both numerical and categorical features
       + Split the dataset into training and testing sets
       + Train a Linear Regression model.
       + Evaluate and display the model performance using MSE and R-squared for both training and testing sets.

The output of the program:
- Initial dataset preview and summary
- Graph visualizations for age, bmi, children, and charges distribution.
- Count plots for categorical variables: sex, region, and smoker status.
- Model evaluation metrics:
        + Mean Squared Error (MSE) for both training and test sets.
        + R-squared (R^2) for both training and test sets.


Major Assignment 1 - Implementation Question 2

Lung Cancer Prediction Using Logistic Regression

Model Description:
The model aims to predict if an individual has lung cancer based on different types of
features such as age, gender, if the person is a smoker, has yellow fingers, has anxiety has peer pressure,
has chronic disease, fatigue, etc. The dataset used to train this model is the dataset.csv file, which has been
imported from a website called "Kaggle". It contains several numerical and categorical
variables to support our prediction.

The steps to create the model:
1. Data loading: We upload the dataset from the dataset.csv file.
2. Data visualization: We visualize the distribution of the datasets.
3. Data processing: We convert categorical variables into numerical form using LabelEncoder.
4. Model creation: We create the logistic regression model to predict the lung cancer possibility by
dividing 70% of the dataset into the training process and 30% into the testing process.
5. Model performance: We evaluate teh model's performance using accuracy, confusion matrix, precision,
recall, F1 score, and classification reports.

Files related to the model:
1. cancer_predictor.py: The main Python file that contains the codes for all the functions
2. dataset.csv: The CSV file that containing all the features nessescary for the training anf testing the model.
3. README.txt: The file contains instructions for running the program and an overview of the model.

The program requires the Python libraries such as:
- pandas
- matplotlib
- seaborn
- numpy
- scikit-learn

How to install these libraries:
- Have Python3 ready
- Install the packages by running the following command:

        pip install pandas matplotlib seaborn numpy scikit-learn

How to run the program:
- Download the program and make sure 'dataset.csv' is in the same directory as
'insurance_predictor.py'
- Open your terminal or command prompt and navigate to the directory where the files are located
- Run the program by using this command: python cancer_predictor.py
- What to predict:
       + Program load and display the head of the dataset
       + Show information about the dataset and any null values
       + Display the visualization for both numerical and categorical features
       + After training the Logistic Regression model, it will display performance metrics,
       including accuracy, precision, recall, and F1 score for both the training and test sets.
       + Evaluate and display the model performance using confusion matrix heatmap.

The output of the program:
- Initial dataset preview and summary
- Graph visualizations for all the features.
- Classification reports
- Confusion Matrix