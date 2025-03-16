Logistic Regression Model

Overview

This project implements a Logistic Regression Model using Python for binary classification. The model follows a structured pipeline, including data preprocessing, normalization, model training, and evaluation.

Libraries Used

The following Python libraries were used in this project:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

Steps to Create the Logistic Regression Model

1. Load the Dataset

Read the dataset using Pandas.

Display initial data insights.

2. Data Cleaning and Processing

Handle missing values (if any).

Remove duplicate records.

Convert categorical variables if necessary.

3. Convert the Target Variable to 0s and 1s

Ensure that the target variable is binary (0 and 1) for logistic regression.

Use encoding if needed.

4. Normalizing the Data (Important Step)
Standardize or normalize numerical features for better model performance.
Helps in faster convergence and avoids large-scale differences between features.

5. Preparing the Model by Splitting the Data
Divide the dataset into training and testing sets.
Maintain a reasonable split (e.g., 80% train, 20% test).

6. Splitting the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

7. Fitting and Training the Model
Use Logistic Regression from sklearn.linear_model.
model = LogisticRegression()
model.fit(X_train, y_train)

8. Evaluating the Model
Check the model accuracy using accuracy_score.
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

9. Classification Report
Get detailed performance metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

10. Drawing the Conclusion
