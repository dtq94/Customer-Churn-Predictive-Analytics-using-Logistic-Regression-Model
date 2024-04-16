# Customer Churn Predictive Analytics using Logistic Regression Model

## Overview
XYZ is a service-providing company that provides customers with a one-year
subscription plan for their product. The company wants to know if the customers will
renew the subscription for the coming year or not. 

## Objective
Build a logistics regression learning model on the given dataset to determine whether
the customer will churn or not.

## Data Dictionary 
1. Year
2. Customer_id - unique id
3. Phone_no - customer phone no
4. Gender -Male/Female
5. Age
6. No of days subscribed - the number of days since the subscription
7. Multi-screen - does the customer have a single/ multiple screen subscription
8. Mail subscription - customer receive mails or not
9. Weekly mins watched - number of minutes watched weekly
10.Minimum daily mins - minimum minutes watched
11.Maximum daily mins - maximum minutes watched
12.Weekly nights max mins - number of minutes watched at night time
13.Videos watched - total number of videos watched
14.Maximum_days_inactive - days since inactive
15.Customer support calls - number of customer support calls
16.Churn

## Approach
1. Importing the required libraries and reading the dataset.
2. Inspecting and cleaning up the data
3. Perform data encoding on categorical variables
4. Exploratory Data Analysis (EDA)
- Data Visualization
5. Feature Engineering
- Dropping of unwanted columns
6. Model Building
- Using the statsmodel library
7. Model Building
- Performing train test split
- Logistic Regression Model
8. Model Validation (predictions)
- Accuracy score
- Confusion matrix
- ROC and AUC
- Recall score
- Precision score
- F1-score
9. Handling the unbalanced data
- With balanced weights
- Random weights
- Adjusting imbalanced data
- Using SMOTE
10.Feature Selection
- Barrier threshold selection
- RFE method
11.Save the model in the form of a pickle file.

## Dependencies
- Python
- Libraries: numpy, pandas, matplotlib, seaborn, sklearn, pickle, imblearn,
statsmodel 
