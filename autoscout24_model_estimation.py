## In this section I am deploying differnt ML-Models to predict car prices
#  based of the autoscout24 data

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, r2_score
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Preliminary settings
# Set the path to the working directory. 
path = r"C:\Users\Anwender\Desktop\Autoscout24_project"
os.chdir(path)

# Load the data
X = pd.read_csv('cleaned_data.csv')
X = X.drop('Unnamed: 0', axis=1)

## Preperation

# Extract the target 'price'
y = X.pop('price')

# Remove the categorical features the have been recoded to dummy variables previously
X = X.drop(columns = ['marke', 'model', 'fuel', 'gear', 'offerType', 'year'])

# Extract feature names for later use
feature_names = X.columns.tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


## Linear Regression

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_predict_lr = lr.predict(X_test)

# Model evaluation: Linear Regression
r2_score_lr = metrics.r2_score(y_test, y_predict_lr)
mse_lr      = metrics.mean_squared_error(y_test, y_predict_lr)
mae_lr      = metrics.mean_absolute_error(y_test, y_predict_lr)
print("Results: Linear Regression")
print('R-squared:', round(r2_score_lr, 2))
print('MSE:', round(mse_lr, 2))
print('MAE:', round(mae_lr, 2))
print("")

# Visualization: Linear Regression
results = pd.DataFrame({'Price': y_test, 'Predicted Output': y_predict_lr})
plt.figure(figsize=(10, 10))
sns.regplot(data=results, y='Predicted Output', x='Price', color='teal')
plt.title("Comparison of predicted values and the actual values (Linear Regression)", fontsize=20)
plt.ylabel('Predicted Price')
plt.savefig("linear_regression", dpi=600, bbox_inches='tight')
plt.show()


## DecisionTree Regression

# Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=1)
dt_reg.fit(X_train, y_train)
y_predict_dt = dt_reg.predict(X_test)

# Model evaluation
r2_score_dt = metrics.r2_score(y_test, y_predict_dt)
mse_dt      = metrics.mean_squared_error(y_test, y_predict_dt)
mae_dt      = metrics.mean_absolute_error(y_test, y_predict_dt)
print("Results on Test Set: Decision Tree Regressor")
print('R-squared:', round(r2_score_dt, 2))
print('MSE:', round(mse_dt, 2))
print('MAE:', round(mae_dt, 2))
print("")

# Visualization
results_dt = pd.DataFrame({'Price': y_test, 'Predicted Price': y_predict_dt})
plt.figure(figsize=(10, 10))
sns.regplot(data=results_dt, y='Predicted Price', x='Price', color='palevioletred', marker='o')
plt.title("Comparison of predicted values and the actual values (Decision Tree Regressor)", fontsize=20)
plt.savefig("decision_tree_regressor", dpi=600, bbox_inches='tight')
plt.show()



## Random Forest Regression

# This part is quoted out, since the crossvalidation takes a while.
"""
optimal_depth = {'max_depth': [30,31,32]}
rf_reg = RandomForestRegressor(random_state=42)
rfreg_grid = GridSearchCV(rf_reg, optimal_depth, cv=5)
rfreg_grid.fit(X_train, y_train)
optimal_depth = rfreg_grid.best_params_['max_depth']  
"""
optimal_depth = 31

# RandomForest Regressor
rf_reg = RandomForestRegressor(max_depth=optimal_depth, random_state=42)
rf_reg.fit(X_train, y_train)
y_predict_rfreg = rf_reg.predict(X_test)

# Feature importances
importances = rf_reg.feature_importances_
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)
print("")

# Model evaluation on test set
r2_score_rfreg = metrics.r2_score(y_test, y_predict_rfreg)
mse_rfreg = metrics.mean_squared_error(y_test, y_predict_rfreg)
mae_rfreg = metrics.mean_absolute_error(y_test, y_predict_rfreg)
print("")
print("Results on Test Set: RandomForrest Regression")
print(f"Optimal depth: max_depth={optimal_depth}")
print('R-squared:', round(r2_score_rfreg, 2))
print('MSE:', round(mse_rfreg, 2))
print('MAE:', round(mae_rfreg, 2))

# Visualization
results_rf = pd.DataFrame({'Price': y_test, 'Pedicted Price': y_predict_rfreg})
plt.figure(figsize=(10, 10))
sns.regplot(data=results_dt, y='Predicted Price', x='Price', color='coral', marker='o')
plt.title("Comparison of predicted values and the actual values (Random Forrest Regressor)", fontsize=20)
plt.savefig("random_forrest_regressor", dpi=600, bbox_inches='tight')
plt.show()



## inputs & prediction

# Define user inputs (these are for testing purposes)
inputs = {
    "mileage": 235.0,
    "marke": "BMW",
    "model": "316",
    "fuel": "Diesel",
    "gear": "Manual",
    "offerType": "Used",
    "hp": 116,
    "year": 2012
}

# Create a DataFrame from the user inputs
input_df = pd.DataFrame(inputs, index=[0])

# Perform one-hot encoding on categorical variables
input_encoded = pd.get_dummies(input_df, columns=["marke", "model", "fuel", "gear", "offerType", "year"])

# Create a DataFrame with all possible dummy variables
all_vars = pd.DataFrame(columns=feature_names)

# Concatenate input_encoded with all_vars
input_final = pd.concat([all_vars, input_encoded], axis=0, ignore_index=True, sort=False).fillna(0)

# Sort the features in the final input dataframe 
input_final_sorted = input_final.reindex(columns=feature_names)

# Make predictions using linear regression
lr_prediction = lr.predict(input_final_sorted)

# Make predictions using decision tree regression
dt_reg_prediction = dt_reg.predict(input_final_sorted)

# Make predictions using random forest regression
rf_reg_prediction = rf_reg.predict(input_final_sorted)

# Print the predictions
print("Linear Regression Prediction:", round(lr_prediction[0] * 1000, 2))
print("Decision Tree Regression Prediction:", round(dt_reg_prediction[0] * 1000, 2))
print("Random Forest Regression Prediction:", round(rf_reg_prediction[0] * 1000, 2))

# Save the estimated models for use in the Streamlit App (this will not be pushed to GitHub, because the file is too large)
from joblib import dump

dump({
    'lr_model': lr,
    'dt_reg_model': dt_reg,
    'rf_reg_model': rf_reg
}, 'trained_models.joblib')









