import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('regularization_data.csv', header = None)
# six predictors go into one outcome
X = train_data.iloc[:, 0:6]    # or can use  train_data.iloc[:, :-1] to grab first 6 
y = train_data.iloc[:, -1]

# Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# Fit the model.
lasso_reg.fit(X, y)

# Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)

# print y intercept 
print(lasso_reg.intercept_)

# print score
print(f"score: {lasso_reg.score(X, y)}")