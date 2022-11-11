# Use Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression to create a model that can predict the price of a car based on its characteristics.

# Simple Linear Regression is when we use one independent variable to predict the value of a dependent variable.
# The independent variable is also called the predictor variable (x1), and the dependent variable is also called the target variable (y).
# y = b0 + b1*x1
# note that y is the predicted value of the target variable, x1 is the value of the predictor variable, b0 is the intercept, and b1 is the slope.

# Multiple Linear Regression is when we multiple independent variables (or predictor variables x1, x2, ...) to predict the value of a dependent variable (or target variable y).
# y = b0 + b1x1 + b2x2 + b3x3 + ... + bnxn
# note that y is the predicted value of the target variable, x1, x2, ..., xn are the values of the predictor variables, b0 is the intercept, and b1, b2, ..., bn are the slopes.


from prep_data import prep_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = prep_data()


# ---- Simple Linear Regression ----
print('\nSimple Linear Regression:\n')

# create a linear regression object
# use highway-mpg as the independent variable, and price as the dependent variable
lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X, Y)

# print the intercept and slope of the linear regression model
print('Intercept: ', lm.intercept_)
print('Slope: ', lm.coef_)

# Obtain a prediction when the highway-mpg is 30, and print it out
Yhat = lm.predict([[30]])


# ---- Multiple Linear Regression ----
print('\nMultiple Linear Regression:\n')

# Use 4 predictor variables horsepower, curb-weight, engine-size, and highway-mpg to predict the price of a car.
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])

# print the intercept and slopes of the linear regression model
print('Intercept: ', lm.intercept_)
print('Slope: ', lm.coef_)

# predict the price of a car where the horsepower is 98, curb-weight is 2337, engine-size is 122, and highway-mpg is 25
Yhat = lm.predict([[98, 2337, 122, 25]])
print(Yhat)


# --- Model Evaluation using Visualization ----

# Create a regression plot to visualize the relationship between the predictor variable (highway-mpg) and the target variable (price)
sns.regplot(x='highway-mpg', y='price', data=df)
plt.show()

# reset the plot
plt.clf()

# Create a residual plot to visualize the distribution of the error terms
# The residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.
# If the points in a residual plot are randomly spread out around the x-axis, then a linear model is appropriate for the data.
# If the points in a residual plot are not randomly spread out around the x-axis, then a non-linear model is more appropriate for the data.
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.show()
# * because the residuals have a curvature, a non-linear model is more appropriate for the data.

# reset the plot
plt.clf()

# Create a distribution plot to visualize the distribution of the fitted values that result from the model and the distribution of the actual values.
Yhat = lm.predict(Z)
ax1 = sns.distplot(df['price'], hist=False, color='r', label='Actual Value')
sns.distplot(Yhat, hist=False, color='b', label='Fitted Values', ax=ax1)
plt.show()
# * We can see here that Yhat (based on Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]) is a pretty good fit for the actual values of price, 
# * because the two distributions overlap farily well.

