# Use Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression to create a model that can predict the price of a car based on its characteristics.

# Simple Linear Regression is when we use one independent variable to predict the value of a dependent variable.
# The independent variable is also called the predictor variable (x1), and the dependent variable is also called the target variable (y).
# y = b0 + b1*x1
# note that y is the predicted value of the target variable, x1 is the value of the predictor variable, b0 is the intercept, and b1 is the slope.

# Multiple Linear Regression is when we multiple independent variables (or predictor variables x1, x2, ...) to predict the value of a dependent variable (or target variable y).
# y = b0 + b1x1 + b2x2 + b3x3 + ... + bnxn
# note that y is the predicted value of the target variable, x1, x2, ..., xn are the values of the predictor variables, b0 is the intercept, and b1, b2, ..., bn are the slopes.

# Polynomial regression is useful when the relationship between the independent variable and the dependent variable is not linear.
# Curvilinear relationships can be modeled using polynomial regression.
# Quadritic (2nd Order): y = b0 + b1x1 + b2x1^2
# 2nd Order with more than one independent variable: y = b0 + b1x1 + b2x2 + b3x1^2 + b4x1x2 + b5x2^2
# Cubic (3rd Order): y = b0 + b1x1 + b2x1^2 + b3x1^3
# Higher Order: y = b0 + b1x1 + b2x1^2 + b3x1^3 + ... + bnx1^n

from prep_data import prep_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



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


# ---- Polynomial Regression ----
print('\nPolynomial Regression:\n')

# Quick example: calculate polynomial of 3rd order using highway-mpg as the independent variable, and price as the dependent variable
f = np.polyfit(df['highway-mpg'], df['price'], 3)
p = np.poly1d(f)
print(p)

# Create 2nd Order polynomial with more than one independent variable using sklearn.preprocessing.PolynomialFeatures
pr = PolynomialFeatures(degree=2, include_bias=False)
# transform based on the horsepower and curb-weight features into a "polynomial feature" object.
x_polly = pr.fit_transform(df[['horsepower', 'curb-weight']])
print(x_polly)

# Normalizing (Pre-Processing) example:
# Normalize the df based on hoursepower highway-mpg using StandardScaler.
# StandardScaler transforms the data such that its distribution will have a mean value 0 and standard deviation of 1.
# This is useful for algorithms that assume the data is normally distributed.
SCALE=StandardScaler()
SCALE.fit(df[['highway-mpg', 'horsepower']])
df_scale=SCALE.transform(df[['highway-mpg', 'horsepower']])
print(df_scale)

# Simplify the process by using a pipeline.
# Pipelines sequentially apply a list of transforms and a final estimator.
# One pipeline example: Normalize the data, then perform a polynomial transform, then fit a linear regression model.
Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree=2)), ('model', LinearRegression())]
pipe=Pipeline(Input)
# Train the pipeline with horsepower, curb-weight, engine-size, and highway-mpg as the independent variables, and price as the dependent variable.
pipe.fit(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], df['price'])
# Obtain a prediction
ypipe=pipe.predict(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(ypipe)
# Visualize the model. We can see that this model is better than the linear model.
plt.clf()
ax1 = sns.distplot(df['price'], hist=False, color='r', label='Actual Value')
sns.distplot(ypipe, hist=False, color='b', label='Fitted Values', ax=ax1)
plt.show()


# ---- Model Evaluation and Refinement ----

# Two important measures that are often used in Statistics to determine the accuracy of a model are: R^2 and Mean Squared Error (MSE).

print('\nSimple Linear Regression Model Evaluation:\n')

# first let's see how a simple linear regression model performs on our data.
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X, Y)

# Measure the MSE of the model
# MSE measures the average of the squares of errors, that is, the difference between actual value (y) and the estimated value (ŷ).
# MSE is a measure of the quality of an estimator—it is always non-negative, and the closer to zero the better.
print(mean_squared_error(df['price'], lm.predict(X)))

# Calculate the R^2 of the model
# R^2 measures the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.
# R^2 is always between 0 and 100%:
# 0% indicates that the model explains none of the variability of the response data around its mean.
# 100% indicates that the model explains all the variability of the response data around its mean.
print(lm.score(X, Y))


print('\nMultiple Linear Regression Model Evaluation:\n')

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
# we can see that the MSE is smaller, and the R^2 is larger, than the simple linear regression model, which means that the multiple linear regression model is a better fit for this data.
print(mean_squared_error(df['price'], lm.predict(Z)))
print(lm.score(Z, df['price']))


print('\nPolynomial Regression Model Evaluation:\n')

# We can see that polynomial regression also performs better than the simple linear regression model, and the multiple linear regression model.
print(mean_squared_error(df['price'], ypipe))
print(r2_score(df['price'], ypipe))











