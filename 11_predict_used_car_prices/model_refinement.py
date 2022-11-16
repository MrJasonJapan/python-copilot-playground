# ---- Model Refinement ----

# About in-sample evaluation:
# In-sample evaluation is when we use the same data that we used to train the model to evaluate the model.
# This will give us a very good estimate of how accurately the model will perform on out-of-sample data, 
# but it does not tell us the amount of error that the model makes in predicting new data that the model has not seen before.

# The solution:
# Split the data into training and testing data (out of sample data). Usually 80% of the data is used for training, and the rest for testing.
# After be complete testing the model, we should re-train the model on the entire dataset to make the model ready for production.

# About cross validation (what we will use in this file):
# Cross-validation is the process of splitting the data into training and testing data multiple times and
# computing the testing accuracy for each time. The average testing accuracy is used as the estimate of out-of-sample accuracy.
# Cross-validation is a better estimate of out-of-sample accuracy because it is less sensitive to which observations happen to be in the testing set.
# Each split of the data, also known as a fold, is used for both training and testing, so cross-validation is sometimes called rotation estimation.

# About underfitting:
# Underfitting occurs when a model is too simple to capture the underlying trend of the data.

# About overfitting:
# Overfitting occurs when a model captures the noise in the data, rather than the underlying trend.
# In general, we want the test error to decrease, but not to the point where it increases again due to overfitting.
# So in polynomial regression, for example, we select the order that minimizes the test error.

# Note about noise:
# Because noise is random, it is not possible to fit a model to noise.
# In this way, noise is somtimes referred to as an irreducible error.

# Import the necessary modules
from prep_data import prep_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

df = prep_data()

# Recreate our previous model (from the model_development.py file) using a pipeline with polynomial regression.
print('\n--- Pipeline model with horsepower, curb-weight, engine-size, and highway-mpg as features to predict price ---')
Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree=1)), ('model', LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], df['price'])
# Obtain a prediction
ypipe=pipe.predict(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# use cross_val_score to evaluate the model
# The parameter cv=4 specifies the number of folds; in this case, four.
# The function will return an array with the 4 R^2 scores.
Rcross = cross_val_score(pipe, df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], df['price'], cv=4)

# print the average and the standard deviation of our estimate
# Rcross.mean() shows the average R^2 score, and Rcross.std() shows the average deviation between the folds.
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())


# * Note, you could also use train_test_split to split the data into training and testing data on our own, without relying on cross_val_score, but it is more convenient to use cross_val_score.
# Then, you could use the testing data to evaluate the model.
# However, cross-validation is a better estimate of out-of-sample accuracy because it is less sensitive to which observations happen to be in the testing set.
print('\n--- Using train_test_split to split the data into training and testing data ---')
x_train, x_test, y_train, y_test = train_test_split(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], df['price'], test_size=0.3, random_state=0)
# Use the training data to fit the model
pipe.fit(x_train, y_train)
# Use the testing data to obtain a prediction, and print the R^2 value
# We can see that it is not far off from the R^2 value we obtained using cross-validation, but cross-validation is a better estimate of out-of-sample accuracy.
ypipe=pipe.predict(x_test)
print("R^2: ", pipe.score(x_test, y_test))


# Okay, not let's explore how we cause use multiple orders of polynomial regression (followd by linear regression) to see which one is best.
# In this case we will keep it simple, and just use horsepower to predict price.
# We will use a for loop to create 10 different models using a range from 1 to 10.
# We print the R^2 value for each model, and we see that the R^2 gradually increases until a specific order, and then decreases again.
# * In this case we can see that order 5 seems to be the best.
print('\n--- With only horsepower as the predictor variable, find the best order of polynomial regression ---')
order = [1,2,3,4,5,6,7,8,9,10]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    print("R^2 for order", n, "is", poly.score(x_test_pr, y_test))

print('\n Try the same thing but with engine-size ---')
# * In this case we can see that order 3 seems to be the best.
order = [1,2,3,4,5,6,7,8,9,10]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['engine-size']])
    x_test_pr = pr.fit_transform(x_test[['engine-size']])
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    print("R^2 for order", n, "is", poly.score(x_test_pr, y_test))    

# We could attempt to get better results if we used more variables, to predict price.
# Let's see this in action, by considering ['horsepower', 'curb-weight', 'engine-size', 'highway-mpg'] to predict price.
# * Unfortunately, we see that the R^2 value is only slightly better than the R^2 value we obtained using a single variable, which was at order 1.
print('\n--- With 4 features as the predictor variables, find the best order of polynomial regression ---')
order = [1,2,3,4,5,6,7,8,9,10]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    print("R^2 for order", n, "is", poly.score(x_test_pr, y_test))

# * Key takeaway, because we manually cut training and testing data, and used it without cross-validation, We might assume that the R^2 are not optimal.
# * When we used cross-validation, we usually get a more realistic estimate of the model's accuracy.
# * Also note that because we used a loop to experiment with different orders, we were able to go back and accordingly adjust the Input degree insisde our pipeline.


# --- Use ridge regression to see if we can improve the model ---
print('\n--- Ridge Regression model with horsepower, curb-weight, engine-size, and highway-mpg as features to predict price ---')
RidgeModel = Ridge(alpha=0.01)
RidgeModel.fit(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], df['price'])

# use cross_val_score to evaluate the model
RcrossRidge = cross_val_score(RidgeModel, df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], df['price'], cv=4)

# print the average and the standard deviation of our estimate
print("The mean of the folds are", RcrossRidge.mean(), "and the standard deviation is", RcrossRidge.std())
# output with a degree of 3, and no ridge regression: R^2 mean of -11.063572818747476
# output with a degree of 3, and ridge regression with alpha 0.1: R^2 mean of 0.6644056262129685
# output with a degree of 3, and ridge regression with alpha 0.01: R^2 mean of 0.6644051006641822

# Use Grid Search to find the best alpha value
# If we chose to normalize the data, use the parameter normalize': [True, False] 
# This would bea trying two different settings for the normalization parameter, True or False
# The benefit of normalization is that it transforms the variables so that they each have the property
# of a standard normal distribution with a mean of zero and a standard deviation of one.
# When I tried out normalization on/off (normalize': [True, False]), I got a error, so I left it out.
paramaeters1 = [{'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}]
Grid1 = GridSearchCV(RidgeModel, paramaeters1, cv=4) # the default scoring method is R^2
Grid1.fit(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], df['price'])
BestRR = Grid1.best_estimator_
scores = Grid1.cv_results_
print(scores['mean_test_score'])
# * We can see that an alpha value of 10000 gives us the best result: 0.67237476.
# * This is slightly better than the result we got using cross-validation with alpha 0.1: 0.6644056262129685
# * Notice how GridSearch is similar to cross-validation, in that we can cut the data into folds, and then use different folds for training and testing.
# * The testing, validation, and testing happens in the background.

# <next>: check out the lab for more on ridge regression
