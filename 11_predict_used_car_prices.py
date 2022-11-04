# import pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# obtain used car price data from the internet, and save into a dataframe
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df = pd.read_csv(url, header=None)

# the first column is for symboling, from -3 to 3, which is a measure of how risky the car is. negative values are safer cars, positive values are riskier cars
# the second column is for normalized losses, which is a measure of how much the car depreciates in value over time
# the remaining columns are for features of the car, such as make, fuel type, body style, wheel base, length, width, height, curb weight, engine type,
#   number of cylinders, engine size, fuel system, bore, stroke, compression ratio, horsepower, peak rpm, city mpg, highway mpg, and price
# the last column is for price, which is the target variable, the one we want to predict. we will use the other columns to predict the price.


headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height',
           'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
df.columns = headers

# show details of df
print(df.info())

# correct type mismatches in the dataframe for the following columns
#   normalized-losses, bore, stroke, horsepower, peak-rpm, price
#   note that the columns are all strings (object), but we want them to be numeric
#   we will replace the ? values with NaN, and then convert the column to numeric
df['normalized-losses'] = pd.to_numeric(df['normalized-losses'], errors='coerce')  # corece means replace with NaN
df['bore'] = pd.to_numeric(df['bore'], errors='coerce')
df['stroke'] = pd.to_numeric(df['stroke'], errors='coerce')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['peak-rpm'] = pd.to_numeric(df['peak-rpm'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# print a stastics summary of the dataframe, including for object columns
print(df.describe(include='all'))

# remove any rows with a missing price value
df.dropna(subset=['price'], axis=0, inplace=True)

# convert the datatype of price to int
df['price'] = df['price'].astype('int')

# replace missing values of normalized-losses with the mean
df['normalized-losses'].replace(np.nan, df['normalized-losses'].mean(), inplace=True)

# convert the city-mpg from mpg to Liters per 100km
df['city-mpg'] = 235/df['city-mpg']

# rename city-mpg to city-L/100km
df.rename(columns={'city-mpg': 'city-L/100km'}, inplace=True)

# normalize the length using the simple feature scaling method
df['length'] = df['length']/df['length'].max()

# normalize the width using the min-max method
df['width'] = (df['width']-df['width'].min())/(df['width'].max()-df['width'].min())

# normalize the height using the z-score method
df['height'] = (df['height']-df['height'].mean())/df['height'].std()

# create a price-binned column with equal width bins based on low, medium, and high price
df['price-binned'] = pd.cut(df['price'], 3, labels=['low', 'medium', 'high'])

# convert the fuel column to dummy variables
df = pd.concat([df, pd.get_dummies(df['fuel-type'])], axis=1)


# print the first 5 rows of the dataframe
print(df.head())

# plot a histogram of the price using 3 bins. label the x and y axis, and give the plot a title
plt.hist(df['price'], bins=3)
plt.xlabel('price')
plt.ylabel('count')
plt.title('price bins')
plt.show()


# export the dataframe to a csv file (for the purpose of viewing the data in other tools such as Excel)
# note what we could export the file as other formats such as Excel, JSON, HTML, etc.
df.to_csv('11_used_car_prices.csv')
