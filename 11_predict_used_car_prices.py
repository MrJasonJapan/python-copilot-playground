# import pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1) --------- Data Preparation  ---------

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
# print(df.head())

# plot a histogram of the price using 3 bins. label the x and y axis, and give the plot a title
# plt.hist(df['price'], bins=3)
# plt.xlabel('price')
# plt.ylabel('count')
# plt.title('price bins')
# plt.show()

# export the dataframe to a csv file (for the purpose of viewing the data in other tools such as Excel)
# note what we could export the file as other formats such as Excel, JSON, HTML, etc.
df.to_csv('11_used_car_prices.csv')


# 2) --------- Exporatory Data Analysis ---------

# print a stastics summary of the dataframe, including for object columns
print(df.describe(include='all'))

# Use value counts on drive-wheels to get a count of each type of drive wheel.
print(df['drive-wheels'].value_counts())

# Using Seaborn, create a box-plot of price vs drive-wheels, with drive-wheels on the x-axis and price on the y-axis. Exclude any NaN values.
# df.boxplot(column='price', by='drive-wheels', showfliers=False) # using standard matplotlib
sns.boxplot(x='drive-wheels', y='price', data=df) # for some reason this doesn't work on my machine
plt.show()

# Use a scatter plot to show the relationship between engine size and price
plt.scatter(df['engine-size'], df['price'])
plt.xlabel('engine-size')
plt.ylabel('price')
plt.title('engine-size vs price')
plt.show()

# group the data by drive-wheels and body-style, and then calculate the mean price for each group
df_test = df[['drive-wheels', 'body-style', 'price']]
# as_index=False means don't use the groupby column as the index. this is useful if you want to use the groupby column as a regular column
df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean() 
# pivot the data so that drive-wheels is the index, body-style is the column, and price is the value.
df_pivot = df_grp.pivot(index='drive-wheels', columns='body-style')
# print the pivot table
print(df_pivot)

# convert the pivot table into a heatmap, and show the color bar. Lable the x and y axis, and give the plot a title
plt.pcolor(df_pivot, cmap='RdBu') # RdBu stands for Red-Blue
plt.colorbar()
plt.xlabel('body-style')
plt.ylabel('drive-wheels')
plt.title('price vs drive-wheels and body-style')
plt.show()

# Use Analysis of Variance (ANOVA) to determine if there is a significant difference between the average price of cars between Honda and Subaru
# first, create a dataframe with only the price and make columns
df_anova = df[['make', 'price']]
# group the data by make, and then calculate the mean price for each group
grouped_anova = df_anova.groupby(['make'])
# use the f_oneway function from the scipy.stats module to calculate the ANOVA. 
# Notice how the prices between Honda and Subaru are very similar, and we can confim this because the f-test score is less than 1, and the p-value is greater than 0.05.
f_val, p_val = stats.f_oneway(grouped_anova.get_group('honda')['price'], grouped_anova.get_group('subaru')['price'])
print('ANOVA results: F=', f_val, ', P=', p_val)

# Do the same for Honda and Juaguar
# Notice how the prices between Honda and Jaguar are very different, and we can confim this because the f-test score is greater than 1 (around 400), and the p-value is relatively small.
f_val, p_val = stats.f_oneway(grouped_anova.get_group('honda')['price'], grouped_anova.get_group('jaguar')['price'])
print('ANOVA results: F=', f_val, ', P=', p_val)

# print a regression line for engine-size vs price
sns.regplot(x='engine-size', y='price', data=df)
plt.show()

# reset the plot
plt.clf()

# on the other hand print a regression line for highway-mpg vs price
sns.regplot(x='highway-mpg', y='price', data=df)
plt.ylim(0,)
plt.show()

# determine the characteristics of the data that have the highest correlation with price
#   note that we are only interested in the correlation of the numeric columns with price
#   we will use the Pearson correlation method
#   we will only show the correlation of the top 10 columns with price.
#   Include both positive and negative correlations.
# Notice how engine-size and highway-mpg have the highest correlation with price.
df_corr = df.corr()
print(df_corr['price'].sort_values(ascending=False)[1:6])
print(df_corr['price'].sort_values(ascending=False)[-5:])
# Print the p-values as well (for engine-size and highway-mpg)
# Note that the p-values are very small, which means that the correlation is statistically significant.
# For example a p-value less than 0.001 means that there is a 99.9% chance that the correlation is statistically significant.
print(stats.pearsonr(df['engine-size'], df['price']))
print(stats.pearsonr(df['highway-mpg'], df['price']))

# Cretae a heatmap of the correlation matrix
sns.heatmap(df_corr, cmap='RdBu')
plt.show()




