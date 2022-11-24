#def prep_data():

# import pandas and numpy
import pandas as pd
import numpy as np

# 1) --------- Data Preparation  ---------

# From the internett, obtain data about immigration to Canada in csv format.
# Save the data into a dataframe.

df = pd.read_csv(url, header=None)

# headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height',
#         'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
# df.columns = headers

# show details of df
print(df.info())

# print the first 5 rows of the dataframe
print(df.head())

# export the dataframe to a csv file (for the purpose of viewing the data in other tools such as Excel)
# note what we could export the file as other formats such as Excel, JSON, HTML, etc.
df.to_csv('cananda_immigration.csv')

#return df