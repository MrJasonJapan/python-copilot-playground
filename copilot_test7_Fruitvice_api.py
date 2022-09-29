# Fruitvice API test

# import modules
import requests
import json
import pandas as pd

# get all fruit as json format into a data variable
data = requests.get("https://www.fruityvice.com/api/fruit/all").json()

# normalize the data, while simultaneously converting it into a pandas dataframe
df = pd.json_normalize(data)

# print the data
print(df)

# filter the datafame to rows with name "Cheery" only
cherries = df[df['name'] == 'Cherry']

# print the family and genus for all of the cheerries, and also show the index
print("First Cheery details:" +
      cherries['family'].values[0], cherries['genus'].values[0])

# Loop through all cherries, and print the row number, family, and genus
for index, row in cherries.iterrows():
    print(index, row['family'], row['genus'])
