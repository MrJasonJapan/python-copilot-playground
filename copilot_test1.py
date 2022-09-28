# this works, but not very useful lol

# import plotly and pandas
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# grab simple weather data sample from internet
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv') 
print(df["name"].unique())

# set index to name 
df.set_index('name', inplace=True)

# create line trace for pop for each name
trace = go.Scatter(x=df.index, y=df['pop'], name='pop')

data = [trace]

# create layout
layout = go.Layout(title='2014 US City Populations', xaxis=dict(title='City'), yaxis=dict(title='Population'))

# create figure
fig = go.Figure(data=data, layout=layout)

# plot figure
fig.show()
