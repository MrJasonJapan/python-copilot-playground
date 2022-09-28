# import plotly and pandas and numpy
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# grab simple weather data sample from internet
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv') 

# set index to name 
df.set_index('name', inplace=True)

# filter the list where pop is over 1000000
df = df[df['pop'] > 1000000]

print(df)

# create barchart trace barchart for each city
trace = go.Bar(x=df.index, y=df['pop'])

data = [trace]

# create layout
layout = go.Layout(title='2014 US City Populations', xaxis=dict(title='City'), yaxis=dict(title='Population'))

# create figure
fig = go.Figure(data=data, layout=layout)

# plot figure
fig.show()
