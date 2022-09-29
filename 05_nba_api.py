# import nba_api teams, and pandas
from nba_api.stats.static import teams
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder 


# get all teams
nba_teams = teams.get_teams()

# convert into a pandas dataframe
df = pd.DataFrame(nba_teams)

# filter based on the team Lakers
lakers = df[df['full_name'] == 'Los Angeles Lakers']

# get the id from the first row of lakers
lakers_id = lakers['id'].values[0]

print(lakers_id)

# find all games for the lakers_id
gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=lakers_id)

# get data frames from gamefinder (this doesn't seem to work)
games = gamefinder.get_data_frames()[0]

games.head()



