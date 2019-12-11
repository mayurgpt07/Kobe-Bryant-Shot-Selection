import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import reverse_geocoder as rgc
import warnings

#Create Locations From Latitude and Longitude
def getPlacesFromLatandLong(train_data):
	#GeoLocation to be added into the picture. Try to convert Latitude and Longitude to names of places in Kings County Region
	for i in range(0, len(train_data['lat'])):
		coordiantes = (train_data.loc[i,'lat'], train_data.loc[i,'long'])
		train_data.loc[i, 'Location'] = rgc.search(coordiantes, mode = 1)[0].get('name')

#Select data to train based on data prior to that date itself
def TrainModel(game_date, train_data):
	train_data[train_data['DateinFormat'] < game_date]
	return train_data


ShotsMade = pd.read_csv('./data.csv', sep = ',', header = 0)
print(type(ShotsMade.loc[0,'game_date']))
ShotsMade['DoubleDigitMinutes'] = ShotsMade['minutes_remaining'].apply(lambda x: format(x, '02')) 
ShotsMade['DoubleDigitSeconds'] = ShotsMade['seconds_remaining'].apply(lambda x: format(x, '02'))

ShotsMade['DateTimeinMinutes'] = ShotsMade['game_date'] + " " + "00:" + ShotsMade['DoubleDigitMinutes'] + ":" + ShotsMade['DoubleDigitSeconds']
print(ShotsMade['DateTimeinMinutes'])
ShotsMade['DateinFormat'] = ShotsMade['DateTimeinMinutes'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())

print(ShotsMade['shot_made_flag'].unique())

ValidationShotsMade = ShotsMade[pd.isnull(ShotsMade['shot_made_flag'])]

plt.hist(ShotsMade['shot_zone_range'], color = 'green')
plt.show()
