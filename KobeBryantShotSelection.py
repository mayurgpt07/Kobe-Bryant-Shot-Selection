import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
import seaborn as sns
import reverse_geocoder as rgc
import warnings
from scipy.sparse import hstack

#Create Locations From Latitude and Longitude
def getPlacesFromLatandLong(train_data):
	#GeoLocation to be added into the picture. Try to convert Latitude and Longitude to names of places in Kings County Region
	for i in range(0, len(train_data['lat'])):
		coordiantes = (train_data.loc[i,'lat'], train_data.loc[i,'lon'])
		#print(rgc.search(coordiantes, mode = 1))
		train_data.loc[i, 'Location'] = rgc.search(coordiantes, mode = 1)[0].get('name')

#Select data to train based on data prior to that date itself
def TrainModel(game_date, train_data):
	train_data[train_data['DateinFormat'] < game_date]
	return train_data

def prepare_inputs(train_data):
	oe = OrdinalEncoder()
	oe.fit(train_data)
	train_data_enc = oe.transform(train_data)
	return train_data_enc


ShotsMade = pd.read_csv('./data.csv', sep = ',', header = 0)
print(type(ShotsMade.loc[0,'game_date']))
ShotsMade['DoubleDigitMinutes'] = ShotsMade['minutes_remaining'].apply(lambda x: format(x, '02')) 
ShotsMade['DoubleDigitSeconds'] = ShotsMade['seconds_remaining'].apply(lambda x: format(x, '02'))

ShotsMade['DateTimeinMinutes'] = ShotsMade['game_date'] + " " + "00:" + ShotsMade['DoubleDigitMinutes'] + ":" + ShotsMade['DoubleDigitSeconds']
print(ShotsMade['DateTimeinMinutes'])
ShotsMade['DateinFormat'] = ShotsMade['DateTimeinMinutes'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())
getPlacesFromLatandLong(ShotsMade)
print(ShotsMade.head(10))

print(ShotsMade['shot_made_flag'].unique())

ValidationShotsMade = ShotsMade[pd.isnull(ShotsMade['shot_made_flag'])]
TrainShotsMade = ShotsMade[pd.notnull(ShotsMade['shot_made_flag'])]

CategoricalVariable = ['action_type','combined_shot_type','shot_type','shot_zone_area','shot_zone_basic','shot_zone_range', 'Location', 'opponent','playoffs','period']
DependentVariable = ['shot_made_flag']
NumericalVariable = ['game_event_id','game_id','lat','loc_x','loc_y','lon','minutes_remaining','season','seconds_remaining','shot_distance']

TrainShotsMadeEncoded = prepare_inputs(TrainShotsMade[CategoricalVariable])
np.savetxt('FeatureOutput.csv', TrainShotsMadeEncoded, delimiter=',', header = 'action_type,combined_shot_type,shot_type,shot_zone_area,shot_zone_basic,shot_zone_range,Location,opponent,playoffs,period', fmt="%i", comments='')

TrainShotsMadeNumerical = TrainShotsMade[NumericalVariable]
print(TrainShotsMadeNumerical.shape)

TrainShotsDecoded = pd.read_csv('./FeatureOutput.csv', sep = ',', header = 0)
print(TrainShotsDecoded.shape)
#hstack(TrainShotsMade[NumericalVariable], TrainShotsMadeEncoded).tocsr()
TrainShotsDecoded = TrainShotsDecoded.astype(int)
chi2_features = SelectKBest(chi2, k = 4)
BestFeatures = chi2_features.fit_transform(TrainShotsDecoded, TrainShotsMade[DependentVariable])
print(BestFeatures)

for i in CategoricalVariable:
	TrainShotsMade[i] = TrainShotsDecoded[i]

#plt.plot(TrainShotsMade['combined_shot_type'], TrainShotsMade['shot_made_flag'], 'o', color='black');
#plt.hist(ShotsMade['shot_zone_range'], color = 'green')

sns.boxplot(x = 'shot_made_flag', y = 'combined_shot_type', data = TrainShotsMade)
plt.show()