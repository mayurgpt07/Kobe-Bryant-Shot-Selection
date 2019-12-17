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
	#print(oe.categories_)
	train_data_enc = oe.transform(train_data)
	return train_data_enc

def generateVisual(train_data, inputCol, outputCol):
	VisualizationDataFrame = pd.DataFrame()
	inputList = []
	outputList = []
	uniqueInputCol = train_data[inputCol].unique()
	uniqueOutputCol = train_data[outputCol].unique()
	print(uniqueInputCol, uniqueOutputCol)
	for i in uniqueInputCol:
		for j in uniqueOutputCol:
			inputList.append(str(i) + ' ' + str(j))
			interValue = np.where((train_data[inputCol] == i) & (train_data[outputCol] == j))
			outputList.append(np.count_nonzero(interValue))
	VisualizationDataFrame['X Label'] = pd.Series(inputList)
	VisualizationDataFrame['Y Label'] = pd.Series(outputList)
	index = np.arange(len(inputList))
	plt.bar(index, VisualizationDataFrame['Y Label'])
	plt.xlabel(inputCol, fontsize=10)
	plt.ylabel(outputCol, fontsize=10)
	plt.xticks(index, VisualizationDataFrame['X Label'], fontsize=5, rotation=30)
	plt.title('Number of Successful or Unsuccessful Shots based on '+inputCol)
	plt.show()

ShotsMade = pd.read_csv('./data.csv', sep = ',', header = 0)
print(type(ShotsMade.loc[0,'game_date']))
ShotsMade['DoubleDigitMinutes'] = ShotsMade['minutes_remaining'].apply(lambda x: format(x, '02')) 
ShotsMade['DoubleDigitSeconds'] = ShotsMade['seconds_remaining'].apply(lambda x: format(x, '02'))

ShotsMade['DateTimeinMinutes'] = ShotsMade['game_date'] + " " + "00:" + ShotsMade['DoubleDigitMinutes'] + ":" + ShotsMade['DoubleDigitSeconds']
print(ShotsMade['DateTimeinMinutes'])
ShotsMade['DateinFormat'] = ShotsMade['DateTimeinMinutes'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())
getPlacesFromLatandLong(ShotsMade)
#print(ShotsMade.head(10))

#print(ShotsMade['shot_made_flag'].unique())

ValidationShotsMade = ShotsMade[pd.isnull(ShotsMade['shot_made_flag'])]
TrainShotsMade = ShotsMade[pd.notnull(ShotsMade['shot_made_flag'])]

#print("Shot Type Made", TrainShotsMade['shot_type'].unique())
generateVisual(TrainShotsMade, 'shot_type', 'shot_made_flag')
generateVisual(TrainShotsMade, 'shot_zone_area', 'shot_made_flag')
generateVisual(TrainShotsMade, 'combined_shot_type', 'shot_made_flag')


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
#sns.catplot(x = 'shot_made_flag', y = 'combined_shot_type', hue = 'shot_type' ,data = TrainShotsMade)
#plt.show()