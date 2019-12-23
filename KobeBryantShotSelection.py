import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
import seaborn as sns
import reverse_geocoder as rgc
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import model_selection
from scipy.sparse import hstack
from sklearn.decomposition import PCA

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

def secondsRemaning(train_data):
	totalTimeinSec = 12*4*60
	train_data['secondsTotal'] = train_data['minutes_remaining']*60 + train_data['seconds_remaining']

	#train_data['Time']



ShotsMade = pd.read_csv('./data.csv', sep = ',', header = 0)
ShotsMade['DoubleDigitMinutes'] = ShotsMade['minutes_remaining'].apply(lambda x: format(x, '02')) 
ShotsMade['DoubleDigitSeconds'] = ShotsMade['seconds_remaining'].apply(lambda x: format(x, '02'))

ShotsMade['DateTimeinMinutes'] = ShotsMade['game_date'] + " " + "00:" + ShotsMade['DoubleDigitMinutes'] + ":" + ShotsMade['DoubleDigitSeconds']
ShotsMade['DateinFormat'] = ShotsMade['DateTimeinMinutes'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S').date())
getPlacesFromLatandLong(ShotsMade)
secondsRemaning(ShotsMade)
print(ShotsMade['secondsTotal'].max(), ShotsMade['secondsTotal'].min())

ValidationShotsMade = ShotsMade[pd.isnull(ShotsMade['shot_made_flag'])]
TrainShotsMade = ShotsMade[pd.notnull(ShotsMade['shot_made_flag'])]
print('Training Data shape', TrainShotsMade.shape)

# generateVisual(TrainShotsMade, 'shot_type', 'shot_made_flag')
# generateVisual(TrainShotsMade, 'shot_zone_area', 'shot_made_flag')
# generateVisual(TrainShotsMade, 'combined_shot_type', 'shot_made_flag')
# generateVisual(TrainShotsMade, 'shot_zone_basic', 'shot_made_flag')
# generateVisual(TrainShotsMade, 'period', 'shot_made_flag')

CategoricalVariable = ['action_type','combined_shot_type','shot_type','shot_zone_area','shot_zone_basic','shot_zone_range', 'Location', 'opponent','playoffs','period','season']
DependentVariable = ['shot_made_flag']
NumericalVariable = ['game_event_id','game_id','lat','loc_x','loc_y','lon','minutes_remaining','seconds_remaining','shot_distance','secondsTotal','shot_made_flag']

TrainShotsMadeEncoded = prepare_inputs(TrainShotsMade[CategoricalVariable])
print('Train Shots Encoded Size',TrainShotsMadeEncoded.shape)
np.savetxt('FeatureOutput.csv', TrainShotsMadeEncoded, delimiter=',', header = 'action_type,combined_shot_type,shot_type,shot_zone_area,shot_zone_basic,shot_zone_range,Location,opponent,playoffs,period,season', fmt="%i", comments='')

TrainShotsMadeNumerical = TrainShotsMade[NumericalVariable]

TrainShotsDecoded = pd.read_csv('./FeatureOutput.csv', sep = ',', header = 0)

TrainShotsDecoded = TrainShotsDecoded.astype(int)

print('Train Encoded Read Shape',TrainShotsDecoded.shape)

chi2_features = SelectKBest(chi2, k = 2)
BestFeatures = chi2_features.fit_transform(TrainShotsDecoded, TrainShotsMade[DependentVariable])
print(BestFeatures)
TrainShotsMadeLater = TrainShotsMade[NumericalVariable]

for i in CategoricalVariable:
	print(list(TrainShotsMadeLater.columns))
	tempList = TrainShotsDecoded[i].tolist()
	print(len(tempList))
	TrainShotsMadeLater.loc[:,i] = pd.Series(tempList, index = TrainShotsMadeLater.index)

TrainShotsMadeLater.to_csv('Training.csv', sep = ',', header = True)
print(TrainShotsMadeLater.shape)
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['shot_type'], prefix = 'shot_type', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['combined_shot_type'], prefix = 'combined_shot_type', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['shot_zone_basic'],  prefix = 'shot_zone_basic', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['shot_zone_range'],  prefix = 'shot_zone_range', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['playoffs'], prefix = 'playoffs', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['period'], prefix = 'period', prefix_sep = '@'))
print(list(TrainShotsMadeLater.columns))

#plt.plot(TrainShotsMade['combined_shot_type'], TrainShotsMade['shot_made_flag'], 'o', color='black');
#plt.hist(ShotsMade['shot_zone_range'], color = 'green')
#sns.catplot(x = 'shot_made_flag', y = 'secondsTotal',hue = 'period',data = TrainShotsMade)
#plt.show()

IndependentVariables = ['season','shot_distance','secondsTotal','period@0','period@1', 'period@2', 'period@3', 'period@4', 'period@5', 'period@6','combined_shot_type@0', 'combined_shot_type@1', 'combined_shot_type@2', 'combined_shot_type@3', 'combined_shot_type@4', 'combined_shot_type@5', 'shot_zone_basic@0','playoffs@0','playoffs@1','shot_type@0', 'shot_type@1', 'shot_zone_basic@0', 'shot_zone_basic@1', 'shot_zone_basic@2', 'shot_zone_basic@3', 'shot_zone_basic@4', 'shot_zone_basic@5', 'shot_zone_basic@6', 'shot_zone_range@0', 'shot_zone_range@1', 'shot_zone_range@2', 'shot_zone_range@3', 'shot_zone_range@4']

X, y = TrainShotsMadeLater[IndependentVariables], TrainShotsMadeLater[DependentVariable]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.33)

LogisticModel = LogisticRegression(penalty = 'elasticnet', C = 0.1, solver = 'saga', max_iter = 4000, l1_ratio = 0.5)
fittedLogisticModel = LogisticModel.fit(X_train, Y_train)
print(model_selection.cross_val_score(fittedLogisticModel, X_train, Y_train, cv = 10), np.mean(model_selection.cross_val_score(fittedLogisticModel, X_train, Y_train, cv = 10)))

SupportVectorModel = SVC(kernel = 'rbf', C = 0.1, cache_size = 10000.0, decision_function_shape = 'ovo')
FittedSVModel = SupportVectorModel.fit(X_train, Y_train)
print(model_selection.cross_val_score(FittedSVModel, X_train, Y_train, cv = 3), np.mean(model_selection.cross_val_score(FittedSVModel, X_train, Y_train, cv = 3)))