import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from datetime import datetime
import seaborn as sns
import reverse_geocoder as rgc
import math
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import model_selection
from scipy.sparse import hstack
from sklearn.metrics import log_loss, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.simplefilter('ignore')

#Create Locations From Latitude and Longitude
def getPlacesFromLatandLong(train_data):
	for i in range(0, len(train_data['lat'])):
		coordiantes = (train_data.loc[i,'lat'], train_data.loc[i,'lon'])
		train_data.loc[i, 'Location'] = rgc.search(coordiantes, mode = 1)[0].get('name')
	train_data['HomeOrAway'] = train_data['Location'].apply(lambda x: 1 if x.strip().lower() == 'los angeles' else 0)

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
	for i in range(0, len(train_data['secondsTotal'])):
		if (train_data.loc[i, 'period'] >=4 and train_data.loc[i, 'secondsTotal'] <= 300):
			train_data.loc[i, 'ClutchOrNot'] = 1
		else:
			train_data.loc[i, 'ClutchOrNot'] = 0

def logLoss(y_original, y_predicted):
	return log_loss(y_original, y_predicted)

def numberOfActionsInShot(train_data):
	train_data['NumberOfActions'] = train_data['action_type'].apply(lambda x: len(x.strip().split(" "))-1)
	#print(train_data['NumberOfActions'])

def shotAngle(train_data):
	for i in range(0, len(train_data['loc_x'])):
		train_data.loc[i,'ShotAngle'] = math.atan2(-(train_data.loc[i,'loc_y']),train_data.loc[i,'loc_x'])/math.pi*180

def conferenceTeams(train_data):
	conferenceDictionary = {
	'ATL':0,
	'BKN':0,
	'BOS':0,
	'CHA':0,
	'CHI':0,
	'CLE':0,
	'DAL':1,
	'DEN':1,
	'DET':0,
	'GSW':1,
	'HOU':1,
	'IND':0,
	'LAC':1,
	'MEM':1,
	'MIA':0,
	'MIL':0,
	'MIN':1,
	'NJN':0,
	'NOH':1,
	'NOP':1,
	'NYK':0,
	'OKC':1,
	'ORL':0,
	'PHI':0,
	'PHX':1,
	'POR':1,
	'SAC':1,
	'SAS':1,
	'SEA':1,
	'TOR':0,
	'UTA':1,
	'VAN':1,
	'WAS':0
	}
	train_data['Conference'] = train_data['opponent'].apply(lambda x: conferenceDictionary.get(x))

ShotsMade = pd.read_csv('./data.csv', sep = ',', header = 0)
ShotsMade['DoubleDigitMinutes'] = ShotsMade['minutes_remaining'].apply(lambda x: format(x, '02')) 
ShotsMade['DoubleDigitSeconds'] = ShotsMade['seconds_remaining'].apply(lambda x: format(x, '02'))

ShotsMade['DateTimeinMinutes'] = ShotsMade['game_date'] + " " + "00:" + ShotsMade['DoubleDigitMinutes'] + ":" + ShotsMade['DoubleDigitSeconds']
ShotsMade['DateinFormat'] = ShotsMade['DateTimeinMinutes'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S').date())
getPlacesFromLatandLong(ShotsMade)
secondsRemaning(ShotsMade)
numberOfActionsInShot(ShotsMade)
shotAngle(ShotsMade)
conferenceTeams(ShotsMade)
print(ShotsMade['secondsTotal'].max(), ShotsMade['secondsTotal'].min())

ValidationShotsMade = ShotsMade[pd.isnull(ShotsMade['shot_made_flag'])]
TrainShotsMade = ShotsMade#ShotsMade[pd.notnull(ShotsMade['shot_made_flag'])]
print('Training Data shape', TrainShotsMade.shape)

# generateVisual(TrainShotsMade, 'shot_type', 'shot_made_flag')
# generateVisual(TrainShotsMade, 'shot_zone_area', 'shot_made_flag')
# generateVisual(TrainShotsMade, 'combined_shot_type', 'shot_made_flag')
# generateVisual(TrainShotsMade, 'shot_zone_basic', 'shot_made_flag')
# generateVisual(TrainShotsMade, 'period', 'shot_made_flag')
CategoricalVariable = ['action_type','combined_shot_type','shot_type','shot_zone_area','shot_zone_basic','shot_zone_range', 'Location', 'opponent','playoffs','period','season','ClutchOrNot']
DependentVariable = ['shot_made_flag']
NumericalVariable = ['game_event_id','game_id','lat','loc_x','loc_y','lon','minutes_remaining','seconds_remaining','shot_distance','secondsTotal','shot_made_flag','shot_id', 'NumberOfActions', 'ShotAngle', 'HomeOrAway','Conference']

TrainShotsMadeEncoded = prepare_inputs(TrainShotsMade[CategoricalVariable])
print('Train Shots Encoded Size',TrainShotsMadeEncoded.shape)
np.savetxt('FeatureOutput.csv', TrainShotsMadeEncoded, delimiter=',', header = 'action_type,combined_shot_type,shot_type,shot_zone_area,shot_zone_basic,shot_zone_range,Location,opponent,playoffs,period,season,ClutchOrNot', fmt="%i", comments='')

TrainShotsMadeNumerical = TrainShotsMade[NumericalVariable]

TrainShotsDecoded = pd.read_csv('./FeatureOutput.csv', sep = ',', header = 0)

TrainShotsDecoded = TrainShotsDecoded.astype(int)

print('Train Encoded Read Shape',TrainShotsDecoded.shape)

#chi2_features = SelectKBest(chi2, k = 2)
#BestFeatures = chi2_features.fit_transform(TrainShotsDecoded, TrainShotsMade[DependentVariable])
#print(BestFeatures)
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
#TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['shot_zone_range'],  prefix = 'shot_zone_range', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['playoffs'], prefix = 'playoffs', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['period'], prefix = 'period', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['ClutchOrNot'], prefix = 'ClutchOrNot', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['NumberOfActions'], prefix = 'NumberOfActions', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['HomeOrAway'], prefix = 'HomeOrAway', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['Conference'], prefix = 'Conference', prefix_sep = '@'))

print(list(TrainShotsMadeLater.columns))
TrainShotsMadeLater['AgeOfKobeBrayant'] = TrainShotsMadeLater['season'].apply(lambda x: x + 17)

#plt.plot(TrainShotsMade['combined_shot_type'], TrainShotsMade['shot_made_flag'], 'o', color='black');
#plt.hist(ShotsMade['shot_zone_range'], color = 'green')
#sns.catplot(x = 'shot_made_flag', y = 'secondsTotal',hue = 'period',data = TrainShotsMade)
#plt.show()

IndependentVariables = ['AgeOfKobeBrayant','ShotAngle','shot_distance','secondsTotal','loc_x','loc_y', 'Conference@0', 'Conference@1','HomeOrAway@0', 'HomeOrAway@1','ClutchOrNot@0', 'ClutchOrNot@1', 'NumberOfActions@1', 'NumberOfActions@2', 'NumberOfActions@3', 'NumberOfActions@4','combined_shot_type@0', 'combined_shot_type@1', 'combined_shot_type@2', 'combined_shot_type@3', 'combined_shot_type@4', 'combined_shot_type@5','playoffs@0','playoffs@1','shot_type@0', 'shot_type@1']

TrainingDataFrame = TrainShotsMadeLater[pd.notnull(ShotsMade['shot_made_flag'])]
TestingDataFrame = TrainShotsMadeLater[pd.isnull(ShotsMade['shot_made_flag'])]
TestingDataFrameWithShotId = pd.DataFrame(columns = ['shot_id', 'shot_made_flag', 'shot_made_flag_2'])
TestingDataFrameWithShotId['shot_id'] = TestingDataFrame['shot_id']

#generateVisual(TrainingDataFrame, 'HomeOrAway', 'shot_made_flag')
#generateVisual(TrainingDataFrame, 'Conference', 'shot_made_flag')


X, y = TrainingDataFrame[IndependentVariables], TrainingDataFrame[DependentVariable]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.33)

LogisticModel = LogisticRegression(C = 0.1, max_iter = 5000)
fittedLogisticModel = LogisticModel.fit(X_train, Y_train)
Y_pred_prob = LogisticModel.predict_proba(X_train)
print('Log Loss', logLoss(Y_train, Y_pred_prob))
cvScoreLogistic = model_selection.cross_val_score(fittedLogisticModel, X_train, Y_train, cv = 5, scoring = 'neg_log_loss')
cvLogisticValidation = model_selection.cross_val_score(fittedLogisticModel, X_test, Y_test, cv = 5, scoring = 'neg_log_loss')
Y_predicted = fittedLogisticModel.predict(X_test)
tn, fp, fn, tp = confusion_matrix(Y_test, Y_predicted).ravel()
sensitivity = tp/(tp+fn) 
Specificity = tn/(tn+fp)
fpr, tpr, threshold = roc_curve(Y_test, Y_predicted)
rc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % rc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print('True Negative',tn, 'False Positive',fp, 'False Negative',fn, 'True Positive',tp)
print('Sensitivity', sensitivity, 'Specificity', Specificity)
print('True Negative',tn, 'False Positive',fp, 'False Negative',fn, 'True Positive',tp)
print(len(fittedLogisticModel.predict(TestingDataFrame[IndependentVariables])))
TestingDataFrameWithShotId['shot_made_flag'] = fittedLogisticModel.predict(TestingDataFrame[IndependentVariables])
print(TestingDataFrame.shape, TestingDataFrameWithShotId['shot_made_flag'].shape)
print(cvScoreLogistic, np.mean(cvScoreLogistic))
print(cvLogisticValidation, np.mean(cvLogisticValidation))


RandomClassifier = RandomForestClassifier(max_depth=8, criterion = 'entropy', max_features = 'log2')
fittedRandomClassfier = RandomClassifier.fit(X_train, Y_train)
Y_pred_prob_RandomForest = RandomClassifier.predict_proba(X_train)
print('Log Loss Random Forest', logLoss(Y_train, Y_pred_prob_RandomForest))
cvScoreRandom = model_selection.cross_val_score(fittedRandomClassfier, X_train, Y_train, cv = 5, scoring = 'neg_log_loss')
cvRandomValidation = model_selection.cross_val_score(fittedRandomClassfier, X_test, Y_test, cv = 5, scoring = 'neg_log_loss')
Y_predicted_random = fittedRandomClassfier.predict(X_test)
tn, fp, fn, tp = confusion_matrix(Y_test, Y_predicted_random).ravel()
sensitivity = tp/(tp+fn) 
Specificity = tn/(tn+fp)
fpr, tpr, threshold = roc_curve(Y_test, Y_predicted_random)
rc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % rc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print('True Negative',tn, 'False Positive',fp, 'False Negative',fn, 'True Positive',tp)
print('Sensitivity', sensitivity, 'Specificity', Specificity)
TestingDataFrameWithShotId['shot_made_flag_2'] = fittedRandomClassfier.predict(TestingDataFrame[IndependentVariables])
print(cvScoreRandom, np.mean(cvScoreRandom))
print(cvRandomValidation, np.mean(cvRandomValidation))
TestingDataFrameWithShotId.to_csv('Predictions.csv', sep = ',', header = True)
