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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from scipy.sparse import hstack
from sklearn.metrics import log_loss, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA 
import warnings

warnings.simplefilter('ignore')

#Create Locations From Latitude and Longitude
def getPlacesFromLatandLong(train_data):
	train_data['HomeOrAway'] = train_data['matchup'].str.contains('vs').astype('int')

def createShortType(train_data):
	rare_action_types = train_data['action_type'].value_counts().sort_values().index.values[:20]
	train_data.loc[train_data['action_type'].isin(rare_action_types), 'action_type'] = 'Other'
	print(train_data['action_type'].value_counts().sort_values())
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
	train_data['ShotAngle'] = train_data.apply(lambda row: 90 if row['loc_y']==0 else math.degrees(math.atan(row['loc_x']/abs(row['loc_y']))),axis=1)
	train_data['Angle_Bin'] = pd.cut(train_data.ShotAngle, 7, labels = range(7))
	train_data['Angle_Bin'] = train_data.Angle_Bin.astype(int)

def shotDistance(train_data):
	train_data['shotClass'] = pd.cut(train_data.shot_distance, 7, labels = range(7))

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
ShotsMade['GameMonth'] = ShotsMade['DateinFormat'].apply(lambda x: x.month)
ShotsMade['GameYear'] = ShotsMade['DateinFormat'].apply(lambda x: x.year)
getPlacesFromLatandLong(ShotsMade)
secondsRemaning(ShotsMade)
createShortType(ShotsMade)
numberOfActionsInShot(ShotsMade)
shotAngle(ShotsMade)
conferenceTeams(ShotsMade)
shotDistance(ShotsMade)
print(ShotsMade['secondsTotal'].max(), ShotsMade['secondsTotal'].min())

ValidationShotsMade = ShotsMade[pd.isnull(ShotsMade['shot_made_flag'])]
TrainShotsMade = ShotsMade#ShotsMade[pd.notnull(ShotsMade['shot_made_flag'])]
print('Training Data shape', TrainShotsMade.shape)

# generateVisual(TrainShotsMade, 'shot_type', 'shot_made_flag')
CategoricalVariable = ['action_type','combined_shot_type','shot_type','shot_zone_area','shot_zone_basic','shot_zone_range', 'opponent','playoffs','period','season','ClutchOrNot','GameMonth','Angle_Bin','GameYear', 'shotClass']
DependentVariable = ['shot_made_flag']
NumericalVariable = ['game_event_id','game_id','lat','loc_x','loc_y','lon','minutes_remaining','seconds_remaining','shot_distance','secondsTotal','shot_made_flag','shot_id', 'NumberOfActions', 'ShotAngle', 'HomeOrAway','Conference']

TrainShotsMadeEncoded = prepare_inputs(TrainShotsMade[CategoricalVariable])
print('Train Shots Encoded Size',TrainShotsMadeEncoded.shape)
np.savetxt('FeatureOutput.csv', TrainShotsMadeEncoded, delimiter=',', header = 'action_type,combined_shot_type,shot_type,shot_zone_area,shot_zone_basic,shot_zone_range,opponent,playoffs,period,season,ClutchOrNot,GameMonth,Angle_Bin,GameYear,shotClass', fmt="%i", comments='')

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
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['shot_zone_range'],  prefix = 'shot_zone_range', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['shot_zone_area'],  prefix = 'shot_zone_area', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['action_type'],  prefix = 'action_type', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['playoffs'], prefix = 'playoffs', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['period'], prefix = 'period', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['ClutchOrNot'], prefix = 'ClutchOrNot', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['NumberOfActions'], prefix = 'NumberOfActions', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['HomeOrAway'], prefix = 'HomeOrAway', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['Conference'], prefix = 'Conference', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['GameMonth'], prefix = 'GameMonth', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['Angle_Bin'], prefix = 'Angle_Bin', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['GameYear'], prefix = 'GameYear', prefix_sep = '@'))
TrainShotsMadeLater = TrainShotsMadeLater.join(pd.get_dummies(TrainShotsMadeLater['shotClass'], prefix = 'shotClass', prefix_sep = '@'))
#TrainShotsMadeLater['Log_ShotDistance'] = TrainShotsMadeLater['shot_distance'].apply(lambda x: np.cbrt(x))
#TrainShotsMadeLater['Log_secondsTotal'] = TrainShotsMadeLater['secondsTotal'].apply(lambda x: np.cbrt(x))
print(list(TrainShotsMadeLater.columns))
TrainShotsMadeLater['AgeOfKobeBrayant'] = TrainShotsMadeLater['season'].apply(lambda x: x + 17)
print(TrainShotsMadeLater['season'].head(5),TrainShotsMadeLater['AgeOfKobeBrayant'].head(5))
#plt.plot(TrainShotsMade['combined_shot_type'], TrainShotsMade['shot_made_flag'], 'o', color='black');
#plt.hist(ShotsMade['shot_zone_range'], color = 'green')
#sns.catplot(x = 'shot_made_flag', y = 'secondsTotal',hue = 'period',data = TrainShotsMade)
#plt.show()

#'AgeOfKobeBrayant','ShotAngle','shot_distance','secondsTotal','loc_x','loc_y',

IndependentVariables = ['action_type@0', 'action_type@1', 'action_type@2', 'action_type@3', 'action_type@4', 'action_type@5', 'action_type@6', 
	'action_type@7', 'action_type@8', 'action_type@9', 'action_type@10', 'action_type@11', 'action_type@12', 'action_type@13', 'action_type@14', 'action_type@15', 
	'action_type@16', 'action_type@17', 'action_type@18', 'action_type@19', 'action_type@20', 'action_type@21', 'action_type@22', 'action_type@23', 'action_type@24', 
	'action_type@25', 'action_type@26', 'action_type@27', 'action_type@28', 'action_type@29', 'action_type@30', 'action_type@31', 'action_type@32', 'action_type@33', 
	'action_type@34', 'action_type@35', 'action_type@36', 'action_type@37','GameMonth@0', 'GameMonth@1', 'GameMonth@2', 'GameMonth@3', 'GameMonth@4', 'GameMonth@5', 
	'GameMonth@6', 'GameMonth@7', 'GameMonth@8','AgeOfKobeBrayant','shot_distance','secondsTotal', 'Conference@0', 'Conference@1',
	'HomeOrAway@0', 'HomeOrAway@1','ClutchOrNot@0', 'ClutchOrNot@1', 'NumberOfActions@1', 'NumberOfActions@2', 'NumberOfActions@3', 'NumberOfActions@4','combined_shot_type@0', 
	'combined_shot_type@1', 'combined_shot_type@2', 'combined_shot_type@3', 'combined_shot_type@4', 'combined_shot_type@5','playoffs@0','playoffs@1','shot_type@0', 'shot_type@1',
	'shot_zone_basic@0', 'shot_zone_basic@1', 'shot_zone_basic@2', 'shot_zone_basic@3', 'shot_zone_basic@4', 'shot_zone_basic@5', 'shot_zone_basic@6', 'shot_zone_range@0', 
	'shot_zone_range@1', 'shot_zone_range@2', 'shot_zone_range@3', 'shot_zone_range@4', 'shot_zone_area@0', 'shot_zone_area@1', 'shot_zone_area@2', 'shot_zone_area@3', 
	'shot_zone_area@4', 'shot_zone_area@5','Angle_Bin@0', 'Angle_Bin@1', 'Angle_Bin@2', 'Angle_Bin@3', 'Angle_Bin@4', 'Angle_Bin@5', 'Angle_Bin@6', 'GameYear@0', 'GameYear@1',
	'GameYear@2', 'GameYear@3', 'GameYear@4', 'GameYear@5', 'GameYear@6', 'GameYear@7', 'GameYear@8', 'GameYear@9', 'GameYear@10', 'GameYear@11', 'GameYear@12', 'GameYear@13', 
	'GameYear@14', 'GameYear@15', 'GameYear@16', 'GameYear@17', 'GameYear@18', 'GameYear@19', 'GameYear@20', 'shotClass@0', 'shotClass@1', 'shotClass@2', 'shotClass@3', 'shotClass@4', 'shotClass@5', 'shotClass@6']

TrainingDataFrame = TrainShotsMadeLater[pd.notnull(ShotsMade['shot_made_flag'])]
TestingDataFrame = TrainShotsMadeLater[pd.isnull(ShotsMade['shot_made_flag'])]
TestingDataFrameWithShotId = pd.DataFrame(columns = ['shot_id', 'shot_made_flag', 'shot_made_flag_2', 'shot_made_flag_3'])
TestingDataFrameWithShotId['shot_id'] = TestingDataFrame['shot_id']

#generateVisual(TrainingDataFrame, 'HomeOrAway', 'shot_made_flag')
#generateVisual(TrainingDataFrame, 'Conference', 'shot_made_flag')


X, y = TrainingDataFrame[IndependentVariables], TrainingDataFrame[DependentVariable]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.20)

# parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none'], 'C' : [0.1,0.7,0.01,0.05], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
# RandomClassifier = LogisticRegression(max_iter = 5000)
# clf = model_selection.GridSearchCV(RandomClassifier, parameters, scoring = 'neg_log_loss',cv=5, verbose=True, n_jobs = -1)
# clf.fit(X_train, Y_train)
# print(sorted(clf.cv_results_.keys()))
# print(clf.best_estimator_)
# print(clf.best_params_)

LogisticModel = LogisticRegression(C = 0.7, penalty = 'l1', solver = 'liblinear', max_iter = 5000)
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


# parameters = {'max_depth': [2,4,6,8,10,12,14], 'criterion' : ['entropy', 'gini'], 'max_features': ['log2', 'sqrt']}
# RandomClassifier = RandomForestClassifier()
# clf = model_selection.GridSearchCV(RandomClassifier, parameters, scoring = 'neg_log_loss',cv=5, verbose=True, n_jobs = -1)
# clf.fit(X_train, Y_train)
# print(sorted(clf.cv_results_.keys()))
# print(clf.best_estimator_)
# print(clf.best_params_)

RandomClassifier = RandomForestClassifier(max_depth=12, criterion = 'gini', max_features = 'sqrt')
#RandomClassifier = RandomForestClassifier(max_depth=8, criterion = 'gini', max_features = 'log2')


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
print('Random Train data',cvScoreRandom, np.mean(cvScoreRandom))
print('Random Validation Data',cvRandomValidation, np.mean(cvRandomValidation))



AdaBoostClassifier = AdaBoostClassifier(n_estimators = 100, random_state = 0)
fittedAdaBoostClassifier = AdaBoostClassifier.fit(X_train, Y_train)
Y_pred_prob_AdaBoost = AdaBoostClassifier.predict_proba(X_train)
print('Log Loss error Ada Boost', logLoss(Y_train, Y_pred_prob_AdaBoost))
cvScoreAda = model_selection.cross_val_score(AdaBoostClassifier, X_train, Y_train, cv = 5, scoring = 'neg_log_loss')
cvAdaValidation = model_selection.cross_val_score(AdaBoostClassifier, X_test, Y_test, cv = 5, scoring = 'neg_log_loss')
Y_predicted_Ada = AdaBoostClassifier.predict(X_test)

tn, fp, fn, tp = confusion_matrix(Y_test, Y_predicted_Ada).ravel()
sensitivity = tp/(tp+fn) 
Specificity = tn/(tn+fp)
fpr, tpr, threshold = roc_curve(Y_test, Y_predicted_Ada)
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
TestingDataFrameWithShotId['shot_made_flag_3'] = AdaBoostClassifier.predict(TestingDataFrame[IndependentVariables])
print(cvScoreAda, np.mean(cvScoreAda))
print(cvAdaValidation, np.mean(cvAdaValidation))
TestingDataFrameWithShotId.to_csv('Predictions.csv', sep = ',', header = True)