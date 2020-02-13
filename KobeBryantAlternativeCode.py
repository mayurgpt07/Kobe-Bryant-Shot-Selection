#importing
import pandas as pd

from sklearn import model_selection
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_rows = 500
pd.options.display.max_columns = 50

import warnings

warnings.filterwarnings('ignore')

import sklearn.metrics as metrics


from sklearn.ensemble import  RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
#import xgboost
import math as m


#reading files
df = pd.read_csv('./data.csv')

sub = pd.read_csv('./sample_submission.csv')

df.head()
df.dtypes
df.describe(include=['int64'])
df.describe(include=['object', 'category'])
#seperating train and test set, for some visualizations (especially countplots)
df_test = df.loc[df.shot_made_flag.isnull(), :]
df_test.index = range(len(df_test))

df.dropna(inplace=True)

df.index =  range(len(df))


sns.countplot(df.shot_made_flag)

print('Initial columns Addressed',df.columns)

plt.figure(figsize=(12,12))
plt.subplot(121)
plt.scatter(df.loc[df.shot_made_flag==1, 'loc_x'], df.loc[df.shot_made_flag==1, 'loc_y'], alpha=0.08, c='g')
plt.title('Shots Made')
plt.ylim(-100, 900)
plt.subplot(122)
plt.scatter(df.loc[df.shot_made_flag==0, 'loc_x'], df.loc[df.shot_made_flag==0, 'loc_y'], alpha=0.08, c='r')
plt.title('Shots Missed')
plt.ylim(-100, 900)
##plt.show()

c= df.loc[(df.loc_x==0) & (df.loc_y==0)]

print('Location shots fired',c.shot_made_flag.value_counts())

c = df.groupby('minutes_remaining')['shot_made_flag'].mean()
plt.plot(c.index, c.values)
plt.xlabel('Minutes Remaining')
plt.ylabel('Mean(shot_made_flag)')
plt.scatter(c.index, c.values)
#plt.show()

sns.countplot(df.minutes_remaining)

c = df.groupby('seconds_remaining')['shot_made_flag'].mean()
plt.plot(c.index, c.values)
plt.xlabel('Seconds Remaining')
plt.ylabel('Mean(shot_made_flag)')
plt.scatter(c.index, c.values)
#plt.show()

plt.figure(figsize=(15,5))
sns.countplot(df.seconds_remaining)

c = df.groupby('shot_distance')['shot_made_flag'].mean()
plt.plot(c.index, c.values)
plt.xlabel('Shot_distance')
plt.ylabel('Mean(shot_made_flag)')
plt.scatter(c.index, c.values)
#plt.show()


#Just a random color generator, will use for visualizations.
def color_generator(num_colors):
    colors = []
    for i in range(num_colors):
        colors.append((np.random.rand(), np.random.rand(), np.random.rand()))
    return colors

#for plotting zone_wise points and checking their mean of target variable
def plot_zone_wise(zone_name):
    c_mean = df.groupby(zone_name)['shot_made_flag'].mean()
    plt.figure(figsize=(15,15))
    for i, area in enumerate(df[zone_name].unique()):
        plt.subplot(121)
        c = df.loc[(df[zone_name]==area)]
        plt.scatter(c.loc_x, c.loc_y,alpha=0.5 ,c=colors[i])
        plt.text(c.loc_x.mean(), c.loc_y.quantile(0.80), '%0.3f'%(c_mean[area]),size=15, bbox=dict(facecolor='red', alpha=0.5))
        plt.ylim(-100, 900)
    plt.legend(df[zone_name].unique())
    plt.title(zone_name)
    #plt.show()

colors = color_generator(100)
plot_zone_wise('shot_zone_area')

plot_zone_wise('shot_zone_basic')

plot_zone_wise('shot_zone_range')

c = df.groupby('period')['shot_made_flag'].mean()
plt.plot(c.index, c.values)
plt.scatter(c.index, c.values)
#plt.show()

sns.barplot('playoffs', 'shot_made_flag', data=df)

sns.countplot('playoffs', hue ='shot_made_flag', data=df)

sns.barplot('season', 'shot_made_flag', data=df)
plt.xticks(rotation='vertical')
#plt.show()

sns.barplot(df.shot_type, df.shot_made_flag)
sns.barplot(df.combined_shot_type, df.shot_made_flag)


plt.figure(figsize=(15,6))
sns.barplot('action_type', 'shot_made_flag', data=df)
plt.xticks(rotation='vertical')
#plt.show()

#getting combined data for feature transformation.
#df = pd.read_csv('../input/data.csv')

#The angle from which the shot was made.
df['angle'] = df.apply(lambda row: 90 if row['loc_y']==0 else m.degrees(m.atan(row['loc_x']/abs(row['loc_y']))),axis=1)

#Binning the angle, optimum size selected by cross validation.
df['angle_bin'] = pd.cut(df.angle, 7, labels=range(7))
df['angle_bin'] = df.angle_bin.astype(int)

plot_zone_wise('angle_bin')

#two types of valuein matchup @ and vs.. coding those values.
df['matchup_code'] = df.matchup.apply(lambda x: 0 if (x.split(' ')[1])=='@' else 1)

# Preprocessing the text for some words for later use.
df['action_type'] = df.action_type.apply(lambda x: x.replace('-', ''))
df['action_type'] = df.action_type.apply(lambda x: x.replace('Follow Up', 'followup'))
df['action_type'] = df.action_type.apply(lambda x: x.replace('Finger Roll','fingerroll'))

#using countvectorizer to generate feature matrix
cv = CountVectorizer(max_features=50, stop_words=['shot'])

shot_features = cv.fit_transform(df['action_type']).toarray()

shot_features = pd.DataFrame(shot_features, columns=cv.get_feature_names())

print(shot_features.head())

#combining with the dataframe
df = pd.concat([df,shot_features], axis=1)

df['game_date'] = pd.to_datetime(df.game_date)

#His performance shouldn't depend on year or month but let's try.
df['game_date_month'] = df.game_date.dt.month

df['game_date_quarter'] = df.game_date.dt.quarter

#total time
df['time_remaining'] = df.apply(lambda row: row['minutes_remaining']*60+row['seconds_remaining'], axis=1)
#As seen from visualizations last 3 seconds success rate is lower.
df['timeUnder4'] = df.time_remaining.apply(lambda x: 1 if x<4 else 0)

df['distance_bin'] = pd.cut(df.shot_distance, bins=10, labels=range(10))

ang_dist = df.groupby(['angle_bin', 'distance_bin'])['shot_made_flag'].agg([np.mean],as_index= False).reset_index()

ang_dist['group'] = range(len(ang_dist))

ang_dist.drop('mean', inplace=True, axis=1)

print('Combination of angle and distance',ang_dist.head())

print('Shape of angle and distance combined',ang_dist.shape)

df = df.merge(ang_dist, 'left', ['angle_bin', 'distance_bin'])
plot_zone_wise('group')

#The columns which we have now are
print('Final Columns',df.columns)

predictors = df.columns.drop(['game_event_id' #unique
                              , 'shot_made_flag'
                              , 'game_id' #unique
                              , 'shot_id' # id feature
                              , 'game_date'#other faetures from date used
                              , 'minutes_remaining'#transformed
                              , 'seconds_remaining'#transformed
                              ,'lat', 'lon' #same as loc_x, loc_y
                              , 'playoffs'#not important - from visualization
                              , 'team_id', 'team_name'#always same
                              , 'matchup' #transformed
                             ])
#TOP 21
predictors = ['action_type', 'combined_shot_type', 'shot_distance',
       'shot_zone_basic', 'opponent', 'matchup_code',
       'bank', 'driving', 'dunk', 'fadeaway', 'jump', 'pullup', 'running',
       'slam', 'turnaround','timeUnder4', 'angle_bin','loc_x', 'loc_y','period', 'season']

#label encoding
le = LabelEncoder()
for col in predictors:
    if df[col].dtype=='object':
        df[col] = le.fit_transform(df[col])


#seperating train and test set
df_test = df.loc[pd.isnull(df['shot_made_flag'])]
df_test.index = range(len(df_test))

print('Dataframe To test',df_test)

df.dropna(inplace=True)

df.index =  range(len(df))


X, y = df[predictors], df['shot_made_flag']
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.20)

parameters = {'n_estimators': [10,40,50,80, 100], 'learning_rate' : [0.1,0.5,1.0,1.05,1.1,1.5]}
RandomClassifier = AdaBoostClassifier()
clf = model_selection.GridSearchCV(RandomClassifier, parameters, scoring = 'neg_log_loss',cv=5, verbose=True, n_jobs = -1)
clf.fit(X_train[predictors], Y_train)
print(sorted(clf.cv_results_.keys()))
print(clf.best_estimator_)
print(clf.best_params_)



#XGBoost with best parameters
AdaBoostClassifier =  AdaBoostClassifier(n_estimators = 10, learning_rate = 0.1)

fittedAdaBoostClassifier = AdaBoostClassifier.fit(X[predictors], y)
Y_pred_prob_AdaBoost = AdaBoostClassifier.predict_proba(X_train[predictors])
#print('Log Loss error Ada Boost', logLoss(Y_train, Y_pred_prob_AdaBoost))
cvScoreAda = model_selection.cross_val_score(AdaBoostClassifier, X_train[predictors], Y_train, cv = 5, scoring = 'neg_log_loss')
cvAdaValidation = model_selection.cross_val_score(AdaBoostClassifier, X_test[predictors], Y_test, cv = 5, scoring = 'neg_log_loss')
Y_predicted_Ada = AdaBoostClassifier.predict(X_test[predictors])

tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_predicted_Ada).ravel()
sensitivity = tp/(tp+fn) 
Specificity = tn/(tn+fp)
fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_predicted_Ada)
rc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % rc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()
print('True Negative',tn, 'False Positive',fp, 'False Negative',fn, 'True Positive',tp)
print('Sensitivity', sensitivity, 'Specificity', Specificity)
print(cvScoreAda, np.mean(cvScoreAda))
print(cvAdaValidation, np.mean(cvAdaValidation))

print(df_test[predictors])
df_test['shot_made_flag'] = AdaBoostClassifier.predict(df_test[predictors])
print(df_test['shot_id','shot_made_flag'].head())

df_test['shot_id','shot_made_flag'].to_csv('late_submission.csv', index = False)
# #5-fold cross validation
# def run_test(predictors):
#     all_score = []
#     for train_index, test_index in kf:
#         xgb.fit(df.loc[train_index, predictors], df.loc[train_index, 'shot_made_flag'])
#         score = metrics.log_loss(df.loc[test_index, 'shot_made_flag'], xgb.predict_proba(df.loc[test_index, predictors])[:,1])
#         all_score.append(score)
#         print(score)
#     print('Mean score =', np.array(all_score).mean())

# run_test(predictors)


# AdaBoostClassifier.fit(df[predictors], df['shot_made_flag'])

# print(AdaBoostClassifier.predict_proba(df_test[predictors]))

# preds = AdaBoostClassifier.predict_proba(df_test[predictors])

# df_test['shot_made_flag'] = AdaBoostClassifier.predict_proba(df_test[predictors])[:,1]

# submission = df_test[['shot_id', 'shot_made_flag']]
# submission.to_csv('last_submission.csv', index=False)