import warnings
warnings.filterwarnings("ignore")

import os
import gc
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils
import math as m


def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return metrics.log_loss(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)

def create_model(data, catcols):    
    print(len(catcols))
    inputs = []
    outputs = []
    for c in catcols:
        print(c)
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values)/2), 70))
        inp = layers.Input(shape=(1,))
        print(inp.shape)
        print(num_unique_values)
        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)
        out = layers.SpatialDropout1D(0.3)(out)
        out = layers.Reshape(target_shape=(embed_dim, ))(out)
        print(out.shape)
        inputs.append(inp)
        outputs.append(out)
    
    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    y = layers.Dense(2, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=y)
    return model

kobeData = pd.read_csv('./data.csv')
sampleSubmission = pd.read_csv('./sample_submission.csv')
kobeData['angle'] = kobeData.apply(lambda row: 90 if row['loc_y']==0 else m.degrees(m.atan(row['loc_x']/abs(row['loc_y']))),axis=1)
kobeData['angle_bin'] = pd.cut(kobeData.angle, 7, labels=range(7))
kobeData['angle_bin'] = kobeData.angle_bin.astype(int)
kobeData['matchup_code'] = kobeData.matchup.apply(lambda x: 0 if (x.split(' ')[1])=='@' else 1)
kobeData['action_type'] = kobeData.action_type.apply(lambda x: x.replace('-', ''))
kobeData['action_type'] = kobeData.action_type.apply(lambda x: x.replace('Follow Up', 'followup'))
kobeData['action_type'] = kobeData.action_type.apply(lambda x: x.replace('Finger Roll','fingerroll'))

kobeData['game_date'] = pd.to_datetime(kobeData.game_date)

#His performance shouldn't depend on year or month but let's try.
kobeData['game_date_month'] = kobeData.game_date.dt.month
kobeData['game_date_quarter'] = kobeData.game_date.dt.quarter
#total time
kobeData['time_remaining'] = kobeData.apply(lambda row: row['minutes_remaining']*60+row['seconds_remaining'], axis=1)


kobeData['ClutchOrNot'] = kobeData.apply(lambda row: 1 if row['time_remaining'] <= 300 and row['period'] >=4 else 0, axis = 1)

#As seen from visualizations last 3 seconds success rate is lower.
kobeData['timeUnder4'] = kobeData.time_remaining.apply(lambda x: 1 if x<4 else 0)

kobeData['distance_bin'] = pd.cut(kobeData.shot_distance, bins=10, labels=range(10))

predictors = kobeData.columns.drop(['game_event_id', 'game_id', 'game_date', 'minutes_remaining', 'seconds_remaining','lat', 'lon', 'team_id', 'team_name', 'matchup','time_remaining', 'angle','shot_id'])

totalKobeData = kobeData[predictors]


features = list(totalKobeData.columns)
features.remove('shot_made_flag')

for feat in features:
	lbl_enc = preprocessing.LabelEncoder()
	totalKobeData[feat] = lbl_enc.fit_transform(totalKobeData[feat].astype(str).fillna("-1").values)


train = totalKobeData.loc[pd.notnull(totalKobeData['shot_made_flag'])].reset_index(drop=True)
test = totalKobeData.loc[pd.isnull(totalKobeData['shot_made_flag'])].reset_index(drop=True)
test_data = [test.loc[:, features].values[:, k] for k in range(test.loc[:, features].values.shape[1])]

oof_preds = np.zeros((len(train)))
test_preds = np.zeros((len(test)))

skf = StratifiedKFold(n_splits=20)
for train_index, test_index in skf.split(train, train.shot_made_flag.values):
	X_train, X_test = train.iloc[train_index, :], train.iloc[test_index, :]
	X_train = X_train.reset_index(drop=True)
	X_test = X_test.reset_index(drop=True)
	y_train, y_test = X_train.shot_made_flag.values, X_test.shot_made_flag.values
	model = create_model(train, features)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	X_train = [X_train.loc[:, features].values[:, k] for k in range(X_train.loc[:, features].values.shape[1])]
	X_test = [X_test.loc[:, features].values[:, k] for k in range(X_test.loc[:, features].values.shape[1])]
	
	es = callbacks.EarlyStopping(monitor='acc', min_delta=0.001, patience=5,
                                 verbose=1, mode='max', baseline=None, restore_best_weights=True)

	rlr = callbacks.ReduceLROnPlateau(monitor='acc', factor=0.5,
                                      patience=3, min_lr=1e-6, mode='max', verbose=1)
    
	model.fit(X_train,
              utils.to_categorical(y_train),
              validation_data=(X_test, utils.to_categorical(y_test)),
              verbose=1,
              batch_size=1024,
              callbacks=[es, rlr],
              epochs=25
             )
	valid_fold_preds = model.predict(X_test)[:, 1]
	test_fold_preds = model.predict(test_data)[:, 1]
	print(len(test_fold_preds))
	sampleSubmission['shot_made_flag'] = np.argmax(model.predict(test_data), axis = 1)
	oof_preds[test_index] = valid_fold_preds.ravel()
	test_preds += test_fold_preds.ravel()
	print(metrics.log_loss(y_test, valid_fold_preds))
	K.clear_session()

sampleSubmission.to_csv('SampleSubmission.csv')
# model = create_model(train, features)
# #print(model.summary())
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5,verbose=1, mode='max', baseline=None, restore_best_weights=True)

# rlr = callbacks.ReduceLROnPlateau(monitor='acc', factor=0.5,patience=3, min_lr=1e-6, mode='max', verbose=1)

# #print(train.loc[:,trainFeatures].to_numpy())

# X_train = [train.loc[:, features].values[:, k] for k in range(train.loc[:, features].values.shape[1])]
# X_test = [test.loc[:, features].values[:, k] for k in range(test.loc[:, features].values.shape[1])]

# model.fit(X_train, utils.to_categorical(train.shot_made_flag.values),batch_size = 1024, epochs = 50, callbacks = [rlr])
# values = model.predict(X_test, batch_size = 1024, callbacks = [rlr])

# print(values)