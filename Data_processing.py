import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

train_length = len(data_train)
y_train = data_train['target']
data_train.drop('target', inplace=True, axis=1)
# cat_cols = ['v3', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v107', 'v110', 'v112',
#             'v113', 'v125']

# print(data_test.columns)
# print(data_train.columns)
complete_df = pd.concat([data_train, data_test], ignore_index=True)
complete_df.drop('ID', inplace=True, axis=1)
# complete_df.drop(cat_cols, inplace=True, axis=1)


# print(complete_df.columns)

def label_encoder(data):
    for f in data.columns:
        if data[f].dtypes == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
    return data


complete_df = label_encoder(complete_df)

complete_df.fillna(0, inplace=True)
# print(complete_df.isnull().any())
# print(complete_df.info())

train = complete_df.iloc[:train_length, ]
test = complete_df.iloc[train_length:, ]

train_X, validation_X, train_Y, validation_Y = train_test_split(train, y_train, test_size=0.1)

# Again divide the training data for testing

new_train_X, testing_data_X, new_train_Y, testing_data_Y = train_test_split(train_X, train_Y, test_size=0.2)
'''
d_train = xgb.DMatrix(new_train_X, new_train_Y, missing=np.nan)
d_validation = xgb.DMatrix(validation_X, validation_Y, missing=np.nan)
d_test = xgb.DMatrix(test, missing=np.nan)

new_d_test = xgb.DMatrix(testing_data_X, testing_data_Y)

watchlist = [(d_train, 'train'), (d_validation, 'validation')]

param = {'max_depth': 10, 'objective': "binary:logistic", 'nthread': 4, 'eval_metric': 'logloss',
         'n_estimators': 200, 'eta': 0.01,
         'learning_rate': 0.02, 'gamma': 0.001, 'subsample': 0.8, 'lambda': 1, 'alpha': 0}
# param['scale_pos_weight'] = float(np.sum(train_Y == 0)) / np.sum(train_Y == 1)
param['colsample_bytree'] = 0.2

num_round = 10000

gbm = xgb.train(param, d_train, num_round, watchlist, early_stopping_rounds=10)

prediction = gbm.predict(new_d_test)
prediction_1 = [1 if x > 0.5 else 0 for x in prediction]

print(confusion_matrix(testing_data_Y, prediction_1))
print(log_loss(testing_data_Y, prediction))
print(roc_auc_score(testing_data_Y, prediction))
'''
#####################################################################################################################
######################################################################################################################
# Trying Deep Learning
# print(len(train_X[0]))
# print(train_Y[0])
'''
new_train_X = new_train_X.as_matrix()
print(len(new_train_X[0]))
validation_X = validation_X.as_matrix()
testing_data_X = testing_data_X.as_matrix()

from keras.models import Sequential

model = Sequential()

from keras.layers.core import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(500, input_dim=131, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(300, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# "class_mode" defaults to "categorical". For correctly displaying accuracy
# in a binary classification problem, it should be set to "binary".
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              class_mode='binary')
model.fit(new_train_X,new_train_Y, batch_size=100, nb_epoch=10, show_accuracy=True,
          validation_data=(validation_X, validation_Y))
# objective_score = model.evaluate(test_X, test_Y, batch_size=32)
# print(objective_score)
prediction = model.predict_proba(testing_data_X, batch_size=200)
print(prediction[0:20])
prediction_1 = [1 if x>0.5 else 0 for x in prediction]

print(roc_auc_score(testing_data_Y, prediction, average='weighted'))
print(confusion_matrix(testing_data_Y, prediction_1))
print(log_loss(testing_data_Y, prediction))

# test_prediction = gbm.predict(d_test)
# sample_submission['PredictedProb'] = test_prediction

# print(sample_submission.head())
# sample_submission.to_csv('first_submission.csv', index=False)
'''
