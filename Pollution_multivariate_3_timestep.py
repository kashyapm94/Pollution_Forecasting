# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 19:12:36 2018

@author: kashy
"""

import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder#LabelEncoder is the class for encoding names to numbers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#function parsing date and time
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

#loading the file and parsing the date using the parser function
dataset = pd.read_csv('pollution.csv', index_col = 0, header = 0, parse_dates = [['year', 'month', 'day', 'hour']], date_parser = parse)
#print (dataset)

#dropping the column No and manually specifying column names
dataset.drop ('No', axis = 1, inplace = True)
dataset.index.name = 'date'
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
#print(dataset)

#replacing NaN with zeros
dataset.fillna(value = '0', axis = 1, inplace = True)
#print(dataset)

#dropping the first 24 hrs
dataset = dataset[24:]
#print(dataset)

values = dataset.values

#conversion dataset from series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#encoding the wind direction column
#print(values[:,4])
labelencoder_X = LabelEncoder() #labelencoder_X is the object
values[:, 4] = labelencoder_X.fit_transform(values[:, 4])
#print(values[:,4])

#ensuring all the data is a float type
values = values.astype('float32')
#print(values)

#Normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)
n_hours = 3
n_features = 8

#reframe the scaled data as supervised table
reframed = series_to_supervised(scaled_values, n_hours, 1)
#print(reframed)

#splitting into training and test set
values = reframed.values
n_training_hours = 365*24 #Here, we are only taking one year data as training data
n_observations = n_hours*n_features
train_set = values[:n_training_hours, :]
test_set = values[n_training_hours:, :]
X_train, y_train = train_set[:,:n_observations], train_set[:,-n_features]
X_test, y_test = test_set[:,:n_observations], test_set[:,-n_features]

#reshaping input to be 3D; dimensions as accepted by LSTM [samples, timesteps, features]
#The input can also be in the following format as expressed in Keras documentation (batch_size, timesteps, input_dim)
X_train = X_train.reshape(X_train.shape[0], n_hours, n_features)
X_test = X_test.reshape(X_test.shape[0], n_hours, n_features)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#Network design
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], n_hours*n_features))
X_test = scaler.inverse_transform(X_test)

#invert scaling for forecast
# create empty table with 8 fields
yhat_inv = np.zeros(shape=(len(yhat), 8))
# put the predicted values in the right field
yhat_inv[:,0] = yhat[:,0]
# inverse transform and then select the right field
yhat = scaler.inverse_transform(yhat_inv)[:,0]

# invert scaling for actual
y_test_inv = np.zeros(shape=(len(y_test), 8))
y_test = y_test.reshape(y_test.shape[0],1) #Important to change the dimensions of y_test from (35039,) to (35039,1)
y_test_inv[:,0] = y_test[:,0]
y_test = scaler.inverse_transform(y_test_inv)[:,0]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test,yhat))

print('Test RMSE: %.3f' % rmse)