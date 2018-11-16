#importing libraries
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

# summarize first and last 5 rows
#print(dataset.head(5))
#print(dataset.tail(5))

#plotting the graph
values = dataset.values #conversion into an array
groups = [0,1,2,3,5,6,7] #column numbers starting from 'pollution', leaving out wind direction (wnd_dir)
i = 1
#plotting each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i) #(len(groups), 1, i) -> plots len(groups)= 7 graphs (in 7 rows), in 1 column of index i
    pyplot.plot(values[:, group])
    #print((values[:, group]))
    pyplot.title(dataset.columns[group], y=0.5, loc='right') #y is the font number and loc is where the title must be
    i+=1
pyplot.show()

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

#reframe the scaled data as supervised table
reframed = series_to_supervised(scaled_values, 1, 1)
#print(reframed)
#drop columns we don't want to predict
reframed = reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis = 1)
print(reframed.head()) #prints the first 5 values

#splitting into training and test set
values = reframed.values
n_training_hours = 365*24 #Here, we are only taking one year data as training data
train_set = values[:n_training_hours, :]
test_set = values[n_training_hours:, :]
X_train, y_train = train_set[:,:-1], train_set[:,-1]
X_test, y_test = test_set[:,:-1], test_set[:,-1]

#reshaping input to be 3D; dimensions as accepted by LSTM [samples, timesteps, features]
#The input can also be in the following format as expressed in Keras documentation (batch_size, timesteps, input_dim)
#print(X_train.shape)
#print(X_train.shape[0])
#print(X_train.shape[1])
#print(X_test.shape)
#print(X_test.shape[0])
#print(X_test.shape[1])
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
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
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
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