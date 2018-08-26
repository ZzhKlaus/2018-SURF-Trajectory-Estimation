import csv
import os
import numpy as np
import pandas as pd
from standard_plots import *

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Embedding, Input, Activation, Bidirectional, TimeDistributed
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import theano
#import tensorflow


#Parameters
time_steps = 1
#学习率
learning_rate = 0.001 
#迭代次数
epochs = 200
#每块训练样本数
batch_size = 20
#LSTM Cell
n_hidden = 128
#类别
n_classes = 3


adam = Adam(lr=learning_rate)

with open('waypoint_trace_new.csv', 'r', newline='') as csvfile: 
	reader = csv.reader(csvfile)
	dataset = np.array([row for row in reader])

	Loc_x = dataset[1:,0]
	Loc_y = dataset[1:,1]
	Mag_x = dataset[1:,3]
	Mag_y = dataset[1:,4]
	Mag_z = dataset[1:,5]


# convert an array of values into a dataset matrix
def create_dataset_input(dataset, look_back=1):
    dataX = []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
    return np.array(dataX)

def create_dataset_output(dataset, look_back=1):
    dataY = []
    for i in range(len(dataset)-look_back):
        dataY.append(dataset[i + look_back])
    return  np.array(dataY)

def load_data(file_name, sequence_length=20, split=0.75):
	Loc_x = pd.read_csv(file_name, sep=',', usecols=[0])
	Loc_x = np.array(Loc_x).astype(float)
	scaler = MinMaxScaler()
	Loc_x = scaler.fit_transform(Loc_x)

	Loc_y = pd.read_csv(file_name, sep=',', usecols=[1])
	Loc_y = np.array(Loc_y).astype(float)
	scaler = MinMaxScaler()
	Loc_y = scaler.fit_transform(Loc_y)

	Mag_x = pd.read_csv(file_name, sep=',', usecols=[3])
	Mag_x = np.array(Mag_x).astype(float)
	scaler = MinMaxScaler()
	Mag_x = scaler.fit_transform(Mag_x)

	Mag_y = pd.read_csv(file_name, sep=',', usecols=[4])
	Mag_y = np.array(Mag_y).astype(float)
	scaler = MinMaxScaler()
	Mag_y = scaler.fit_transform(Mag_y)

	Mag_z = pd.read_csv(file_name, sep=',', usecols=[5])
	Mag_z = np.array(Mag_z).astype(float)
	scaler = MinMaxScaler()
	Mag_z = scaler.fit_transform(Mag_z)

	train_size = int(len(Mag_x) * split)
	test_size = len(Mag_x) - train_size
	train_loc_x, test_loc_x = Loc_x[0:train_size], Loc_x[train_size:len(Loc_x)]
	train_loc_y, test_loc_y = Loc_y[0:train_size], Loc_y[train_size:len(Loc_x)]
	train_mag_x, test_mag_x = Mag_x[0:train_size], Mag_x[train_size:len(Mag_x)]
	train_mag_y, test_mag_y = Mag_y[0:train_size], Mag_y[train_size:len(Mag_y)]
	train_mag_z, test_mag_z = Mag_z[0:train_size], Mag_z[train_size:len(Mag_z)]
	
	#train_loc_x, test_loc_x = train_loc_x[:,:,np.newaxis], test_loc_x[:,:,np.newaxis]
	#train_loc_y, test_loc_y = train_loc_y[:,:,np.newaxis], test_loc_y[:,:,np.newaxis]
	train_mag_x, test_mag_x = train_mag_x[:,:,np.newaxis], test_mag_x[:,:,np.newaxis]
	train_mag_y, test_mag_y = train_mag_y[:,:,np.newaxis], test_mag_y[:,:,np.newaxis]
	train_mag_z, test_mag_z = train_mag_z[:,:,np.newaxis], test_mag_z[:,:,np.newaxis]
	

	'''
	train_mag_x = create_dataset_input(train_mag_x, 1)
	#train_mag_x = np.reshape(train_mag_x, (len(train_mag_x),len(train_mag_x[0])))
	train_mag_y = create_dataset_input(train_mag_y, 1)
	#train_mag_y = np.reshape(train_mag_y, (len(train_mag_y),len(train_mag_y[0])))
	train_mag_z = create_dataset_input(train_mag_z, 1)
	#train_mag_z = np.reshape(train_mag_z, (len(train_mag_z),len(train_mag_z[0])))
	
	test_mag_x = create_dataset_input(test_mag_x, look_back)
	test_mag_y = create_dataset_input(test_mag_y, look_back)
	test_mag_z = create_dataset_input(test_mag_z, look_back)
	

	train_loc_x = create_dataset_input(train_loc_x, look_back)
	train_loc_y = create_dataset_input(train_loc_y, look_back)
	test_loc_x = create_dataset_input(test_loc_x, look_back)
	test_loc_y = create_dataset_input(test_loc_y, look_back)
	'''
	trainX = np.concatenate((train_mag_x,train_mag_y,train_mag_z),axis = 2)
	testX = np.concatenate((test_mag_x,test_mag_y,test_mag_z),axis = 2)
	trainY = np.concatenate((train_loc_x,train_loc_y),axis = 1)
	testY = np.concatenate((test_loc_x,test_loc_y),axis = 1)
	print(trainX.shape)
	print(trainX[0][0])
	print(trainX[0][0][0])
	return trainX,trainY,testX,testY
	
def build_model(batch_size = 20, time_steps = 1, feature_size = 3):
	# input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
	model = Sequential()
	#model.add(LSTM(120,batch_input_shape = (batch_size, time_steps, feature_size), stateful = True, return_sequences=True))
	
	# Dense(2 : outputs two value per timestep

	model.add(Bidirectional(LSTM(n_hidden, stateful = True, return_sequences=True),batch_input_shape = (batch_size, time_steps, feature_size)))
	model.add(Bidirectional(LSTM(n_hidden)))
	#TimeDistributed: Why cannot use it, return input_shape
	model.add(Dense(2))

	#model.add(Bidirectional(LSTM(200, stateful = True, return_sequences=True),batch_input_shape = (batch_size, time_steps, feature_size)))
	#model.add(LSTM(6, input_dim=1, return_sequences=True))
	#model.add(LSTM(6, input_shape=(None, 1),return_sequences=True))

	"""
	#model.add(LSTM(input_dim=1, output_dim=6,input_length=10, return_sequences=True))
	#model.add(LSTM(6, input_dim=1, input_length=10, return_sequences=True))
	model.add(LSTM(6, input_shape=(10, 1),return_sequences=True))
	"""
	model.add(Dropout(2))
	print(model.layers)
	#model.add(LSTM(100, return_sequences=True))
	#model.add(LSTM(100, return_sequences=True))
	#model.add(LSTM(60, return_sequences=True))
	#model.add(Dropout(0.3))
	
	#model.add(LSTM(30))
	#model.add(Dropout(0.3))
	
	#model.add(Dense(n_classes))
	#model.add(Activation('softmax'))
	#model.add(Activation('linear'))

	model.summary()
	model.compile(loss='mean_squared_error', optimizer=adam,metrics=['acc'])
	#metrics.mae, metrics.sparse_categorical_accuracy])
	return model



def train_model(train_x, train_y, test_x, test_y):
	model = build_model()

	try:
		model.fit(train_x, train_y, batch_size=20, epochs = epochs, verbose=1, validation_data=(test_x, test_y),shuffle = False) #validation_split=0.1
		#model.reset_states()
		trainScore = model.evaluate(train_x, train_y,batch_size=20, verbose=0)
		print('Train Score: ',trainScore)
		testScore = model.evaluate(test_x, test_y,batch_size=20, verbose=0)
		print('Test Score: ', testScore[0])
		print('Test Accuracy: ', testScore[1])
		#predict = model.predict(test_x)
		#predict = np.reshape(predict, (predict.size, ))
	except KeyboardInterrupt:
		#print(predict)
		print(test_y)
	#print(predict)
	print(test_y)
	try:
		fig = plt.figure(1)
		#plt.plot(predict, 'r:')
		plt.plot(test_y, 'g-')
		plt.legend(['predict', 'true'])
	except Exception as e:
		print(e)
	return   test_y #predict,


trainX, trainY, testX, testY = load_data('waypoint_trace_new.csv')
test_y = train_model(trainX, trainY, testX, testY) #predict_y, 



'''
trainLocX = create_dataset_output(train_loc_x,look_back)
trainLocY = create_dataset_output(train_loc_y,look_back)
testLocX = create_dataset_output(test_loc_x,look_back)
testLocY = create_dataset_output(test_loc_y,look_back)

trainMagX = create_dataset_input(train_mag_x,look_back)
trainMagY = create_dataset_input(train_mag_y,look_back)
trainMagZ = create_dataset_input(train_mag_z,look_back)
testMagX = create_dataset_input(test_mag_x,look_back)
testMagY = create_dataset_input(test_mag_y,look_back)
testMagZ = create_dataset_input(test_mag_z,look_back)
'''
'''
trainX = np.array([trainMagX,trainMagY,trainMagZ])
#trainX = np.reshape(trainX,(trainMagX.shape[0], trainMagX.shape[1]))
trainY = np.array([train_loc_x,train_loc_y])
testX = np.array([testMagX,testMagY,testMagZ])
testY = np.array([testLocX,testLocY])

print(trainX.shape)
print(trainY.shape)
'''
'''
trainMagX = np.reshape(trainMagX, (trainMagX.shape[0], trainMagX.shape[1], 1))
testMagX = np.reshape(testMagX, (testMagX.shape[0], testMagX.shape[1], 1))
trainMagY = np.reshape(trainMagY, (trainMagY.shape[0], trainMagY.shape[1], 1))

testMagY = np.reshape(testMagY, (testMagY.shape[0], testMagY.shape[1], 1))
trainMagZ = np.reshape(trainMagZ, (trainMagZ.shape[0], trainMagZ.shape[1], 1))
testMagZ = np.reshape(testMagZ, (testMagZ.shape[0], testMagZ.shape[1], 1))
'''









'''
model.add(LSTM(32,input_shape=(1,3)))
model.add(Dropout(0.3))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=batch_size, verbose=2)
'''
'''
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
print('Test Score: ', testScore)
'''


'''
theano.config.compute_test_value = "ignore"
# create and fit the LSTM network
batch_size = 1
model = Sequential()
model.add(LSTM(32,input_dim=3))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=batch_size, verbose=2)

model = Sequential()

model.add(LSTM(128, batch_input_shape = (batch_size, 1, 3), stateful = True, return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(128, batch_input_shape = (batch_size, 1, 3), stateful = True))
model.add(Dropout(0.3))

model.add(Dense(2))

'''