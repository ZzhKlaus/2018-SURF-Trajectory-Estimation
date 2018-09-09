#Use scikit-learn to grid search the batch size and epochs
import csv
import os
import numpy as np
import pandas as pd
from standard_plots import *
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input
from keras.models import model_from_json, load_model
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Embedding, Input, Activation, Bidirectional, TimeDistributed, RepeatVector, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
resultpath = "result"

#Parameters

#time steps 
look_back = 20

#Optimizer Adam
adam = Adam(lr=learning_rate)

'''
filename = 'waypoint_trace_new.csv'
Loc_x = pd.read_csv(filepath_or_buffer = filename, sep = ',')["Loc_x"].values.astype('int')
Loc_y = pd.read_csv(filepath_or_buffer = filename, sep = ',')["Loc_y"].values.astype('int')
Mag_x = pd.read_csv(filepath_or_buffer = filename, sep = ',')["Mag_x"].values.astype('float')
Mag_y = pd.read_csv(filepath_or_buffer = filename, sep = ',')["Mag_y"].values.astype('float')
Mag_z = pd.read_csv(filepath_or_buffer = filename, sep = ',')["Mag_z"].values.astype('float')
'''
# convert an array of values into a dataset matrix for both input and output
def create_dataset_input(dataset, look_back=look_back):
	dataX = []
	for i in range(len(dataset)-look_back):
		dataX.append(dataset[i:(i+look_back)])
	return np.array(dataX)

#Process training data
def load_data(file_name, batch_size, split=0.75, look_back = look_back):
	Loc_x = pd.read_csv(file_name, sep=',', usecols=[0])
	Loc_x = np.array(Loc_x).astype(float)
	scaler_loc_x = MinMaxScaler()
	Loc_x = scaler_loc_x.fit_transform(Loc_x)

	Loc_y = pd.read_csv(file_name, sep=',', usecols=[1])
	Loc_y = np.array(Loc_y).astype(float)
	scaler_loc_y = MinMaxScaler()
	Loc_y = scaler_loc_y.fit_transform(Loc_y)

	Mag_x = pd.read_csv(file_name, sep=',', usecols=[3])
	Mag_x = np.array(Mag_x).astype(float)
	scaler_mag_x = MinMaxScaler()
	Mag_x = scaler_mag_x.fit_transform(Mag_x)

	Mag_y = pd.read_csv(file_name, sep=',', usecols=[4])
	Mag_y = np.array(Mag_y).astype(float)
	scaler_mag_y = MinMaxScaler()
	Mag_y = scaler_mag_y.fit_transform(Mag_y)

	Mag_z = pd.read_csv(file_name, sep=',', usecols=[5])
	Mag_z = np.array(Mag_z).astype(float)
	scaler_mag_z = MinMaxScaler()
	Mag_z = scaler_mag_z.fit_transform(Mag_z)

	train_size = int(len(Mag_x) * split)
	test_size = len(Mag_x) - train_size
	train_loc_x, test_loc_x = Loc_x[0:train_size], Loc_x[train_size:len(Loc_x)]
	train_loc_y, test_loc_y = Loc_y[0:train_size], Loc_y[train_size:len(Loc_x)]
	train_mag_x, test_mag_x = Mag_x[0:train_size], Mag_x[train_size:len(Mag_x)]
	train_mag_y, test_mag_y = Mag_y[0:train_size], Mag_y[train_size:len(Mag_y)]
	train_mag_z, test_mag_z = Mag_z[0:train_size], Mag_z[train_size:len(Mag_z)]
	
	print('shapeX:',train_mag_x.shape)
	train_mag_x = create_dataset_input(train_mag_x, look_back = look_back)
	print('shapeX:',train_mag_x.shape)
	train_mag_y = create_dataset_input(train_mag_y, look_back = look_back)
	train_mag_z = create_dataset_input(train_mag_z, look_back = look_back)

	
	test_mag_x = create_dataset_input(test_mag_x, look_back = look_back)
	test_mag_y = create_dataset_input(test_mag_y, look_back = look_back)
	test_mag_z = create_dataset_input(test_mag_z, look_back = look_back)
	
	#print('trian_mag_x:',train_mag_x)
	
	train_loc_x = create_dataset_input(train_loc_x, look_back = look_back)
	train_loc_y = create_dataset_input(train_loc_y, look_back = look_back)
	test_loc_x = create_dataset_input(test_loc_x, look_back = look_back)
	test_loc_y = create_dataset_input(test_loc_y, look_back = look_back)
	
	#train_loc_x, test_loc_x = train_loc_x[:,:,np.newaxis], test_loc_x[:,:,np.newaxis]
	#train_loc_y, test_loc_y = train_loc_y[:,:,np.newaxis], test_loc_y[:,:,np.newaxis]
	#train_mag_x, test_mag_x = train_mag_x[:,:,np.newaxis], test_mag_x[:,:,np.newaxis]
	#train_mag_y, test_mag_y = train_mag_y[:,:,np.newaxis], test_mag_y[:,:,np.newaxis]
	#train_mag_z, test_mag_z = train_mag_z[:,:,np.newaxis], test_mag_z[:,:,np.newaxis]

	trainX = np.concatenate((train_mag_x,train_mag_y,train_mag_z),axis = 2)
	testX = np.concatenate((test_mag_x,test_mag_y,test_mag_z),axis = 2)
	print('train_loc_x.shape:',train_loc_x.shape)
	trainY = np.concatenate((train_loc_x,train_loc_y),axis = 2)
	testY = np.concatenate((test_loc_x,test_loc_y),axis = 2)
	trainY = np.reshape(trainY, (len(trainY),look_back,2))
	print('trianY:',trainY.shape)
	#print('trianX:',trainX)
	print(trainX[0][0][0])
	lengthTrain = len(trainX)
	lengthTest = len(testX)
	while(lengthTrain % batch_size != 0):
		lengthTrain -= 1
	while(lengthTest % batch_size != 0):
		lengthTest -= 1


	return scaler_loc_x,scaler_loc_y,trainX[0:lengthTrain],trainY[0:lengthTrain],testX[0:lengthTest],testY[0:lengthTest]
	#,scaler_mag_x,scaler_mag_y,scaler_mag_z
