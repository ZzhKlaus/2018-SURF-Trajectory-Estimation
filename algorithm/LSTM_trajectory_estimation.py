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
	raw_Loc_x = pd.read_csv(file_name, sep=',', usecols=[0])
	raw_Loc_x = np.array(raw_Loc_x).astype(float)
	scaler_loc_x = MinMaxScaler()
	Loc_x = scaler_loc_x.fit_transform(raw_Loc_x)

	raw_Loc_y = pd.read_csv(file_name, sep=',', usecols=[1])
	raw_Loc_y = np.array(raw_Loc_y).astype(float)
	scaler_loc_y = MinMaxScaler()
	Loc_y = scaler_loc_y.fit_transform(raw_Loc_y)

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

	#To get ground truth of evaluation data
	#begin
	raw_final = test_size -1
	test_raw_loc_x = pd.DataFrame(raw_Loc_x[train_size:(train_size+raw_final)])
	test_raw_loc_y = pd.DataFrame(raw_Loc_y[train_size:(train_size+raw_final)])
	test_raw_loc = pd.concat([test_raw_loc_x,test_raw_loc_y],axis = 1)
	test_raw_loc.to_csv('test_raw_loc.csv', mode='w', header= ['Loc_x','Loc_y'])
	#end

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


def build_model(hidden_nodes,batch_size , time_steps = look_back, feature_size = 3):
	# train_x has shape of (n_samples, time_steps, input_dim)
	
	#define model, a stacked two-layer LSTM
	inputs1 = Input(batch_shape = (batch_size,look_back,feature_size))
	lstm1 = LSTM(hidden_nodes, stateful = True, return_sequences=True, return_state=True)(inputs1)
	lstm1 = Dropout(0.2)(lstm1)
	lstm1 = LSTM(hidden_nodes,return_sequences=True)(lstm1)
	lstm1 = Dropout(0.2)(lstm1)
	lstm1 = TimeDistributed(Dense((2)))(lstm1)
	model = Model(input = inputs1, outputs = lstm1)
	print(model.layers)
	model.compile(loss='mean_squared_error', optimizer=adam,metrics=['acc'])
	model.summary()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(batch_size, look_back, 3)
	yhat = model.predict(X, batch_size=batch_size)
	return yhat

# inverse scaling for a forecasted value
def invert_scale(scaler, yhat):
	inverted = scaler.inverse_transform(yhat)
	return inverted[:]

def train_model(hidden_nodes,batch_size,epochs,file_structure,file_acc2loss):
	scaler_loc_x,scaler_loc_y,train_x, train_y, test_x, test_y = load_data(file_name = 'waypoint_trace_new.csv', batch_size = batch_size)
	#,scaler_mag_x,scaler_mag_y,scaler_mag_z
	#For funtional API
	#model,hidden_state = build_model(batch_size=batch_size)

	model = build_model(hidden_nodes,batch_size=batch_size)
	
	#draw the model structure
	plot_model(model, show_shapes=True,to_file=os.path.join(resultpath, file_structure))

	# input data to model and train
	print('train_x.shape:',train_x.shape)
	print('train_y.shape:',train_y.shape)
	print('test_x.shape:',test_x.shape) 
	print('test_y.shape:',test_y.shape)
	for i in range(epochs):
		history = model.fit(train_x, train_y, batch_size=batch_size, epochs = 1, verbose=1,shuffle = False) #validation_split=0.1, validation_data=(test_x, test_y)
		# need to reset state for every epoch
		model.reset_states()
		#print('hidden_state:',hidden_state)
		# list all data in history
		'''
		print('history.keys()',hist.history.keys())
		# summarize history for accuracy
		plt.plot(hist.history['acc'])
		plt.plot(hist.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		'''
		print('Real Epoches:',i+1)
		with open(file_acc2loss,'a', newline='') as csvfile:
			if not os.path.getsize(file_acc2loss):        #file is empty
				spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
				spamwriter.writerow(['epochs','loss','acc'])#, 'val_loss','val_acc' 
				
			data = ([
				i,history.history['loss'][0],history.history['acc'][0]#, history.history['val_loss'][0], history.history['val_acc'][0]
				])  
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow(data)
	return model 

def my_save_model(hidden_nodes,batch_size,epochs,file_structure,file_acc2loss,file_model):
	model = train_model(hidden_nodes,batch_size,epochs,file_structure,file_acc2loss)
	model.save(os.path.join(resultpath, file_model))
	print('Model have been saved to my_model.h5')

def my_load_model():
	# test data
	batch_size =  20

	scaler_loc_x,scaler_loc_y,train_x, train_y, test_x, test_y = load_data(file_name = 'waypoint_trace_new.csv', batch_size = batch_size)
	
	print(len(test_x))
	print(len(test_y))
	print(len(test_x)/batch_size)
	model2 = load_model(os.path.join(resultpath, 'my_model_256_10_30.h5'))
	print("Load model successfully!")
	model2.compile(loss='mean_squared_error', optimizer=adam,metrics=['acc'])
	print("Compile model successfully!")

	predictions = list()
	for i in range(int(len(test_x)/batch_size)):
		#predict
		print('%d / %d:' % (i,int(len(test_x)/batch_size)))
		yhat = forecast_lstm(model2, batch_size=batch_size,X = test_x[i*batch_size:(i+1)*batch_size])

		#invert scaling		
		yhat_loc_x = invert_scale(scaler_loc_x,yhat[:,:,0])
		yhat_loc_x = np.reshape(yhat_loc_x,(batch_size,len(yhat_loc_x[0])))	
		yhat_loc_y = invert_scale(scaler_loc_y,yhat[:,:,1])
		yhat_loc_y = np.reshape(yhat_loc_y,(batch_size,len(yhat_loc_y[0])))

		test_raw_loc_x = invert_scale(scaler_loc_x,test_x[:,:,0])
		test_raw_loc_x = np.reshape(test_raw_loc_x[i*batch_size:(i+1)*batch_size],(batch_size,len(test_raw_loc_x[1])))
		test_raw_loc_y = invert_scale(scaler_loc_y,test_x[:,:,1])
		test_raw_loc_y = np.reshape(test_raw_loc_y[i*batch_size:(i+1)*batch_size],(batch_size,len(test_raw_loc_y[1])))

		yhat_loc_x = pd.DataFrame(yhat_loc_x)
		yhat_loc_y = pd.DataFrame(yhat_loc_y)
		test_raw_loc_x = pd.DataFrame(test_raw_loc_x)
		test_raw_loc_y = pd.DataFrame(test_raw_loc_y)
		
		if not os.path.isfile('test_raw_loc_x.csv'):
			test_raw_loc_x.to_csv('test_raw_loc_x.csv', header=0)
		else: # else it exists so append without writing the header
			test_raw_loc_x.to_csv('test_raw_loc_x.csv', mode='a', header=False)

		if not os.path.isfile('test_raw_loc_y.csv'):
			test_raw_loc_y.to_csv('test_raw_loc_y.csv', header=0)
		else: # else it exists so append without writing the header
			test_raw_loc_y.to_csv('test_raw_loc_y.csv', mode='a', header=False)
		
		
		if not os.path.isfile('yhat_loc_x.csv'):
			yhat_loc_x.to_csv('yhat_loc_x.csv', header=0)
		else: # else it exists so append without writing the header
			yhat_loc_x.to_csv('yhat_loc_x.csv', mode='a', header=False)

		if not os.path.isfile('yhat_loc_y.csv'):
			yhat_loc_y.to_csv('yhat_loc_y.csv', header=0)
		else: # else it exists so append without writing the header
			yhat_loc_y.to_csv('yhat_loc_y.csv', mode='a', header=False)
		
		'''
		print('test_raw_loc:',test_raw_loc)
		print('yhat_loc:',yhat_loc)
		
		rmse = sqrt(mean_squared_error(test_raw_loc,yhat_loc))
		print(rmse)
		print('%d)Test RMSE: %.3f'% (i+1,rmse))
		'''
		'''
		print('yhat_loc_x:',yhat_loc_x)
		print('yhat_loc_y:',yhat_loc_y)
		print('test_raw_loc_x:',test_raw_loc_x)
		print('test_raw_loc_y:',test_raw_loc_y)
		'''
def main():
	#Train parameters in Grid
	'''
	scaler_loc_x,scaler_loc_y,trainX, trainY, test_x, test_y = load_data('waypoint_trace_new.csv')
	model = KerasClassifier(build_fn=build_model, verbose=0)
	batch_size = [1, 20, 50, 100, 200]
	epochs = [10,30, 50, 100,200, 300]
	param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
	print('Xshape:',trainX.shape)
	print('Yshape:',trainY.shape)
	grid_result = grid.fit(trainX, trainY)
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	for params, mean_score, scores in grid_result.grid_scores_:
		print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

	best_batchsize =  int(grid_result.best_params_['batch_size'])
	best_epoch =  int(grid_result.best_params_['nb_epoch'])
	print('best_batchsize:',best_batchsize)
	print('best_epoch:',best_epoch)
	'''

	# Train for one pair of parameters
	hidden_nodes = 256
	best_batchsize =  5
	best_epoch =  100 
	file_structure = 'model_ts=30_256_5_100.png'
	file_acc2loss =    'log_ts=30_256_5_100.csv'
	file_model =  'my_model_ts=30_256_5_100.h5'
	my_save_model(hidden_nodes,best_batchsize,best_epoch,file_structure,file_acc2loss,file_model)
	
	#load model
	my_load_model()


if __name__ == "__main__":
	main()


