#By Zhenghang(Klaus) Zhong

import matplotlib.pyplot as plt
import csv
import pandas as pd

# file name strcture
# log_ts=20_128_10_120 (time_steps, hidden nodes, batch size, epochs, listed values are default)

#batch size
filename_20_128_1_30 = 'log_1_30.csv'
filename_20_128_5_30 = 'log_5_30.csv'
filename_20_128_10_30 = 'log_10_30.csv'
filename_20_128_20_30 = 'log_20_30.csv'

#batch size
filename_20_256_5_30 = 'log_ts=20_256_5_30.csv'
filename_20_256_10_30 = 'log_256_10_30.csv'
filename_20_256_20_30 = 'log_256_20_30.csv'
filename_20_256_30_30 = 'log_256_30_30.csv'
filename_20_256_40_30 = 'log_256_40_30.csv'

#batch size
filename_20_16_20_30 = 'log_16_20_30.csv'
filename_20_32_20_30 = 'log_32_20_30.csv'
filename_20_64_20_30 = 'log_64_20_30.csv'
filename_20_128_20_30 = 'log_20_30.csv'
filename_20_256_20_30 = 'log_256_20_30.csv'
filename_20_512_20_30 = 'log_512_20_30.csv'

#time steps
filename_10_128_10_120 = 'log_ts=10_128_10_120.csv'
filename_20_128_10_30 = 'log_ts=20_128_10_120.csv'
#filename_20_128_10_30 = 'log_10_30.csv'
filename_30_128_10_120 = 'log_ts=30_128_10_120.csv'
filename_40_128_10_30 = 'log_ts=40_128_10_30.csv'

filename_20_128_20_100 = 'log_20_100.csv'

filename_20_128_10_120 = 'log_ts=20_128_10_120.csv'

filename_30_512_5_100 = 'log_ts=30_512_5_100.csv'

def returnAccLoss(filename):
	csvfile = open(filename, "r")
	Hlength=len(csvfile.readlines())
	print('File \'%s\' length:  %d' % (filename, Hlength-1))

	epochs= pd.read_csv(filepath_or_buffer = filename, sep = ',')["epochs"].values
	acc= pd.read_csv(filepath_or_buffer = filename, sep = ',')["acc"].values
	loss= pd.read_csv(filepath_or_buffer = filename, sep = ',')["loss"].values
	
	#epochs = epochs[:] + 1
	#print(epochs)
	return epochs, acc, loss

epochs_20_128_1_30, acc_20_128_1_30, loss_20_128_1_30 = returnAccLoss(filename_20_128_1_30)
epochs_20_128_5_30, acc_20_128_5_30, loss_20_128_5_30 = returnAccLoss(filename_20_128_5_30)
epochs_20_128_10_30, acc_20_128_10_30, loss_20_128_10_30 = returnAccLoss(filename_20_128_10_30)
epochs_20_128_20_30, acc_20_128_20_30, loss_20_128_20_30 = returnAccLoss(filename_20_128_20_30)
epochs_20_128_10_120, acc_20_128_10_120, loss_20_128_10_120 = returnAccLoss(filename_20_128_10_120)
'''
plt.plot(epochs_20_128_1_30,acc_20_128_1_30, color = 'green',label = 'batch_size = 1')
plt.plot(epochs_20_128_5_30,acc_20_128_5_30, color = 'red',label = 'batch_size = 5')
plt.plot(epochs_20_128_10_30,acc_20_128_10_30, color = 'black',label = 'batch_size = 10')
plt.plot(epochs_20_128_20_30,acc_20_128_20_30, color = 'blue',label = 'batch_size = 20')
plt.plot(epochs_20_128_10_120,acc_20_128_10_120, color = 'y',label = 'batch_size = 10_ep=120')
'''
'''
plt.plot(epochs_20_128_1_30,loss_20_128_1_30, color = 'green',label = 'batch_size = 1')
plt.plot(epochs_20_128_5_30,loss_20_128_5_30, color = 'red',label = 'batch_size = 5')
plt.plot(epochs_20_128_10_30,loss_20_128_10_30, color = 'skyblue',label = 'batch_size = 10')
plt.plot(epochs_20_128_20_30,loss_20_128_20_30, color = 'blue',label = 'batch_size = 20')
'''

'''
# Comparing accuracy of different batchs size 
epochs_20_256_5_30, acc_20_256_5_30, loss_20_256_5_30 = returnAccLoss(filename_20_256_5_30)
epochs_20_256_10_30, acc_20_256_10_30, loss_20_256_10_30 = returnAccLoss(filename_20_256_10_30)
epochs_20_256_20_30, acc_20_256_20_30, loss_20_256_20_30 = returnAccLoss(filename_20_256_20_30)
epochs_20_256_30_30, acc_20_256_30_30, loss_20_256_30_30 = returnAccLoss(filename_20_256_30_30)
epochs_20_256_40_30, acc_20_256_40_30, loss_20_256_40_30 = returnAccLoss(filename_20_256_40_30)
plt.plot(epochs_20_256_5_30,acc_20_256_5_30, color = 'black',label = 'batch_size = 5')
plt.plot(epochs_20_256_10_30,acc_20_256_10_30, color = 'green',label = 'batch_size = 10')
plt.plot(epochs_20_256_20_30,acc_20_256_20_30, color = 'red',label = 'batch_size = 20')
plt.plot(epochs_20_256_30_30,acc_20_256_30_30, color = 'skyblue',label = 'batch_size = 30')
plt.plot(epochs_20_256_40_30,acc_20_256_40_30, color = 'blue',label = 'batch_size = 40')
'''
#begin drawing: comparing different batchs size	
#plt.title('')

'''
#Comparing accuracy of different hidden nodes 
epochs_20_16_20_30, acc_20_16_20_30, loss_20_16_20_30 = returnAccLoss(filename_20_16_20_30)
epochs_20_32_20_30, acc_20_32_20_30, loss_20_32_20_30 = returnAccLoss(filename_20_32_20_30)
epochs_20_64_20_30, acc_20_64_20_30, loss_20_64_20_30 = returnAccLoss(filename_20_64_20_30)
epochs_20_128_20_30, acc_20_128_20_30, loss_20_128_20_30 = returnAccLoss(filename_20_128_20_30)
epochs_20_256_20_30, acc_20_256_20_30, loss_20_256_20_30 = returnAccLoss(filename_20_256_20_30)
epochs_20_512_20_30, acc_20_512_20_30, loss_20_512_20_30 = returnAccLoss(filename_20_512_20_30)
plt.plot(epochs_20_16_20_30,acc_20_16_20_30, color = 'black',label = 'hidden_nodes = 16')
plt.plot(epochs_20_32_20_30,acc_20_32_20_30, color = 'green',label = 'hidden_nodes = 32')
plt.plot(epochs_20_64_20_30,acc_20_64_20_30, color = 'red',label = 'hidden_nodes = 64')
plt.plot(epochs_20_128_20_30,acc_20_128_20_30, color = 'skyblue',label = 'hidden_nodes = 128')
plt.plot(epochs_20_256_20_30,acc_20_256_20_30, color = 'blue',label = 'hidden_nodes = 256')
plt.plot(epochs_20_512_20_30,acc_20_512_20_30, color = 'y',label = 'hidden_nodes = 512')
'''
'''
epochs_10_128_10_120, acc_10_128_10_120, loss_10_128_10_120 = returnAccLoss(filename_10_128_10_120)
epochs_20_128_10_30, acc_20_128_10_30, loss_20_128_10_30 = returnAccLoss(filename_20_128_10_30)
epochs_30_128_10_120, acc_30_128_10_120, loss_30_128_10_120 = returnAccLoss(filename_30_128_10_120)
#epochs_40_128_10_30, acc_40_128_10_30, loss_40_128_10_120 = returnAccLoss(filename_40_128_10_30)

plt.plot(epochs_10_128_10_120[0:91],acc_10_128_10_120[0:91], color = 'black',label = 'time_steps = 10')
plt.plot(epochs_20_128_10_30,acc_20_128_10_30, color = 'green',label = 'time_steps = 20')
plt.plot(epochs_30_128_10_120[0:91],acc_30_128_10_120[0:91], color = 'red',label = 'time_steps = 30')
#plt.plot(epochs_40_128_10_30,acc_40_128_10_30, color = 'skyblue',label = 'time_steps = 40')
'''
epochs_20_128_20_100, acc_20_128_20_100, loss_20_128_20_100 = returnAccLoss(filename_20_128_20_100)
plt.plot(epochs_20_128_20_100,acc_20_128_20_100, color = 'y',label = 'ts = 20,hn=128,bs=20')

epochs_30_512_5_100, acc_30_512_5_100, loss_30_512_5_100 = returnAccLoss(filename_30_512_5_100)
plt.plot(epochs_30_512_5_100,acc_30_512_5_100, color = 'blue',label = 'ts = 30,hn=512,bs=5')



plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
'''
with open('log_1_30.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[2] for row in reader]
'''

'''
csv_file = csv.reader(open('log_1_30.csv','r'))
print(csv_file)

for stu in csv_file:
	print(stu)
'''
