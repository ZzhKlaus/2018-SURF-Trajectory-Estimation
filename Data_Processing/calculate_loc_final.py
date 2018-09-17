'''
By Zhenghang(Klaus) Zhong

With data structure of samples X time_steps X features, and the time steps are 
created by sliding Window, which means there are a lot of repeating data in input
as well as output, the data sample is like (if with one feature):

[1 2 3 4 5 6
 2 3 4 5 6 7
 3 4 5 6 7 8
 4 5 6 7 8 9] 

The output is similar, to get a better location coordination, we get location info
of one point based on the mean of all its estimated state, that is its final location.

For above example, we have input of 9 points' info.

'''

import csv
import os
import numpy as np
import pandas as pd

#Adjust according to parameter setting
time_step = 30

def mean_of_line(dataset, item, all_steps):
	#the points number that is smaller than length of dataset
	if item < len(dataset):
		# 20 is decided by the time steps
		if item >= time_step:
			number = time_step
		else:
			number = item + 1
		
		row = len(dataset)
		length = len(dataset[0])
		total = 0
		j = 0
		
		for i in range(number):
			total += dataset[item][j].astype('float')
			item -= 1
			j +=1
		mean = total / number
		all_steps.append(mean)
		if(item == 0):
			print('0:',mean)	
	#Last several points (whos number equal to (time_step -1))
	else:
		row_start = len(dataset)-1
		total = 0
		j = item - len(dataset)+1
		#print('lengh:',time_step - (item - len(dataset)+1))
		for i in range(time_step - (item - len(dataset)+1)):
			total += dataset[row_start][j].astype('float')
			row_start -= 1
			j += 1
		mean = total / (time_step - (item - len(dataset)+1))
		all_steps.append(mean)
	
# read the raw output data		
with open('yhat_loc_x.csv', 'r', newline='') as csvfile: 
	reader = csv.reader(csvfile)
	dataset_x = np.array([row[1:] for row in reader])
	print(dataset_x)
with open('yhat_loc_y.csv', 'r', newline='') as csvfile: 
	reader = csv.reader(csvfile)
	dataset_y = np.array([row[1:] for row in reader])

#get the final estimated location of all points with x and y coordinates	
all_steps_x = []
for x in range(len(dataset_x)+(time_step-1)):
	mean_of_line(dataset_x,x,all_steps_x)
all_steps_x = pd.DataFrame(all_steps_x)
all_steps_x.to_csv('all_steps_x.csv', mode='w', header=['Loc_x'])

all_steps_y = []
for y in range(len(dataset_y)+(time_step-1)):
	mean_of_line(dataset_y,y,all_steps_y)
#print(all_steps_y)
all_steps_y = pd.DataFrame(all_steps_y)
all_steps_y.to_csv('all_steps_y.csv', mode='w', header=['Loc_y'])

all_steps_x= pd.read_csv(filepath_or_buffer = 'all_steps_x.csv', sep = ',',usecols = [1])
all_steps_y= pd.read_csv(filepath_or_buffer = 'all_steps_y.csv', sep = ',',usecols = [1])

all_steps = pd.concat([all_steps_x,all_steps_y],axis = 1)
all_steps.to_csv('all_steps.csv', mode='w', header=['Loc_x','Loc_y'])



