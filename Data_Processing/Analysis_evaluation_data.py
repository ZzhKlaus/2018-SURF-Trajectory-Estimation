'''
By Zhenghang(Klaus) Zhong

To get the difference of every estimated point and real point
'''

import csv
import os
import numpy as np
import pandas as pd
import math 

with open('all_steps.csv', 'r', newline='') as csvfile: 
	reader = csv.reader(csvfile)
	dataset_evaluate = np.array([row[1:] for row in reader])

with open('test_raw_loc.csv', 'r', newline='') as csvfile: 
	reader = csv.reader(csvfile)
	dataset_raw = np.array([row[1:] for row in reader])

dis_diff = []
for i in range(len(dataset_raw)):
	x_pow_diff = math.pow(dataset_evaluate[i][0].astype('float')-dataset_raw[i][0].astype('float'),2)
	y_pow_diff = math.pow(dataset_evaluate[i][1].astype('float')-dataset_raw[i][1].astype('float'),2)
	dis_diff.append(math.sqrt(x_pow_diff+y_pow_diff))

print(dis_diff)
dis_diff = pd.DataFrame(dis_diff)
dis_diff.to_csv('dis_diff.csv', mode='a', header=False)