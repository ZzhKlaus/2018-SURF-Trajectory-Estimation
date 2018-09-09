'''
Usually we donot need this file.

As at first, we forget to call the function to refresh file 'tempList.csv', 
rather than re-detecting the data, I found that we could delete the repeated 
info from tail to head, as the info is stored into database in groups, compare 
these groups one by one conversely, and delte info that is repeated in latter one.

'''
import csv
import pandas as pdb
import numpy as np
import tensorflow as tf
#from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

PYTHONIOENCODING="UTF-8"  #set the utf-8 encode mode


FILE_0_old = '0_xxx_0_1_5E_39.csv'
FILE_1_old = '1_xxx_0_1_5SW.csv'
FILE_2_old = '2_xxx_0_2_0_3_5f.csv'
FILE_3_old = '3_xxx_0_4_5f.csv'
FILE_4_old = '4_xxx_0_5_5f.csv'
FILE_5_old = '5_xxx_0_6_5f.csv'
FILE_6_old = '6_xxx_0_7_5f.csv'
FILE_7_old = '7_xxx_0_8_5f.csv'
FILE_8_old = '8_xxx_0_9_5f.csv'
FILE_9_old = '9_xxx_0_10_5f.csv'
FILE_10_old = '10_xxx_0_11_5f.csv'
FILE_11_old = '11_xxx_0_12_5f.csv'
FILE_12_old = '12_xxx_0_0_1_4f.csv'
FILE_13_old = '13_xxx_0_2_4f.csv'
FILE_14_old = '14_xxx_0_3_4_5f.csv'
FILE_15_old = '15_xxx_0_5_4f.csv'
FILE_dif_direc_old = 'diff-direction-0_5_4f.csv'

FILE_0_new = '0_new_0_1_5E_39.csv'
FILE_1_new = '1_new_0_1_5SW.csv'
FILE_2_new = '2_new_0_2_0_3_5f.csv'
FILE_3_new = '3_new_0_4_5f.csv'
FILE_4_new = '4_new_0_5_5f.csv'
FILE_5_new = '5_new_0_6_5f.csv'
FILE_6_new = '6_new_0_7_5f.csv'
FILE_7_new = '7_new_0_8_5f.csv'
FILE_8_new = '8_new_0_9_5f.csv'
FILE_9_new = '9_new_0_10_5f.csv'
FILE_10_new = '10_new_0_11_5f.csv'
FILE_11_new = '11_new_0_12_5f.csv'
FILE_12_new = '12_new_0_0_1_4f.csv'
FILE_13_new = '13_new_0_2_4f.csv'
FILE_14_new = '14_new_0_3_4_5f.csv'
FILE_15_new = '15_new_0_5_4f.csv'
FILE_dif_direc_new = 'new_diff-direction-0_5_4f.csv'

FILE_mix_0_old = 'MIX2_0_xxx_0_0_1_4f.csv'
FILE_mix_1_old = 'MIX2_1_xxx_0_2_4f.csv'
FILE_mix_2_old = 'MIX2_2_xxx_0_3-S27_4f.csv'
FILE_mix_3_old = 'MIX2_3_xxx_0_4_4f.csv'
FILE_mix_4_old = 'MIX2_4_xxx_0_5_4f.csv'

FILE_mix_0_new = 'MIX2_0_new_0_0_1_4f.csv'
FILE_mix_1_new = 'MIX2_1_new_0_2_4f.csv'
FILE_mix_2_new = 'MIX2_2_new_0_3-S27_4f.csv'
FILE_mix_3_new = 'MIX2_3_new_0_4_4f.csv'
FILE_mix_4_new = 'MIX2_4_new_0_5_4f.csv'

def delete_repetition(FileName_this, FileName_last, FileName_new):
	with open('mapping.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		APs = [row for row in reader] 
		APlength = len(APs) - 1

	with open(FileName_this, 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		RSS = [row for row in reader]
		samples = int((len(RSS)-1) / APlength)
		"""print('length:',len(RSS))
								print(APlength)
								print(samples)
								print('samples:',samples)
								print('0,0:',RSS[0][0])
								print(float(RSS[2][14]))"""
	
	for s in range(samples,1,-1):
		for AP in range(APlength-1,-1,-1):
			if RSS[s*APlength-AP][14] != 'none' and RSS[(s-1)*APlength-AP][14] != 'none':
				
				if float(RSS[s*APlength-AP][14]) == float(RSS[(s-1)*APlength-AP][14]) and float(RSS[s*APlength-AP][15]) == float(RSS[(s-1)*APlength-AP][15]) and float(RSS[s*APlength-AP][16]) == float(RSS[(s-1)*APlength-AP][16]) or RSS[s*APlength-AP][1]=='surf' or RSS[s*APlength-AP][1]=='SURF':
					RSS[s*APlength-AP][6] = 'none'
					RSS[s*APlength-AP][7] = 'none'
					RSS[s*APlength-AP][8] = 'none'
					RSS[s*APlength-AP][9] = 'none'
					RSS[s*APlength-AP][10] = 'none'
					RSS[s*APlength-AP][11] = 'none'
					RSS[s*APlength-AP][12] = 'none'
					RSS[s*APlength-AP][13] = '-110'
					RSS[s*APlength-AP][14] = 'none'
					RSS[s*APlength-AP][15] = 'none'
					RSS[s*APlength-AP][16] = 'none'
					RSS[s*APlength-AP][18] = 'none'

	with open(FileName_last, 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		RSS_old = [row for row in reader]
		samples_old = int((len(RSS_old)-1) / APlength)

		#print(samples_old*APlength-(APlength-1))
		x =0
		for AP in range(APlength-1,-1,-1):
			#print(RSS[APlength-AP][0])
			if RSS[APlength-AP][16] != 'none' and RSS_old[(samples_old)*APlength-AP][14] != 'none':
				if float(RSS[APlength-AP][14]) == float(RSS_old[(samples_old)*APlength-AP][14]) and float(RSS[APlength-AP][15]) == float(RSS_old[(samples_old)*APlength-AP][15]) and float(RSS[APlength-AP][16]) == float(RSS_old[(samples_old)*APlength-AP][16]) or RSS[APlength-AP][1]=='surf' or RSS[APlength-AP][1]=='SURF':
					RSS[APlength-AP][6] = 'none'
					RSS[APlength-AP][7] = 'none'
					RSS[APlength-AP][8] = 'none'
					RSS[APlength-AP][9] = 'none'
					RSS[APlength-AP][10] = 'none'
					RSS[APlength-AP][11] = 'none'
					RSS[APlength-AP][12] = 'none'
					RSS[APlength-AP][13] = '-110'
					RSS[APlength-AP][14] = 'none'
					RSS[APlength-AP][15] = 'none'
					RSS[APlength-AP][16] = 'none'
					RSS[APlength-AP][18] = 'none'
		
		
	with open(FileName_new, 'w', newline='') as csvfile: 
		spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		
		for i in range(len(RSS)):
			data = ([
					RSS[i][0], RSS[i][1], RSS[i][2], RSS[i][3], RSS[i][4], RSS[i][5], RSS[i][6], RSS[i][7], RSS[i][8], RSS[i][9], RSS[i][10], RSS[i][11], RSS[i][12], RSS[i][13], RSS[i][14], RSS[i][15], RSS[i][16], RSS[i][17], RSS[i][18]
					])	 
			spamwriter.writerow(data)

def delete_repetition_fistFile(FileName_this, FileName_new):
	with open('mapping.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		APs = [row for row in reader] 
		APlength = len(APs) - 1

	with open(FileName_this, 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		RSS = [row for row in reader]
		samples = int((len(RSS)-1) / APlength)
		print('samples:',samples)
	print(samples*APlength+1)
	

	for s in range(samples,0,-1):
		for AP in range(APlength-1,-1,-1):
			if RSS[s*APlength-AP][14] == RSS[(s-1)*APlength-AP][14] and RSS[s*APlength-AP][15] == RSS[(s-1)*APlength-AP][15] and RSS[s*APlength-AP][16] == RSS[(s-1)*APlength-AP][16] or RSS[s*APlength-AP][1]=='surf' or RSS[s*APlength-AP][1]=='SURF':
				RSS[s*APlength-AP][6] = 'none'
				RSS[s*APlength-AP][7] = 'none'
				RSS[s*APlength-AP][8] = 'none'
				RSS[s*APlength-AP][9] = 'none'
				RSS[s*APlength-AP][10] = 'none'
				RSS[s*APlength-AP][11] = 'none'
				RSS[s*APlength-AP][12] = 'none'
				RSS[s*APlength-AP][13] = '-110'
				RSS[s*APlength-AP][14] = 'none'
				RSS[s*APlength-AP][15] = 'none'
				RSS[s*APlength-AP][16] = 'none'
				RSS[s*APlength-AP][18] = 'none'
	
	

	with open(FileName_new, 'w', newline='') as csvfile: 
		spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		
		for i in range(len(RSS)):
			data = ([
					RSS[i][0], RSS[i][1], RSS[i][2], RSS[i][3], RSS[i][4], RSS[i][5], RSS[i][6], RSS[i][7], RSS[i][8], RSS[i][9], RSS[i][10], RSS[i][11], RSS[i][12], RSS[i][13], RSS[i][14], RSS[i][15], RSS[i][16], RSS[i][17], RSS[i][18]
					])	 
			spamwriter.writerow(data)

if __name__ == "__main__":

	

	delete_repetition(FILE_dif_direc_old,FILE_15_old,FILE_dif_direc_new)
	delete_repetition(FILE_15_old,FILE_14_old,FILE_15_new)
	delete_repetition(FILE_14_old,FILE_13_old,FILE_14_new)
	delete_repetition(FILE_13_old,FILE_12_old,FILE_13_new)
	delete_repetition(FILE_12_old,FILE_11_old,FILE_12_new)
	delete_repetition(FILE_11_old,FILE_10_old,FILE_11_new)
	delete_repetition(FILE_10_old,FILE_9_old,FILE_10_new)
	delete_repetition(FILE_9_old,FILE_8_old,FILE_9_new)
	delete_repetition(FILE_8_old,FILE_7_old,FILE_8_new)
	delete_repetition(FILE_7_old,FILE_6_old,FILE_7_new)
	delete_repetition(FILE_6_old,FILE_5_old,FILE_6_new)
	delete_repetition(FILE_5_old,FILE_4_old,FILE_5_new)
	delete_repetition(FILE_4_old,FILE_3_old,FILE_4_new)
	delete_repetition(FILE_3_old,FILE_2_old,FILE_3_new)
	delete_repetition(FILE_2_old,FILE_1_old,FILE_2_new)
	delete_repetition(FILE_1_old,FILE_0_old,FILE_1_new)
	delete_repetition_fistFile(FILE_0_old,FILE_0_new)

	delete_repetition(FILE_mix_4_old,FILE_mix_3_old,FILE_mix_4_new)
	delete_repetition(FILE_mix_3_old,FILE_mix_2_old,FILE_mix_3_new)
	delete_repetition(FILE_mix_2_old,FILE_mix_1_old,FILE_mix_2_new)
	delete_repetition(FILE_mix_1_old,FILE_mix_0_old,FILE_mix_1_new)
	delete_repetition_fistFile(FILE_mix_0_old,FILE_mix_0_new)
	

'''
	with open('mapping.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		APs = [row for row in reader] 
		APlength = len(APs) - 1

	with open('15_xxx_0_5_4f.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		RSS = [row for row in reader]
		samples = int((len(RSS)-1) / APlength)
		print('samples:',samples)
	print(samples*APlength+1)
	

	for s in range(samples,0,-1):
		for AP in range(APlength-1,-1,-1):
			if RSS[s*APlength-AP][14] == RSS[(s-1)*APlength-AP][14] and RSS[s*APlength-AP][15] == RSS[(s-1)*APlength-AP][15] and RSS[s*APlength-AP][16] == RSS[(s-1)*APlength-AP][16]:
				RSS[s*APlength-AP][6] = 'none'
				RSS[s*APlength-AP][7] = 'none'
				RSS[s*APlength-AP][8] = 'none'
				RSS[s*APlength-AP][9] = 'none'
				RSS[s*APlength-AP][10] = 'none'
				RSS[s*APlength-AP][11] = 'none'
				RSS[s*APlength-AP][12] = 'none'
				RSS[s*APlength-AP][13] = '-110'
				RSS[s*APlength-AP][14] = 'none'
				RSS[s*APlength-AP][15] = 'none'
				RSS[s*APlength-AP][16] = 'none'
				RSS[s*APlength-AP][18] = 'none'
				
	with open(FILE_15_new, 'w', newline='') as csvfile: 
		spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		spamwriter.writerow(['BSSID', 'SSID','Building', 'Floor','Location_x', 'Location_y', 'Frequency','AccX','AccY','AccZ', 'ORIx','ORIy','ORIz','Level', 'GeoX','GeoY','GeoZ', 'Model','Time'])
		
		for i in range(len(RSS)):
			data = ([
					RSS[i][0], RSS[i][1], RSS[i][2], RSS[i][3], RSS[i][4], RSS[i][5], RSS[i][6], RSS[i][7], RSS[i][8], RSS[i][9], RSS[i][10], RSS[i][11], RSS[i][12], RSS[i][13], RSS[i][14], RSS[i][15], RSS[i][16], RSS[i][17], RSS[i][18]
					])	 
			spamwriter.writerow(data)
'''
