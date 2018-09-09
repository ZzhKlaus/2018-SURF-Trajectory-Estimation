import csv
import pandas as pd
import numpy as np
PYTHONIOENCODING="UTF-8"  #set the utf-8 encode mode


FileName0 = '0_new_0_1_5E_39.csv'
FileName0_after = 'dataset_0.csv'

FileName1 = '1_new_0_1_5SW.csv'
FileName1_after = 'dataset_1.csv'

FileName2 = '2_new_0_2_0_3_5f.csv'
FileName2_after = 'dataset_2.csv'

FileName3 = '3_new_0_4_5f.csv'
FileName3_after = 'dataset_3.csv'

FileName4 = '4_new_0_5_5f.csv'
FileName4_after = 'dataset_4.csv'

FileName5 = '5_new_0_6_5f.csv'
FileName5_after = 'dataset_5.csv'

FileName6 = '6_new_0_7_5f.csv'
FileName6_after = 'dataset_6.csv'

FileName7 = '7_new_0_8_5f.csv'
FileName7_after = 'dataset_7.csv'

FileName8 = '8_new_0_9_5f.csv'
FileName8_after = 'dataset_8.csv'

FileName9 = '9_new_0_10_5f.csv'
FileName9_after = 'dataset_9.csv'

FileName10 = '10_new_0_11_5f.csv'
FileName10_after = 'dataset_10.csv'

FileName11 = '11_new_0_12_5f.csv'
FileName11_after = 'dataset_11.csv'

FileName12 = '12_new_0_0_1_4f.csv'
FileName12_after = 'dataset_12.csv'

FileName13 = '13_new_0_2_4f.csv'
FileName13_after = 'dataset_13.csv'

FileName14 = '14_new_0_3_4_5f.csv'
FileName14_after = 'dataset_14.csv'

FileName15 = '15_new_0_5_4f.csv'
FileName15_after = 'dataset_15.csv'

datasetOri = 'new_diff-direction-0_5_4f.csv'
datasetOri_after = 'datasetOri.csv'


datasetMix0 = 'MIX2_0_new_0_0_1_4f.csv'
datasetMix0_after = 'datasetMix_0.csv'

datasetMix1 = 'MIX2_1_new_0_2_4f.csv'
datasetMix1_after = 'datasetMix_1.csv'

datasetMix2 = 'MIX2_2_new_0_3-S27_4f.csv'
datasetMix2_after = 'datasetMix_2.csv'

datasetMix3 = 'MIX2_3_new_0_4_4f.csv'
datasetMix3_after = 'datasetMix_3.csv'

datasetMix4 = 'MIX2_4_new_0_5_4f.csv'
datasetMix4_after = 'datasetMix_4.csv'

dataset0 = pd.read_csv('0_new_0_1_5E_39.csv', header = 0)
dataset1 = pd.read_csv('1_new_0_1_5SW.csv', header = 0)
dataset2 = pd.read_csv('2_new_0_2_0_3_5f.csv', header = 0)
dataset3 = pd.read_csv('3_new_0_4_5f.csv', header = 0)
dataset4 = pd.read_csv('4_new_0_5_5f.csv', header = 0)
dataset5 = pd.read_csv('5_new_0_6_5f.csv', header = 0)
dataset6 = pd.read_csv('6_new_0_7_5f.csv', header = 0)
dataset7 = pd.read_csv('7_new_0_8_5f.csv', header = 0)
dataset8 = pd.read_csv('8_new_0_9_5f.csv', header = 0)
dataset9 = pd.read_csv('9_new_0_10_5f.csv', header = 0)
dataset10 = pd.read_csv('10_new_0_11_5f.csv', header = 0)
dataset11 = pd.read_csv('11_new_0_12_5f.csv', header = 0)
dataset12 = pd.read_csv('12_new_0_0_1_4f.csv', header = 0)
dataset13 = pd.read_csv('13_new_0_2_4f.csv', header = 0)
dataset14 = pd.read_csv('14_new_0_3_4_5f.csv', header = 0)
dataset15 = pd.read_csv('15_new_0_5_4f.csv', header = 0)


'''
with open('MIX2_0_new_0_0_1_4f.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		RSS = [row for row in reader]
		print(len(RSS))
'''
'''
with open('2_new_0_2_0_3_5f.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		RSS = [row for row in reader]
		print(len(RSS))
		with open('2_new_0_2_0_3_5f_new.csv', 'w', newline='') as csvfile: 
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			#edition3					
			for row in range(0,188857):
				data = ([
					RSS[row][0], RSS[row][1], RSS[row][2], RSS[row][3], RSS[row][4], RSS[row][5], RSS[row][6], RSS[row][7], RSS[row][8], RSS[row][9], RSS[row][10], RSS[row][11], RSS[row][12], RSS[row][13], RSS[row][14], RSS[row][15], RSS[row][16], RSS[row][17], RSS[row][18]
					])
				spamwriter.writerow(data)
'''

FileName_Total = 'Total.csv'

def concataneate1Files(FileName_Original,FileName_new = FileName_Total):
	
	x = []
	with open(FileName_Original, 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		RSS = [row for row in reader]
		RSS = np.asarray(RSS)
		with open(FileName_new, 'a', newline='') as csvfile: 
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)				

			for row in range(len(RSS)):
				for column in range(RSS.shape[1]):
					x.append(RSS[row][column])
					#x = ([
						#	RSS[row][0], RSS[row][1], RSS[row][2], RSS[row][3], RSS[row][4], RSS[row][5], RSS[row][6], RSS[row][7], RSS[row][8], RSS[row][9], RSS[row][10], RSS[row][11], RSS[row][12], RSS[row][13], RSS[row][14], RSS[row][15], RSS[row][16], RSS[row][17], RSS[row][18]
						#])
			
				spamwriter.writerow(x)
				x = []

def concataneate2Files(FileName_Original,FileName_addition,FileName_new = FileName_Total):
	
	x = []
	with open(FileName_Original, 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		RSS_Ori = [row for row in reader]

		with open(FileName_addition, 'r', newline='') as csvfile: 
			reader = csv.reader(csvfile)
			RSS_Add = [row for row in reader]

			with open(FileName_new, 'a', newline='') as csvfile: 
				spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)				
				RSS = np.vstack((RSS_Ori,RSS_Add))
				print(RSS.shape[1])
				print('type:',type(RSS))

				for row in range(len(RSS)):
					for column in range(RSS.shape[1]):
						x.append(RSS[row][column])
						#x = ([
							#	RSS[row][0], RSS[row][1], RSS[row][2], RSS[row][3], RSS[row][4], RSS[row][5], RSS[row][6], RSS[row][7], RSS[row][8], RSS[row][9], RSS[row][10], RSS[row][11], RSS[row][12], RSS[row][13], RSS[row][14], RSS[row][15], RSS[row][16], RSS[row][17], RSS[row][18]
							#])
				
					spamwriter.writerow(x)
					x = []
												

	
'''
	datasetOri = pd.read_csv(FileName_Original, header = 0)
	datasetAdd = pd.read_csv(FileName_addition, header = 0)
	frames = np.vstack((datasetOri,datasetAdd))
	frames = pd.DataFrame(frames)
	#Ori = pd.DataFrame(datasetOri)
	#Add = pd.DataFrame(datasetAdd)
	new = pd.concat([frames],ignore_index=True,sort=False)
	
	print(new.shape)
	new.to_csv(FileName_new, header = 0)
'''	
'''
concataneate2Files(FileName0_after,FileName1_after)
concataneate2Files(FileName2_after,FileName3_after)
concataneate2Files(FileName4_after,FileName5_after)
concataneate2Files(FileName6_after,FileName7_after)
concataneate2Files(FileName8_after,FileName9_after)
concataneate2Files(FileName10_after,FileName11_after)
concataneate2Files(FileName12_after,FileName13_after)
concataneate2Files(FileName14_after,FileName15_after)
'''

'''
concataneate2Files(datasetMix0_after,datasetMix1_after)
concataneate2Files(datasetMix2_after,datasetMix3_after)
concataneate1Files(datasetMix4_after)
'''
concataneate1Files(datasetMix4_after)


def contructDataset(FileName,FileName_after, isHead):
	dataset = pd.read_csv(FileName, header = 0)
	print('length:',len(dataset))
	RSS = np.asarray(dataset.ix[:, 13])
	if(isHead ==1):
		head = []
		for i in range(516):
			if(i<10):
				head.append('WAP00'+str(i))
			elif(i<100):
				head.append('WAP0'+str(i))
			else:
				head.append('WAP'+str(i))
		head.append('Loc_x')
		head.append('Loc_y')
		head.append('Floor')
		head.append('Building')
		head.append('Model')
		head.append('TimeStamp')
		head.append('GeoX')
		head.append('GeoY')
		head.append('GeoZ')
		head.append('AccX')
		head.append('AccY')
		head.append('AccZ')
		head.append('OriX')
		head.append('GriY')
		head.append('GriZ')

	inertia = []
	GeoX = 0
	GeoY = 0
	GeoZ = 0
	AccX = 0
	AccY = 0
	AccZ = 0
	OriX = 0
	OriY = 0
	OriZ = 0
	APnumber = 0
	for i in range(len(RSS)):
		if dataset.ix[i,14] != 'none':
			APnumber += 1
			GeoX += float(dataset.ix[i,14])
			GeoY += float(dataset.ix[i,15])
			GeoZ += float(dataset.ix[i,16])
			AccX += float(dataset.ix[i,7])
			AccY += float(dataset.ix[i,8])
			AccZ += float(dataset.ix[i,9])
			OriX += float(dataset.ix[i,10])
			OriY += float(dataset.ix[i,11])
			OriZ += float(dataset.ix[i,12])
		if (i+1) % 516 == 0:
			if APnumber != 0:
				GeoX = GeoX/APnumber
				GeoY = GeoY/APnumber
				GeoZ = GeoZ/APnumber
				AccX = AccX/APnumber
				AccY = AccY/APnumber
				AccZ = AccZ/APnumber
				OriX = OriX/APnumber
				OriY = OriY/APnumber
				OriZ = OriZ/APnumber
			inertia.append(GeoX)
			inertia.append(GeoY)
			inertia.append(GeoZ)
			inertia.append(AccX)
			inertia.append(AccY)
			inertia.append(AccZ)
			inertia.append(OriX)
			inertia.append(OriY)
			inertia.append(OriZ)
			GeoX = 0
			GeoY = 0
			GeoZ = 0
			AccX = 0
			AccY = 0
			AccZ = 0
			OriX = 0
			OriY = 0
			OriZ = 0
			APnumber = 0

	labels = []
	for i in range(len(RSS)):
		if (i+1) % 516 == 0:
			labels.append(dataset.ix[i,4])
			labels.append(dataset.ix[i,5])
			labels.append(dataset.ix[i,3])
			labels.append(dataset.ix[i,2])
			labels.append(dataset.ix[i,17])
			labels.append(dataset.ix[i,18])

	RSS = RSS.reshape([int(RSS.shape[0]/516), 516])
	labels = np.asarray(labels).reshape([int(len(labels)/6), 6])
	inertia = np.asarray(inertia).reshape([int(len(inertia)/9), 9])

	dataset = np.hstack((np.hstack((RSS, labels)),inertia))
	print(dataset.shape)

	new = pd.DataFrame(dataset)

	if(isHead == 1):
		new.to_csv(FileName_after, header = head[:])
	elif(isHead == 0):
		new.to_csv(FileName_after, header = 0)

'''
contructDataset(FileName0,FileName0_after,isHead = 1)
contructDataset(FileName1,FileName1_after,isHead = 0)
contructDataset(FileName2,FileName2_after,isHead = 0)
contructDataset(FileName3,FileName3_after,isHead = 0)
contructDataset(FileName4,FileName4_after,isHead = 0)
contructDataset(FileName5,FileName5_after,isHead = 0)
contructDataset(FileName6,FileName6_after,isHead = 0)
contructDataset(FileName7,FileName7_after,isHead = 0)
contructDataset(FileName8,FileName8_after,isHead = 0)
contructDataset(FileName9,FileName9_after,isHead = 0)
contructDataset(FileName10,FileName10_after,isHead = 0)
contructDataset(FileName11,FileName11_after,isHead = 0)
contructDataset(FileName12,FileName12_after,isHead = 0)
contructDataset(FileName13,FileName13_after,isHead = 0)
contructDataset(FileName14,FileName14_after,isHead = 0)
contructDataset(FileName15,FileName15_after,isHead = 0)
contructDataset(datasetOri,datasetOri_after,isHead = 1)
contructDataset(datasetMix0,datasetMix0_after,isHead = 1)
contructDataset(datasetMix1,datasetMix1_after,isHead = 0)
contructDataset(datasetMix2,datasetMix2_after,isHead = 0)
contructDataset(datasetMix3,datasetMix3_after,isHead = 0)
contructDataset(datasetMix4,datasetMix4_after,isHead = 0)
'''
#print(dataset[:, -1])




'''with open('APs.csv', 'w', newline='') as csvfile: 
		if not os.path.getsize('./APs.csv'): 
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow([ 'BSSID','SSID','Building', 'Floor','Location_x', 'Location_y','Frequency','AccX','AccY','AccZ','ORIx','ORIy','ORIz','Level','GeoX','GeoY','GeoZ'])

'''