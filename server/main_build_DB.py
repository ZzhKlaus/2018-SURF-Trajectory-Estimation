'''
By Zhenghang(Klaus) Zhong

############################################################################################
It's a light server based on FLASK micro framework,

1.Requirements: Python 3, Flask and relevant packages

2. How does this work? 
	(1) Firstly, modify the host IP address of your own environment.
	(2) Then run this python file,
		A temporary file called 'tempList.csv' will be initialized with default data 
		(e.g. for signal level RSS it would be -110, magnetic field value would be none)
		with order according to the unchanged file 'APs.csv' (to store the AP info in a defined order)
		
		Each time one complete info of AP arrival, (assume there are 60 APs that is detected once, then 
		the transmission would be repeated 60 times and one symbol called "Done" would be set to '1' for
		last time, which means info of one scan has all been sent), the 'tempList.csv' would be refreshed 
		with one line of AP's info. After 60 times (AP number), the function 'refreshCSV()' would be called.

		Then scan info of once would be be copied from 'tempList.csv' and be added in 'xxx.csv'(which stores 
		all info that is similar to database) and be refreshed in 'oneTime.csv' (for check last time's scan info). 

		Finally, refresh 'tempList.csv' with default value for next time's transmission.
############################################################################################
'''

# coding: utf-8
from flask import Flask, request
from app import db, models
import csv
import os #to get current path
import importlib

from model import *

#algorithm part 
import pandas as pdb
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

PYTHONIOENCODING="UTF-8"  #set the utf-8 encode mode
  
	
# create the application object
app = Flask(__name__)

#edition
# Write all info in DB into a csv file, without SSID stored, encode mode is UTF-8 (as some SSID contains chinese characters)
#edition
def addAllCSV():    #whole database	
	with open('APs.csv', 'w', newline='') as csvfile: 
		if not os.path.getsize('./APs.csv'): 
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow([ 'BSSID','SSID','Building', 'Floor','Location_x', 'Location_y','Frequency','AccX','AccY','AccZ','ORIx','ORIy','ORIz','Level','GeoX','GeoY','GeoZ'])
		
		users = models.User.query.all()
		
		for u in users:
			data = ([u.BSSID, u.SSID, u.Buidling, u.Floor, u.Location_x, u.Location_y, u.Frequency, u.AccX, u.AccY, u.AccZ, u.ORIx, u.ORIy, u.ORIz, u.Level, u.GeoX, u.GeoY, u.GeoZ])
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow(data)
			
#add one time's scanner result			
def addCSV(BSSID, SSID, Building, Floor, Location_x, Location_y, Frequency, AccX, AccY, AccZ, ORIx, ORIy, ORIz, Level, GeoX, GeoY, GeoZ, Model, Time):    
	with open('userinput.csv', 'a', newline='') as csvfile: 		
		if not os.path.getsize('./userinput.csv'): 
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow(['BSSID', 'SSID','Building', 'Floor','Location_x', 'Location_y', 'Frequency','AccX','AccY', 'AccZ','ORIx','ORIy','ORIz','Level', 'GeoX','GeoY','GeoZ', 'Model','Time'])
		
		data = ([
		BSSID, SSID, Building, Floor, Location_x, Location_y,  Frequency, AccX, AccY, AccZ, ORIx, ORIy, ORIz, Level, GeoX, GeoY, GeoZ, Model, Time
		])

		spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		spamwriter.writerow(data)


def initializeTempList():
	with open('mapping.csv', 'r', newline='') as csvfile:  
		reader = csv.reader(csvfile)
		APs = [row[0] for row in reader]
		APlength = len(APs)
		lists = [[0 for col in range(19)] for row in range(APlength)]
		row = 0
		for AP in APs:
			lists[row][0] = AP
			lists[row][1] = 'none'
			lists[row][2] = 'none'
			lists[row][3] = 'none'
			lists[row][4] = 'none'
			lists[row][5] = 'none'
			lists[row][6] = 'none'
			lists[row][7] = 'none'
			lists[row][8] = 'none'
			lists[row][9] = 'none'
			lists[row][10] = 'none'
			lists[row][11] = 'none'
			lists[row][13] = 'none'
			lists[row][14] = '-110'
			lists[row][15] = 'none'
			lists[row][16] = 'none'
			lists[row][17] = 'none'
			lists[row][18] = 'none'
			row += 1

	with open('tempList.csv', 'w', newline='') as csvfile: 
		spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		spamwriter.writerow(['BSSID', 'SSID','Building', 'Floor','Location_x', 'Location_y', 'Frequency','AccX','AccY','AccZ', 'ORIx','ORIy','ORIz','Level', 'GeoX','GeoY','GeoZ', 'Model','Time'])
		for i in range(0,517): 
			data = ([
			lists[i][0], lists[i][1], lists[i][2], lists[i][3], lists[i][4], lists[i][5], lists[i][6], lists[i][7], lists[i][8], lists[i][9], lists[i][10], lists[i][11], lists[i][12], lists[i][13], lists[i][14], lists[i][15], lists[i][16], lists[i][17], lists[i][18]
			])
			spamwriter.writerow(data)
		
#Check if the input AP's BSSID is in the mapping.csv, which contains 200 APs
def checkAP(list, AP):
	row = 0
	
	for row in range(0,517):
		if AP == list[row][0]:
			return row      
	return 'none'           

def tempList(BSSID,SSID, Building, Floor, Location_x, Location_y, Frequency, AccX, AccY, AccZ, ORIx, ORIy, ORIz, Level, GeoX, GeoY, GeoZ, Model, Time): 
	with open('tempList.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		RSS = [row for row in reader]
		#print(RSS,RSS[0][0])
		for row in range(1,517):        
			if  RSS[row][0] == BSSID:
				RSS[row][1] = SSID
				RSS[row][2] = Building
				RSS[row][3] = Floor
				RSS[row][4] = Location_x
				RSS[row][5] = Location_y
				RSS[row][6] = Frequency
				RSS[row][7] = AccX
				RSS[row][8] = AccY
				RSS[row][9] = AccZ
				RSS[row][10] = ORIx
				RSS[row][11] = ORIy
				RSS[row][12] = ORIz
				RSS[row][13] = Level
				RSS[row][14] = GeoX
				RSS[row][15] = GeoY
				RSS[row][16] = GeoZ
				RSS[row][17] = Model
				RSS[row][18] = Time

				with open('tempList.csv', 'w', newline='') as csvfile: 
					spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
					spamwriter.writerow(['BSSID', 'SSID', 'Building','Floor','Location_x','Location_y', 'Frequency','AccX','AccY','AccZ', 'ORIx','ORIy','ORIz', 'Level', 'GeoX','GeoY','GeoZ', 'Model', 'Time'])             
					for i in range(1,517):
						data = ([
						RSS[i][0], RSS[i][1], RSS[i][2], RSS[i][3], RSS[i][4], RSS[i][5], RSS[i][6], RSS[i][7], RSS[i][8], RSS[i][9], RSS[i][10], RSS[i][11], RSS[i][12], RSS[i][13], RSS[i][14], RSS[i][15], RSS[i][16], RSS[i][17], RSS[i][18]
						])					
						spamwriter.writerow(data)
				break
	
def isEmpty():
	with open('xxx.csv', 'a+', newline='') as csvfile:  #Check is tempList is empty	
		if not os.path.getsize('./xxx.csv'):        #file not established
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow(['BSSID','SSID','Building', 'Floor','Location_x','Location_y','Frequency','AccX','AccY','AccZ','ORIx','ORIy','ORIz', 'Level', 'GeoX','GeoY','GeoZ', 'Model', 'Time'])
	
	with open('mapping.csv', 'r', newline='') as csvfile:  
		reader = csv.reader(csvfile)
		APs = [row[0] for row in reader]
		APlength = len(APs)
		lists = [[0 for col in range(19)] for row in range(APlength)]
		row = 0
		for AP in APs:
			lists[row][0] = AP
			lists[row][1] = 'none'
			lists[row][2] = 'none'
			lists[row][3] = 'none'
			lists[row][4] = 'none'
			lists[row][5] = 'none'
			lists[row][6] = 'none'
			lists[row][7] = 'none'
			lists[row][8] = 'none'
			lists[row][9] = 'none'
			lists[row][10] = 'none'
			lists[row][11] = 'none'
			lists[row][12] = 'none'
			lists[row][13] = '-110'
			lists[row][14] = 'none'
			lists[row][15] = 'none'
			lists[row][16] = 'none'
			lists[row][17] = 'none'
			lists[row][18] = 'none'
			row += 1
	#edition2
	with open('tempList.csv', 'a+', newline='') as csvfile:     #Check is tempList is empty	
		if not os.path.getsize('./tempList.csv'):       #file is empty
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow(['BSSID','SSID','Building','Floor','Location_x','Location_y', 'Frequency','AccX','AccY','AccZ','ORIx','ORIy','ORIz','Level', 'GeoX','GeoY','GeoZ', 'Model', 'Time'])
			for i in range(1,517): 
				data = ([
				 lists[i][0], lists[i][1], lists[i][2], lists[i][3], lists[i][4], lists[i][5], lists[i][6], lists[i][7], lists[i][8], lists[i][9], lists[i][10], lists[i][11], lists[i][12], lists[i][13], lists[i][14], lists[i][15], lists[i][16], lists[i][17], lists[i][18]
				])
				print(i)
				spamwriter.writerow(data)

def refreshCSV(Building, Floor, Location_x, Location_y, Model):
	with open('tempList.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		RSS = [row for row in reader]
		
		with open('tempList.csv', 'w', newline='') as csvfile: 
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow(['BSSID','SSID','Building', 'Floor','Location_x', 'Location_y', 'Frequency','AccX','AccY','AccZ','ORIx','ORIy','ORIz','Level', 'GeoX', 'GeoY', 'GeoZ', 'Model', 'Time'])
			for row in range(1,517):
				RSS[row][2] = Building
				RSS[row][3] = Floor
				RSS[row][4] = Location_x
				RSS[row][5] = Location_y
				RSS[row][17] = Model
				x = ([
					RSS[row][0], RSS[row][1], RSS[row][2], RSS[row][3], RSS[row][4], RSS[row][5], RSS[row][6], RSS[row][7], RSS[row][8], RSS[row][9], RSS[row][10], RSS[row][11], RSS[row][12], RSS[row][13], RSS[row][14], RSS[row][15], RSS[row][16], RSS[row][17], RSS[row][18]
					])
				
				spamwriter.writerow(x)
	            
		with open('xxx.csv', 'a', newline='') as csvfile: 
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			#edition3					
			for row in range(1,517):
				data = ([
					RSS[row][0], RSS[row][1], RSS[row][2], RSS[row][3], RSS[row][4], RSS[row][5], RSS[row][6], RSS[row][7], RSS[row][8], RSS[row][9], RSS[row][10], RSS[row][11], RSS[row][12], RSS[row][13], RSS[row][14], RSS[row][15], RSS[row][16], RSS[row][17], RSS[row][18]
					])
				spamwriter.writerow(data)
		
		with open('oneTime.csv', 'a', newline='') as csvfile: 			
			if not os.path.getsize('./oneTime.csv'):        #file is empty
				spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
				spamwriter.writerow(['BSSID','SSID','Building','Floor','Location_x', 'Location_y','Frequency','AccX','AccY','AccZ','ORIx','ORIy','ORIz','Level', 'GeoX','GeoY', 'GeoZ', 'Model', 'Time'])    
			#edition4
			for i in range(1,517):
				data = ([
				RSS[i][0], RSS[i][1], RSS[i][2], RSS[i][3], RSS[i][4], RSS[i][5], RSS[i][6], RSS[i][7], RSS[i][8], RSS[row][9], RSS[row][10], RSS[row][11], RSS[i][12], RSS[row][13], RSS[row][14], RSS[row][15], RSS[row][16], RSS[row][17], RSS[row][18]
				])
				spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
				spamwriter.writerow(data)
				
	
@app.route('/', methods=['POST'])
def post():
#isEmpty()
#edition5
	isEmpty()

	BSSID = request.form['BSSID']
	Building = request.form['Building']
	Floor = request.form['Floor']
	Location_x = request.form['Location_x']
	Location_y = request.form['Location_y']
	Frequency = request.form['Frequency']
	Level = request.form['Level']
	AccX = request.form['AccX']
	AccY = request.form['AccY']
	GeoX = request.form['GeoX']
	GeoY = request.form['GeoY']
	GeoZ = request.form['GeoZ']
	Model = request.form['Model']
	Time = request.form['Time']
	SSID = request.form['SSID']
	AccX = request.form['AccX']
	AccY = request.form['AccY']
	AccZ = request.form['AccZ']
	ORIx = request.form['ORIx']
	ORIy = request.form['ORIy']
	ORIz = request.form['ORIz']
	Done = request.form['Done']

	#addCSV(BSSID, SSID, Building, Floor, Location_x, Location_y, Frequency, AccX, AccY, AccZ, ORIx, ORIy, ORIz, Level, GeoX, GeoY, GeoZ, Model, Time)
	tempList(BSSID, SSID,Building, Floor, Location_x, Location_y, Frequency, AccX, AccY, AccZ, ORIx, ORIy, ORIz, Level, GeoX, GeoY, GeoZ, Model, Time)
	#refreshCSV(SSID,Building, Floor, Location_x, Location_y, Frequency, AccX, AccY, AccZ, ORIx, ORIy, ORIz, Level, GeoX, GeoY, GeoZ, Model, Time)

	#addAPs(BSSID, Building, Location_x, Location_y, Frequency, AccX, AccY, AccZ, ORIx, ORIy, ORIz, Level, GeoX, GeoY, GeoZ, Model, Time)
	#addCSV(BSSID, Building, Location_x, Location_y, Frequency, AccX, AccY, AccZ, ORIx, ORIy, ORIz, Level, GeoX, GeoY, GeoZ, Model, Time)


	#addAPs(list)
	#addAllCSV()
	
	#addAPs(Building, Room, Location_x, Location_y, SSID,BSSID, Frequency, Level)
	
	#addCSV(Building, Room, Location_x, Location_y, BSSID, Frequency, Level)
	
	if Done == '1':
		refreshCSV(Building, Floor, Location_x, Location_y, Model)
		initializeTempList()
		print('1')
	else:
		print('0')


	return 'OK.'

if __name__ == "__main__":
	#Use local host IP for local server
	#Or IPV4 address

	#app.run(host='192.168.xxx.xxx', debug=True)
	app.run(host='192.168.xxx.xxx', debug=True)

'''
#Add RSS info into database whose name is app.db
def addAPs(list):
	for row in range(0,517):
		u = models.User(BSSID = list[row][0], SSID = list[row][1], Building = list[row][2], Floor = list[row][3], Location_x = list[row][4], Location_y = list[row][5], Frequency = list[row][6], AccX = list[row][7], AccY = list[row][8], AccZ = list[row][9], ORIx = list[row][10], ORIy = list[row][11], ORIz = list[row][12], Level = list[row][13], GeoX=list[row][14], GeoY=list[row][15], GeoZ=list[row][16])
		db.session.add(u)
	db.session.commit()

#Show all RSS info from database	
def showAPs(num):	
	ap = models.User.query.get(num)
	print(ap.BSSID, ap.SSID, ap.Building, ap.Floor,ap.Location_x, ap.Location_y, ap.Frequency, ap.AccX, ap.AccY, ap.AccZ, ap.ORIx, ap.ORIy, ap.ORIz, ap.Level, ap.GeoX, ap.GeoY, ap.GeoZ)
	
			
def deleteDB():
	users = models.User.query.all()
	
	for u in users:
		db.session.delete(u)
		
	db.session.commit()
'''