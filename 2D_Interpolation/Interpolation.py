'''
By Zhenghang(Klaus) Zhong

Assuming in a small indoor space, relationship between space and geomagnetic field intensity is linear

Interpolation method is used for two purpose:

1. Imporve the precision of geomagnetic map

2. Work with random waypoint model, with generated walk trace
get the points' magnetic field intensity in x,y,z coordination

For 4th floor detection area, the size is 51 points X 6 points, 
we improve the precision by 6 times, which mean to interpolate 
5 points between two raw points. 

For 5th floor detection area, the size is 51 points X 13 points, 
do the same way as 4th floor.
'''

# coding: utf-8
import csv
import os
import re #get number in string
import math
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.interpolate import SmoothBivariateSpline as SBS
from scipy.interpolate import CloughTocher2DInterpolator
PYTHONIOENCODING="UTF-8"

filename = '5N_all.csv'

Loc_x = pd.read_csv(filepath_or_buffer = filename, sep = ',')["Loc_x"].values.astype('int')
Loc_y = pd.read_csv(filepath_or_buffer = filename, sep = ',')["Loc_y"].values.astype('int')
Mag_x = pd.read_csv(filepath_or_buffer = filename, sep = ',')["GeoX"].values.astype('float')
Mag_y = pd.read_csv(filepath_or_buffer = filename, sep = ',')["GeoY"].values.astype('float')
Mag_z = pd.read_csv(filepath_or_buffer = filename, sep = ',')["GeoZ"].values.astype('float')
#Add one axis, transfer to 2-D data
Loc_x = Loc_x[:,np.newaxis]
Loc_y = Loc_y[:,np.newaxis]

'''
#Old way of doing obove work

with open('5N_all.csv', 'r', newline='') as csvfile: 
	reader = csv.reader(csvfile)
	frames = [row for row in reader]
	frames = np.array(frames)

Loc_x = frames[1:,518].astype('int')
Loc_x = Loc_x[:,np.newaxis]
Loc_y = frames[1:,519].astype('int')
Loc_y = Loc_y[:,np.newaxis]
Mag_x = frames[1:,524].astype('float')
Mag_y = frames[1:,525].astype('float')
Mag_z = frames[1:,526].astype('float')
'''
Loc = np.concatenate((Loc_x,Loc_y),axis = 1)
print(Loc.shape)

# The shape of map is 50 X 12
grid_x, grid_y = np.mgrid[0:50:0.01, 0:12:0.01] 
print(grid_x.shape)

#Get total magnetic field intensity
Mag = []
for i in range(len(Mag_x)):
	Mag.append( math.sqrt(Mag_x[i]*Mag_x[i]+Mag_y[i]*Mag_y[i]+Mag_z[i]*Mag_z[i]))

#Show geomagnetic field with graph
G_y_li = griddata(Loc, Mag, (grid_x, grid_y), method='linear')
plt.plot()
plt.imshow(G_y_li.T, extent=(0,50,0,12), origin='lower')
plt.show()

#Clough-Tocher 2-D Interpolation 
C_x = CloughTocher2DInterpolator(Loc, Mag_x,  tol=1e-6)
C_y = CloughTocher2DInterpolator(Loc, Mag_y,  tol=1e-6)
C_z = CloughTocher2DInterpolator(Loc, Mag_z,  tol=1e-6)

#For forth floor, we collect data in map of 51 X 6 points
def interpolateMap():
	with open('newMap_4S.csv', 'w', newline='') as csvfile: 
		spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		spamwriter.writerow(['Loc_x','Loc_y','Floor','Mag_x','Mag_y','Mag_z'])

		for x in range((51-1)*6+1):		
			for y in range((6-1)*6+1):
				Loc_x = x
				Loc_y = y
				Mag_x = C_x(Loc_x/6,Loc_y/6).__float__()
				Mag_y = C_y(Loc_x/6,Loc_y/6).__float__()
				Mag_z = C_z(Loc_x/6,Loc_y/6).__float__()				
				row = ([
					Loc_x,Loc_y,'4S',Mag_x,Mag_y,Mag_z
					])
				spamwriter.writerow(row)

#Interpolate magnetic values in terms of waypoint traces
#The precision improve 6 times (interpolate 5 points between 2 detected points)
def interpolateValues():	
	with open('waypoint_trace.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		traces = [row for row in reader]
		traces = np.array(traces)
		#traces[1:][:].astype('float')
	
		with open('waypoint_trace_new.csv', 'w', newline='') as csvfile: 
			spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
			spamwriter.writerow(['Loc_x','Loc_y','Floor','Mag_x','Mag_y','Mag_z'])

			for x in range(1,len(traces)):
				Loc_x = traces[x][0].astype('float')
				Loc_y = traces[x][1].astype('float')
				Mag_x = C_x(Loc_x/6,Loc_y/6).__float__()
				Mag_y = C_y(Loc_x/6,Loc_y/6).__float__()
				Mag_z = C_z(Loc_x/6,Loc_y/6).__float__()
				row = ([
					traces[x][0],traces[x][1],'4',Mag_x,Mag_y,Mag_z
					])
				spamwriter.writerow(row)

#For 5th floor
def generate2Dmagnetic_5():
	with open('newMap_5N.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		frames = [row for row in reader]
		frames = np.array(frames)
	#frames = pd.DataFrame(frames)
	#frames = frames.reshape([len(frames), len(frames[1])])
	
		Mag_x = frames[1:,3].astype('float')
		Mag_y = frames[1:,4].astype('float')
		Mag_z = frames[1:,5].astype('float')
		
		Mag_total = []
		for i in range(len(Mag_x)):
			Mag_total.append(math.sqrt(Mag_x[i]*Mag_x[i]+Mag_y[i]*Mag_y[i]+Mag_z[i]*Mag_z[i]))
		#print(Mag_total)
		Mag_total = np.array(Mag_total).reshape(301,73)
		Mag_total = Mag_total.transpose()
		print(Mag_total.shape)
		print(Mag_total)
		x = np.arange(0,301,1)
		y = np.arange(0,73,1)
		X,Y = np.meshgrid(x,y)
		print(X)
		print(Y)
		#plt.contourf(X,Y,Mag_total)
		cset = plt.contourf(X,Y,Mag_total,7,alpha=0.99,cmap=plt.cm.hot)
		contour = plt.contour(X,Y,Mag_total,10,colors = 'k')
		#plt.clabel(contour,fontsize=10,colors = ('k'))
		plt.colorbar(cset,orientation='horizontal')
		plt.show()

#For 4th floor
def generate2Dmagnetic_4():
	with open('newMap_4N.csv', 'r', newline='') as csvfile: 
		reader = csv.reader(csvfile)
		frames = [row for row in reader]
		frames = np.array(frames)
	#frames = pd.DataFrame(frames)
	#frames = frames.reshape([len(frames), len(frames[1])])
	
		Mag_x = frames[1:,3].astype('float')
		Mag_y = frames[1:,4].astype('float')
		Mag_z = frames[1:,5].astype('float')
		

		Mag_total = []
		for i in range(len(Mag_x)):
			Mag_total.append(math.sqrt(Mag_x[i]*Mag_x[i]+Mag_y[i]*Mag_y[i]+Mag_z[i]*Mag_z[i]))
		#print(Mag_total)
		Mag_total = np.array(Mag_total).reshape(301,31)
		Mag_total = Mag_total.transpose()
		print(Mag_total.shape)
		print(Mag_total)
		x = np.arange(0,301,1)
		y = np.arange(0,31,1)
		X,Y = np.meshgrid(x,y)
		print(X)
		print(Y)
		#plt.contourf(X,Y,Mag_total)
		cset = plt.contourf(X,Y,Mag_total,7,alpha=0.99,cmap=plt.cm.hot)
		contour = plt.contour(X,Y,Mag_total,10,colors = 'k')
		#plt.clabel(contour,fontsize=10,colors = ('k'))
		plt.colorbar(cset,orientation='horizontal')
		plt.show()


#Call one function at a time

#interpolateMap()
#interpolateValues()
#generate2Dmagnetic_4()
#generate2Dmagnetic_5()
