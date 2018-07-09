import threading
import socket
import numpy as np
import newread as nrt
import matplotlib.pyplot as plt
import filtertd as flt
import time
import timeit
import pygame.surfarray as surfarray
import pygame
from pygame.locals import *
import cv2
import integral_method as intm
import cnnobject as cno
import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, initializers
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.models import model_from_json
import random
import math as mt
import newfiltertd as nflt
import matplotlib.patches as patches
import realtimelaserdet as rtl
from copy import copy
# import UR10
# from ur10_simulation import ur10_simulator
# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.uic import loadUiType
# import pyqtgraph as pg
import sys
import cnnobjectness as cob
from tensorflow import Graph,Session
from copy import copy
# from lasercalibrationroutine import estimate
# import ilimbcontroller as mylimb



dictofcolors={}
dictofcolors[0]=(255,0,0) ### Red
dictofcolors[1]=(0,255,0)  ### Green
dictofcolors[2]=(0,0,255) #### Blue
dictofcolors[3]=(255,255,0)  ### Yellow
dictofcolors[4]=(128,0,0)   ### Maroon
dictofcolors[5]=(128,0,128) ###  Purple


def laserfilter(listoflasers):

	scores=np.zeros(len(listoflasers))
	# print(listoflasers)
	for i in range(len(listoflasers)):
		for j in range(i+1,len(listoflasers)):
			if(abs(listoflasers[i]-listoflasers[j])< 300):
				scores[i]+=1
				scores[j]+=1
	indices= np.argsort(scores)
	# print(indices)
	return listoflasers[indices[-1]]

def imagelaserfilter(listofimagelasers):

	scores=np.zeros(len(listofimagelasers))

	for i in range(len(listofimagelasers)):
		for j in range(i+1,len(listofimagelasers)):
			if(abs(listofimagelasers[i][0]-listofimagelasers[j][0])+ abs(listofimagelasers[i][1]-listofimagelasers[j][1]) < 6):
				scores[i]+=1
				scores[j]+=1

	indices= np.argsort(scores)
	return 30+listofimagelasers[indices[-1]][0],64+listofimagelasers[indices[-1]][1]


def filtering(boundingboxes,depth,x,y):
	filteredboundingboxes=[]
	for rectangle in boundingboxes:
		leftx=rectangle[0]
		rightx=rectangle[1]
		bottomy=rectangle[2]
		topy=rectangle[3]
		if(x>leftx and x<rightx and y>bottomy and y<topy):
			filteredboundingboxes.append(rectangle)

	return filteredboundingboxes


def iomafilter(objectnessboundingboxes,objectness):
	# newboundingboxes=copy(objectnessboundingboxes)
	# newobjectness=copy(objectness)
	# for i in range(len(objectnessboundingboxes)):
	# 	for j in range(i+1,len(objectnessboundingboxes)):
	# 		check=ioma(objectnessboundingboxes[i], objectnessboundingboxes[j])
	# 		# print(check)
	# 		if(check > 0.9):
	# 			if(objectness[i]> objectness[j]):
	# 				if(objectnessboundingboxes[j] in newboundingboxes):
	# 					newobjectness.remove(objectness[j])
	# 					newboundingboxes.remove(objectnessboundingboxes[j])
	# 			else:
	# 				if(objectnessboundingboxes[i] in newboundingboxes):
	# 					newobjectness.remove(objectness[i])
	# 					newboundingboxes.remove(objectnessboundingboxes[i])

	# return newboundingboxes,newobjectness

	score=np.zeros(len(objectnessboundingboxes))
	newboundingboxes=[]
	newobjectness=[]
	for i in range(len(objectnessboundingboxes)):
		for j in range(i+1,len(objectnessboundingboxes)):
			check=ioma(objectnessboundingboxes[i], objectnessboundingboxes[j])
			if(check>0.9):
				if(objectness[i]>objectness[j]):
					score[i]+=1
					score[j]-=1
				else:
					score[i]-=1
					score[j]+=1
			else:
				score[i]+=1
				score[j]+=1

	if(len(score)>=5):
		indices=np.argsort(score)[-5:]
	else:
		indices=[i for i in range(len(objectnessboundingboxes))]

	for index in list(indices):
		newboundingboxes.append(objectnessboundingboxes[index])
		newobjectness.append(objectness[index])

	return newboundingboxes,newobjectness

def ioma(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	# print(boxA,boxB)
	xA = max(boxA[0], boxB[0])
	xB = min(boxA[1], boxB[1])
	yA = max(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
	boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the max of two areas
	iom = interArea / max(boxAArea,boxBArea)

	# return the intersection over union value
	return iom

def containment(boxA,boxB):
	# print(boxA,boxB)
	xA = max(boxA[0], boxB[0])
	xB = min(boxA[1], boxB[1])
	yA = max(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
	boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)

	overlapA=interArea/boxAArea
	overlapB= interArea/boxBArea

	if(overlapA>0.9 and overlapB<0.5):
		return 1
	elif(overlapA<0.5 and overlapB>0.9):
		return 2
	else:
		return 0

def containedwithin(boundingboxes,objectness):

	filteredboundingboxes=copy(boundingboxes)
	newobjectness=copy(objectness)
	for i in range(len(boundingboxes)):
		for j in range(i+1,len(boundingboxes)):
			containmentcheck=containment(boundingboxes[i],boundingboxes[j])
			if(containmentcheck==1):
				if(boundingboxes[j] in filteredboundingboxes):
					newobjectness.remove(objectness[j])
					filteredboundingboxes.remove(boundingboxes[j])
			elif(containmentcheck==2):
				if(boundingboxes[i] in filteredboundingboxes):
					newobjectness.remove(objectness[i])
					filteredboundingboxes.remove(boundingboxes[i])

	return filteredboundingboxes,newobjectness




def crop(image,leftx,rightx,bottomy,topy):
	newimage=[[image[i][j] for j in range(leftx,rightx)] for i in range(bottomy,topy)]
	return newimage

def padwithzeros(image):
	deltax=len(image[0])
	deltay=len(image)
	deficitdeltax=deltay-deltax
	deficitdeltay=deltax-deltay
	siz=max(deltax,deltay)
	tempimage=np.zeros((siz,siz))
	if(deltax>deltay):
		for i in range(siz):
			for j in range(deltay):
				tempimage[j+mt.floor(deficitdeltay/2)][i]=image[j][i]
	else:
		for i in range(deltax):
			for j in range(siz):
				tempimage[j][i+mt.floor(deficitdeltax/2)]=image[j][i]
	return tempimage

def preprocess1(image,leftx,rightx,bottomy,topy):
	newimage=[[0 for i in range(64)] for j in range(64)]
	deltax=rightx-leftx
	deltay=topy-bottomy
	deficitdeltax=mt.floor(deltay-deltax)
	deficitdeltay=mt.floor(deltax-deltay)
	if(deltax>=deltay):
		alpha=64/deltax
		for i in range(rightx-leftx):
			for j in range(topy-bottomy):
				if(image[j+bottomy][i+leftx]==1):
					newimage[32-mt.ceil(alpha*(mt.floor(deltax/2)-mt.floor(deficitdeltay/2)-j))][32-mt.ceil(alpha*(mt.floor(deltax/2)-i))]=1

	if(deltay>deltax):
		alpha=64/deltay
		for i in range(rightx-leftx):
			for j in range(topy-bottomy):
				if(image[j+bottomy][i+leftx]==1):
					newimage[32-mt.ceil(alpha*(mt.floor(deltay/2)-j))][32-mt.ceil(alpha*(mt.floor(deltay/2)-mt.floor(deficitdeltax/2)-i))]=1

	return newimage


def preprocess3(image):
	image=np.array(image)
	newimage=cv2.resize(image,(64,64),interpolation = cv2.INTER_CUBIC)
	return newimage

def preprocess(image,rectangle):
	image=crop(image,rectangle[0],rectangle[1],rectangle[2],rectangle[3])
	newimage=padwithzeros(image)
	newimage=preprocess3(newimage)
	return newimage



def custommetric(y_true,y_pred):
	return metrics.top_k_categorical_accuracy(y_pred, y_true, k=2)
	

def calculatedepth(lasex,lasery):
	return 5000/lasery

def createimage(output,type):

	if(type==0):
		image=np.zeros((128,128))
		displayimage=np.zeros((509,509))
		for i in range(len(output['t'])):
			image[output['y'][i]][output['x'][i]]=1
			displayimage[4*output['x'][i]][4*output['y'][i]]=1
		image=np.array(image)
		image=image.astype(np.double)
		return image,displayimage,np.sum(image)

	elif(type==1):
		minusinfi= -1e-9
		param=50000
		output['t']=output['t']-output['t'][0]
		prev=minusinfi+np.zeros((128,128))
		prevdisplay=minusinfi+np.zeros((509,509))
		siz=len(output['t'])
		for i in range(siz-1):
			prev[output['y'][i]][output['x'][i]]=output['t'][i]
			prevdisplay[4*output['x'][i]][4*output['y'][i]]=output['t'][i]

		image=[[mt.exp(-((output['t'][siz-1]-prev[q][p])/param)) for p in range(128)] for q in range(128)]
		cnndisplayimage=[[mt.exp(-((output['t'][siz-1]-prevdisplay[p][q])/param)) for q in range(509)] for p in range(509)]

		image=np.array(image)
		cnndisplayimage=np.array(cnndisplayimage)
		return image,cnndisplayimage, np.sum(image)


##################################################


def MyThread1():
	global backendstring
	global stopstore
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	addr = 'localhost', 8991
	s.bind(addr)
	BUFSIZE = 8192
	while(len(backendstring)<12000):
		data, fromaddr = s.recvfrom(BUFSIZE)
		backendstring+=data[4:]


	s.close()
	s.close()
	stopstore=1

def MyThread2():
	global backendstring
	global laserlocations
	global stopstore
	global approxlaserlocations
	global stoplaser
	global depth
	pointerlaser=0

	start=time.clock()
	while(True):
		if(stopstore==1):
			outputforlaser= nrt.xypt(backendstring)
			outputforlaser= nflt.filtertd(outputforlaser)
			imageforlaserdet=mt.floor(outputforlaser['t'][-1]/10)
			index=0

			while(pointerlaser+imageforlaserdet< outputforlaser['t'][-1]):

				previndex=index
				i=previndex
				while(True):
					if(outputforlaser['t'][i]>pointerlaser+imageforlaserdet):
						index=i
						break
					i+=1

				output={}
				output['t']=outputforlaser['t'][previndex:index]
				output['x']=outputforlaser['x'][previndex:index]
				output['y']=outputforlaser['y'][previndex:index]
				output['p']=outputforlaser['p'][previndex:index]
				x,y=rtl.laserdet(output)
				pointerlaser=pointerlaser+imageforlaserdet
				# myimage,garbage1,garbage2 = createimage(output,0)
				# myimage=np.array(myimage)
				# myimage=myimage.astype(np.double)
				# cv2.imshow('image',myimage)
				# cv2.waitKey(0)
				# print(y) 

				if(y!=32):
					tempdepth=calculatedepth(x,y)
					# print(y)
					# print(tempdepth)
					approxlaserlocations.append([x,y])
					laserlocations.append([tempdepth,time.clock()-start])
					start=time.clock()

			laserlocations=np.array(laserlocations)
			depthcheck=laserfilter(laserlocations[:,0])
			if(depthcheck>300):
				depth=depthcheck
			else:
				depth=depthcheck
			stoplaser=1
			break

	

def MyThread3():
	global bounding_boxes
	global cnnimagesset
	# global displayimagesset
	global boundingboxesset
	global stopstore
	global stopcreation
	pointer=0
	imageforcnn=12000
	while(True):
		if(stopstore==1 and pointer+imageforcnn>len(backendstring)):
			break
		if(pointer+imageforcnn<len(backendstring)):
			output=nrt.xypt(backendstring[pointer:pointer+imageforcnn])
			pointer+=imageforcnn
			image,displayimage,noofevents=createimage(output,0)
			bounding_boxes=intm.integral(image,noofevents)
			# print(bounding_boxes)
			cnnimagesset.append(image)
			# displayimagesset.append(displayimage)
			boundingboxesset.append(bounding_boxes)
	stopcreation=1



dictfordepthspeed={}
for i in range(1,21):
	dictfordepthspeed[i*50]=i


model1 = Sequential()


model1.add(Conv2D(32,(3,3),activation = 'relu', input_shape = (64,64,1)))
model1.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model1.add(BatchNormalization())
#model1.add(Dropout(0.1))
model1.add(Conv2D(64, (3,3), activation='relu'))
model1.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model1.add(BatchNormalization())
#model1.add(Dropout(0.1))

model1.add(Conv2D(128, (3,3), activation='relu'))

model1.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model1.add(BatchNormalization())
#model1.add(Dropout(0.1))

model1.add(Conv2D(256, (3,3), activation='relu'))

#model1.add(Dropout(0.1))
model1.add(BatchNormalization())

model1.add(Flatten())
model1.add(Dense(6, activation = 'softmax'))
model1.load_weights('graspnetwork.05.h5')



# model2 = Sequential()
# model2.add(Conv2D(32,(6,6),activation = 'relu', input_shape = (64,64,1)))
# model2.add(MaxPooling2D(pool_size = (2,2),padding='same'))
# model2.add(BatchNormalization())
# #model2.add(Dropout(0.1))
# model2.add(Conv2D(64, (3,3), activation='relu'))
# model2.add(MaxPooling2D(pool_size = (2,2),padding='same'))
# model2.add(BatchNormalization())
# #model2.add(Dropout(0.1))

# model2.add(Conv2D(128, (3,3), activation='relu'))
# model2.add(MaxPooling2D(pool_size = (2,2),padding='same'))
# model2.add(BatchNormalization())
# #model2.add(Dropout(0.1))

# model2.add(Flatten())
# model2.add(Dense(2, activation = 'softmax'))

# model2.load_weights('objectnessnetwork.07.h5')

model2=Sequential()

model2.add(Conv2D(32,(6,6),activation = 'relu', input_shape = (64,64,1), kernel_initializer=initializers.RandomNormal(stddev=0.001)))
model2.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model2.add(BatchNormalization())
#model2.add(Dropout(0.1))
model2.add(Conv2D(64, (3,3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001)))
model2.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model2.add(BatchNormalization())
#model2.add(Dropout(0.1))

model2.add(Conv2D(128, (3,3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001)))
model2.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model2.add(BatchNormalization())
#model2.add(Dropout(0.1))

model2.add(Flatten())
model2.add(Dense(1, activation = 'sigmoid',kernel_initializer=initializers.RandomNormal(stddev=0.001)))
model2.load_weights('objectnessnetworkiou_2objects_nearest.08.h5')


firsttime=0
while(True):
	if(firsttime!=0):
		pass
		# mylimb.restart()
		# input("Position yourself to grasp")
		# print("Start your motion")
	else:
		print("Setting up")
		firsttime+=1
		# mylimb.start()
		# mylimb.start()
		continue


	cnnimagesset=[]
	boundingboxesset=[]
	objectnessboundingboxes=[]
	backendstring=b''
	bounding_boxes=[]
	laserlocations=[]
	# displayimagesset=[]
	beliefs=np.zeros(6)
	stopstore=0
	indicatorfordepth=0
	stopcreation=0
	
	stoplaser=0
	depth=0
	speed=0
	approxlaserlocations=[]
	

	t1 = threading.Thread(target=MyThread1, args=[])
	t2 = threading.Thread(target=MyThread2, args=[])
	t3 = threading.Thread(target=MyThread3,args=[])
	t1.start()
	t2.start()
	t3.start()
	while(True):
		if(stopstore==1 and stopcreation==1  and stoplaser==1):

			# print(depth)
			# approxlaserlocationx,approxlaserlocationy=imagelaserfilter(approxlaserlocations)
			# print(approxlaserlocationx,approxlaserlocationy)

			# boundingboxesset[-1]= filtering(boundingboxesset[-1],depth,approxlaserlocationx,approxlaserlocationy)

			objectness,objectnessboundingboxes = cob.cnnobjectness(cnnimagesset[-1] ,boundingboxesset[-1],model2)

			# print(objectness)

			# for i in range(len(objectnessboundingboxes)):
			# 	# print(objectness[len(boundingboxesset[-1])-i-1])
			# 	img=np.array(cnnimagesset[-1])
			# 	img=img.astype(np.double)
			# 	img=np.expand_dims(img,axis=-1)
			# 	img=np.repeat(img,3,axis=-1)
			# 	img=255*img
			# 	print(objectness[i])
			# 	rectangle=objectnessboundingboxes[i]
			# 	# rectangle=newfilteredboundingboxes[len(newfilteredboundingboxes)-i-1]
			# 	# rectangle=boundingboxesset[-1][len(boundingboxesset[-1])-i-1]
			# 	# rectangle=objectnessboundingboxes[len(objectnessboundingboxes)-i-1]
			# 	cv2.rectangle(img, (rectangle[0],rectangle[2]),(rectangle[1],rectangle[3]),(255,0,0),1)
			# 	cv2.imshow('image',img)
			# 	cv2.waitKey(0)
			# 	cv2.destroyAllWindows()


			# filteredboundingboxes,filteredobjectness= containedwithin(objectnessboundingboxes,objectness)

			# print(filteredboundingboxes,"hello")

			# newfilteredboundingboxes, newfilteredobjectness= iomafilter(filteredboundingboxes,filteredobjectness)

			# print(newfilteredboundingboxes,"ankit")

			# tempbeliefs,impboundingboxes,objecttype=cno.cnnobject(cnnimagesset[-1],filteredboundingboxes,model1)

			# print("buffalo")
			# print(tempbeliefs)
			# print(objecttype)
			# beliefs=tempbeliefs

			# img=np.array(cnnimagesset[-1])
			# img=img.astype(np.double)
			# img=np.expand_dims(img,axis=-1)
			# img=np.repeat(img,3,axis=-1)
			# img=255*img
			for i in range(len(objectnessboundingboxes)):
				print(objectness[len(objectnessboundingboxes)-i-1])
				img=np.array(cnnimagesset[-1])
				img=img.astype(np.double)
				img=np.expand_dims(img,axis=-1)
				img=np.repeat(img,3,axis=-1)
				img=255*img
				# print(filteredobjectness[i])
				# rectangle=filteredboundingboxes[i]
				# rectangle=newfilteredboundingboxes[len(newfilteredboundingboxes)-i-1]
				# rectangle=boundingboxesset[-1][len(boundingboxesset[-1])-i-1]
				rectangle=objectnessboundingboxes[len(objectnessboundingboxes)-i-1]
				
				cv2.rectangle(img, (rectangle[0],rectangle[2]),(rectangle[1],rectangle[3]),(255,0,0),1)
				cv2.imshow('image',img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

			input("Wait")

			# if(np.sum(beliefs)!=0):
				
			# 	input("Wait to check depth")
			# 	timetosleep=dictfordepthspeed[50*int(depth/50)]
			# 	print(timetosleep)
			# 	arg=np.argmax(beliefs)
			# 	if(arg==0):
			# 		mylimb.verticalpowergrasp(timetosleep)
			# 	if(arg==1):
			# 		mylimb.verticaltripod(timetosleep)
			# 	if(arg==2):
			# 		mylimb.verticalpinch(timetosleep)
			# 	if(arg==3):
			# 		mylimb.horizontalpowergrasp(timetosleep)
			# 	if(arg==4):
			# 		mylimb.horizontaltripod(timetosleep)
			# 	if(arg==5):
			# 		mylimb.horizontalpinch(timetosleep)
				
			# 	input("Wait for release")

				
			# 	mylimb.open()
			# 	mylimb.open()
			# 	if(arg==3 or arg==4 or arg==5):
			# 		mylimb.anticlockwise()
			# else:
			# 	print("No object detected")

			break