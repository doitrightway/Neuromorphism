import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.io
from skimage.transform import resize
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf
import keras.backend as K
import os.path
from keras.models import model_from_json
import cv2
import random
import time
import math as mt


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
	# print(rectangle)
	image=crop(image,rectangle[0],rectangle[1],rectangle[2],rectangle[3])
	newimage=padwithzeros(image)
	newimage=preprocess3(newimage)
	return newimage



# dictofobjects={}
# dictofobjects[1]='Camera'
# dictofobjects[2]='Cup'
# dictofobjects[3]='Mouse'
# dictofobjects[4]='Pen'
# dictofobjects[5]='Phone'
# dictofobjects[6]='Scissor'
# dictofobjects[7]='Spectacle'
# dictofobjects[8]='Water Bottle'

def cnnobjectness(image,boundingboxes,loaded_model):
	dim=64
	impboundingboxes=[]
	xte= np.empty((3*len(boundingboxes),dim,dim,1))
	for i in range(len(boundingboxes)):
		# rectangle1=[]
		rectangle1= [boundingboxes[i][0],mt.floor((boundingboxes[i][0]+boundingboxes[i][1])/2),boundingboxes[i][2],boundingboxes[i][3]]
		rectangle2= [mt.floor((boundingboxes[i][0]+boundingboxes[i][1])/2),boundingboxes[i][1],boundingboxes[i][2],boundingboxes[i][3]]
		passimage=preprocess(image,boundingboxes[i])
		passimage=np.array(passimage)
		xte[3*i]=passimage.reshape(dim,dim,1)

		passimage=preprocess(image,rectangle1)
		passimage=np.array(passimage)
		xte[3*i+1]=passimage.reshape(dim,dim,1)

		passimage=preprocess(image,rectangle2)
		passimage=np.array(passimage)
		xte[3*i+2]=passimage.reshape(dim,dim,1)


	inp = loaded_model.input                                           # input placeholder
	outputs = [layer.output for layer in loaded_model.layers]          # all layer outputs

	functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
	# Testing
	test = xte
	layer_outs = [func([test]) for func in functors]
	# print(layer_outs)

	objectness=[]
	for i in range(len(boundingboxes)):
		value1=layer_outs[-1][0][3*i]
		value2=layer_outs[-1][0][3*i+1]
		value3=layer_outs[-1][0][3*i+2]
		probmax=int(np.argmax(value1))
		if(probmax==1 and value1[1]> 0.5):
			if((value2[1]+value3[1])/2 < 0.1):
				impboundingboxes.append(boundingboxes[i])
				objectness.append(layer_outs[-1][0][3*i][1])


	# objectness=[]
	# for i in range(len(boundingboxes)):
	# 	objectness.append(layer_outs[-1][0][3*i][1])

	return objectness,impboundingboxes
