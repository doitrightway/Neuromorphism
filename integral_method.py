import myread as load
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import math as mt
import filtertd as flt
import random
from sklearn.cluster import KMeans
import pygame
import time
from scipy import signal

def myfun(image):
	for i in range(image.ndim):
		image=image.cumsum(axis=i)
	return image

def noofevents(integral_image,coord_x_left,coord_x_right,coord_y_left,coord_y_right):
	# print(coord_x_left,coord_x_right,coord_y_left,coord_y_right)
	a=integral_image[coord_y_left][coord_x_right]
	b=integral_image[coord_y_left][coord_x_left]
	c=integral_image[coord_y_right][coord_x_left]
	return integral_image[coord_y_right][coord_x_right]-a-c+b

def downsize(integral_image,coord_x_left,coord_x_right,coord_y_left,coord_y_right,total):
	runningsum=0
	thresh=0.05*total
	mylist=[[0,0,1,0],[1,0,0,0],[0,0,0,-1],[0,-1,0,0]]
	while(runningsum<thresh):
		eventslist=[]
		eventslist.append(noofevents(integral_image,coord_x_left,coord_x_right,coord_y_left,coord_y_left+1))
		eventslist.append(noofevents(integral_image,coord_x_left,coord_x_left+1,coord_y_left,coord_y_right))
		eventslist.append(noofevents(integral_image,coord_x_left,coord_x_right,coord_y_right-1,coord_y_right))
		eventslist.append(noofevents(integral_image,coord_x_right-1,coord_x_right,coord_y_left,coord_y_right))
		myindex= eventslist.index(min(eventslist))
		coord_x_left+=mylist[myindex][0]
		coord_x_right+=mylist[myindex][1]
		coord_y_left+=mylist[myindex][2]
		coord_y_right+=mylist[myindex][3]
		runningsum+=min(eventslist)

	area=(coord_x_right-coord_x_left)*(coord_y_right-coord_y_left)

	return [coord_x_left,coord_x_right,coord_y_left,coord_y_right],area


def overlap(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
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
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou


def integral(image,total_events):
	# start=time.clock()
	boundingboxes=[]
	siz=total_events
	integral_image= myfun(image)
	numberrange=5
	dividerange= mt.floor(127/numberrange)
	for coord_x_left in range(numberrange):
		for coord_y_left in range(numberrange):
			for coord_x_right in range(coord_x_left+1,numberrange+1):
				for coord_y_right in range(coord_y_left+1,numberrange+1):
					area=(coord_x_right*dividerange-coord_x_left*dividerange)*(coord_y_right*dividerange-coord_y_left*dividerange)
					# print(coord_x_left,coord_x_right,coord_y_left,coord_y_right)
					total = noofevents(integral_image,coord_x_left*dividerange,coord_x_right*dividerange,coord_y_left*dividerange,coord_y_right*dividerange)
					# print("area",area)
					if(total>0.1*siz):
					# if(area>0):
						rectangle,myarea=downsize(integral_image,coord_x_left*dividerange,coord_x_right*dividerange,coord_y_left*dividerange,coord_y_right*dividerange,total)
						# print("myarea",myarea)
						if(myarea!=0 and 0.95*total> (myarea*siz)/(128*128)):
							boundingboxes.append(rectangle)


	for i in range(len(boundingboxes)):
		leftx=boundingboxes[i][0]
		rightx=boundingboxes[i][1]
		bottomy=boundingboxes[i][2]
		topy=boundingboxes[i][3]
		if(leftx>=5):
			leftx=leftx-5
		if(rightx<=122):
			rightx+=5
		if(bottomy>=5):
			bottomy=bottomy-5
		if(topy<=122):
			topy+=5
		boundingboxes[i][0]=leftx
		boundingboxes[i][1]=rightx
		boundingboxes[i][2]=bottomy
		boundingboxes[i][3]=topy


	# return boundingboxes
	print(len(boundingboxes),"before")

	remove=[0 for i in range(len(boundingboxes))]
	for i in range(len(boundingboxes)):
		for j in range(i+1,len(boundingboxes)):
			if(overlap(boundingboxes[i],boundingboxes[j])>0.9 and remove[i]==0 and remove[j]==0):
				remove[i]=1

	newboundingboxes=[]
	for i in range(len(boundingboxes)):
		if(remove[i]==0):
			newboundingboxes.append(boundingboxes[i])

	print(len(newboundingboxes),"after")
	return newboundingboxes

