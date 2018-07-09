import myread as load
import filtertd as flt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.inf)
import math as mt
from scipy import ndimage
import scipy.io
from skimage.transform import resize
import glob
from pathlib import Path
import random



def getcom(image):
	comx=0   ######## x-coordinate Centre of mass of events in filtered image
	comy=0   #######  y-coordinate Centre of mass of events in filtered image
	total_events=0  ######  total evens in that time frame

	for i in range(128):
		for j in range(128):
			comx+=j*image[i][j]
			comy+=i*image[i][j]
			total_events+= image[i][j]

	comx= int(comx/total_events)
	comy= int(comy/total_events)

	return comx,comy

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


def translatetocenter(comx,comy,image,dim):
	new_image_filt=[[0 for i in range(dim)] for j in range(dim)]
	for i in range(dim):
		for j in range(dim):
			coord_x=int(dim/2-comx+i)
			coord_y=int(dim/2-comy+j)
			if(coord_y>=0 and coord_y<dim and coord_x>=0 and coord_x<dim):
				new_image_filt[coord_y][coord_x]= image[j][i]
	return new_image_filt

def createboundingbox(image_orig,noofeventsreq,threshed,comx,comy,upper_bound):
	noofevents=0
	if((comx-2) >= 0):
		leftx= comx-2
	else:
		leftx=comx

	if((comx+2) < 128):
		rightx=comx+2
	else:
		rightx=comx

	if((comy-2) >= 0):
		bottomy= comy-2
	else:
		bottomy= comy

	if((comy+2) < 128):
		topy=comy+2
	else:
		topy=comy

	
	for i in range(leftx,rightx+1):
		for j in range(bottomy,topy+1):
			noofevents += image_orig[j][i]
	####### Expanding the bounding box one step in a direction corresponding to maximum imcrease

	while(noofevents<upper_bound):
		valltrt = 0
		valbttp= 0
		for i in range(bottomy,topy+1):
			if(leftx-1 >=0):
				valltrt += image_orig[i][leftx-1]
			if(rightx+1<128):
				valltrt += image_orig[i][rightx+1]

		for j in range(leftx,rightx+1):
			if(bottomy-1 >=0):
				valbttp += image_orig[bottomy-1][j]
			if(topy+1< 128):
				valbttp += image_orig[topy+1][j]

		if(valbttp>valltrt and noofevents>noofeventsreq):
			if(valbttp<(rightx-leftx)*threshed):
				break
		if(valbttp<=valltrt and noofevents>noofeventsreq):
			if(valbttp<(topy-bottomy)*threshed):
				break

		if(valbttp > valltrt):
			if(topy+1<128):
				topy+=1
			if(bottomy-1>=0):
				bottomy -=1
			noofevents += valbttp
		elif(valltrt > valbttp):
			if(leftx-1>=0):
				leftx-=1
			if(rightx+1<128):
				rightx +=1
			noofevents += valltrt
		else:
			if(topy+1<128 or bottomy-1>=0):
				if(topy+1<128):
					topy+=1
				if(bottomy-1>=0):
					bottomy -=1
				noofevents += valbttp
			else:
				if(leftx-1>=0):
					leftx-=1
				if(rightx+1<128):
					rightx +=1
				noofevents += valltrt

	return leftx,rightx,bottomy,topy

	# # randomlist=[5,10,15,20]
	# fourpoints=[0 for i in range(4)]
	# fourpoints[0]=mt.floor(random.uniform(1,20))
	# fourpoints[1]=mt.floor(random.uniform(1,20))
	# fourpoints[2]=mt.floor(random.uniform(1,20))
	# fourpoints[3]=mt.floor(random.uniform(1,20))

	# orig_left=leftx
	# orig_right=rightx
	# orig_bottomy=bottomy
	# orig_topy= topy

	# leftx=leftx-fourpoints[0]
	# rightx=rightx+fourpoints[1]
	# bottomy=bottomy-fourpoints[2]
	# topy=topy+fourpoints[3]

	# # print(fourpoints)

	# # if(leftx>=5):
	# # 	leftx=leftx-5
	# # if(rightx<=122):
	# # 	rightx+=5
	# # if(bottomy>=5):
	# # 	bottomy=bottomy-5
	# # if(topy<=122):
	# # 	topy+=5

	# if(leftx>=0 and rightx< 128 and bottomy>=0 and topy<128 and rightx>leftx and topy>bottomy):
	# 	return [orig_left,orig_right,orig_bottomy,orig_topy],[leftx,rightx,bottomy,topy]
	# else:
	# 	return [orig_left,orig_right,orig_bottomy,orig_topy],[orig_left,orig_right,orig_bottomy,orig_topy]


def myfun(image):
	for i in range(image.ndim):
		image=image.cumsum(axis=i)
	return image

def numberofevents(integral_image,coord_x_left,coord_x_right,coord_y_left,coord_y_right):
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
		myeventslist=[]
		myeventslist.append(numberofevents(integral_image,coord_x_left,coord_x_right,coord_y_left,coord_y_left+1))
		myeventslist.append(numberofevents(integral_image,coord_x_left,coord_x_left+1,coord_y_left,coord_y_right))
		myeventslist.append(numberofevents(integral_image,coord_x_left,coord_x_right,coord_y_right-1,coord_y_right))
		myeventslist.append(numberofevents(integral_image,coord_x_right-1,coord_x_right,coord_y_left,coord_y_right))
		myindex= myeventslist.index(min(myeventslist))
		coord_x_left+=mylist[myindex][0]
		coord_x_right+=mylist[myindex][1]
		coord_y_left+=mylist[myindex][2]
		coord_y_right+=mylist[myindex][3]
		runningsum+=min(myeventslist)
	return [coord_x_left,coord_x_right,coord_y_left,coord_y_right]

def overlap_iou(boxA, boxB):
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


def createboundingbox2(image):
	image=np.array(image)
	integral_image= myfun(image)
	total=0
	for i in range(32,96):
		for j in range(32,96):
			total+= image[i][j]
	mylist= downsize(integral_image,32,96,32,96,total)
	return mylist

def crop(image,leftx,rightx,bottomy,topy):
	newimage=[[image[i][j] for j in range(leftx,rightx)] for i in range(bottomy,topy)]
	return newimage


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

def preprocess2(image):
	newimage=[[0 for i in range(64)] for j in range(64)]
	for i in range(64):
		for j in range(64):
			newimage[i][j]= image[32+i][32+j]
	return newimage
 
def preprocess3(image):
	image=np.array(image)
	newimage=cv2.resize(image,(64,64),interpolation = cv2.INTER_NEAREST)
	return newimage

##########################################

mypath="testingnewdata"
objectnessdata="testingobjectnessdataiou_2objects_nearest"

myfilespath = glob.glob(mypath + '/**/*.aedat', recursive=True)
listofobjects={}
winubu='/'

if (not(os.path.exists(objectnessdata))):
	os.makedirs(objectnessdata)

name="negative"
listofobjects[name]=0

##########################################

for twoobjects in range(500):

	[filepath1,filepath2]=random.sample(myfilespath,2)

	print(filepath1,filepath2)

	output_orig1 = load.xypt(filepath1)
	output_orig1['t'] = np.array(output_orig1['t'])
	output_orig1['t'] = output_orig1['t']-output_orig1['t'][0]

	output_orig2 = load.xypt(filepath2)
	output_orig2['t'] = np.array(output_orig2['t'])
	output_orig2['t'] = output_orig2['t']-output_orig2['t'][0]

	stdx= 20
	stdy= 20
	stdxy=5
	threshreg= 0.3
	start=3
	group=1000+mt.floor(1000*np.random.random())


####### File selected and now creating images out of it 
	for counter in range(start,min(mt.floor(len(output_orig1['t'])/group),mt.floor(len(output_orig2['t'])/group))):

		image_orig1= [[0 for i in range(128)] for j in range(128)]    ############## Image constructed from original data for events corresponding to last timestamp of filtered events selected

		image_orig2= [[0 for i in range(128)] for j in range(128)]

		for i in range((counter-1)*group,(counter)*group):
			image_orig1[output_orig1['y'][i]][output_orig1['x'][i]] = 1
			image_orig2[output_orig2['y'][i]][output_orig2['x'][i]] = 1


		comx,comy=getcom(image_orig1)
		total_events=0

		for i in range(128):
			for j in range(128):
				total_events+=image_orig1[i][j]

		thresh = 0.8
		threshed=total_events/(128*128)

		noofevents=0    ########## number of events present in current bounding box
		noofeventsreq= int(thresh*total_events)  #####  number of events that should be present in the bounding box
	
		upper_bound=0.95*total_events
		leftx1,rightx1,bottomy1,topy1=createboundingbox(image_orig1,noofeventsreq,threshed,comx,comy,upper_bound)



		comx,comy=getcom(image_orig2)
		total_events=0

		for i in range(128):
			for j in range(128):
				total_events+=image_orig2[i][j]

		thresh = 0.8
		threshed=total_events/(128*128)

		noofevents=0    ########## number of events present in current bounding box
		noofeventsreq= int(thresh*total_events)  #####  number of events that should be present in the bounding box
	
		upper_bound=0.95*total_events
		leftx2,rightx2,bottomy2,topy2=createboundingbox(image_orig2,noofeventsreq,threshed,comx,comy,upper_bound)


		image_orig1=np.array(image_orig1)
		image_orig1=image_orig1.astype(np.double)
		# cv2.rectangle(image_orig1,(leftx1,bottomy1),(rightx1,topy1),(100,100,100),1)
		# cv2.imshow('image',image_orig1)
		# cv2.waitKey(0)

		image_orig2=np.array(image_orig2)
		image_orig2=image_orig2.astype(np.double)
		# cv2.rectangle(image_orig2,(leftx2,bottomy2),(rightx2,topy2),(100,100,100),1)
		# cv2.imshow('image',image_orig2)
		# cv2.waitKey(0)

		width1=rightx1-leftx1
		height1= topy1-bottomy1
		width2= rightx2-leftx2
		height2= topy2-bottomy2

		for generation in range(10):
			negativeimage=[[0 for i in range(128)] for j in range(128)]
			fourpoints=[0 for i in range(4)]
			fourpoints[0]=mt.floor(random.uniform(0,127))
			fourpoints[1]=mt.floor(random.uniform(0,127))
			fourpoints[2]=mt.floor(random.uniform(0,127))
			fourpoints[3]=mt.floor(random.uniform(0,127))

			if(fourpoints[0]+width1<128 and fourpoints[1]+height1<128 and fourpoints[2]+width2<128 and fourpoints[3]+height2<128):
				for i in range(width1):
					for j in range(height1):
						negativeimage[fourpoints[1]+j][fourpoints[0]+i]=image_orig1[bottomy1+j][leftx1+i]


				for i in range(width2):
					for j in range(height2):
						negativeimage[fourpoints[3]+j][fourpoints[2]+i]=image_orig2[bottomy2+j][leftx2+i]
				
				# negativeimage=np.array(negativeimage)
				# negativeimage=negativeimage.astype(np.double)	
				# cv2.imshow('image', negativeimage)
				# cv2.waitKey(0)

				leftx=min(fourpoints[0],fourpoints[2])
				rightx=max(fourpoints[0]+width1,fourpoints[2]+width2)
				bottomy=min(fourpoints[1],fourpoints[3])
				topy=max(fourpoints[1]+height1,fourpoints[3]+height2)

				croppedimage= crop(negativeimage,leftx,rightx,bottomy,topy)

				croppedimage=np.array(croppedimage)
				croppedimage=croppedimage.astype(np.double)	
				# cv2.imshow('image', croppedimage)
				# cv2.waitKey(0)

				# trainingimage=padwithzeros(croppedimage)
				trainingimage=preprocess3(croppedimage)
				# print("positive")
				# cv2.imshow('image',trainingimage)
				# cv2.waitKey(0)

				mydict={}
				mydict['data']=trainingimage
				mydict['label']=0	
				f = open(objectnessdata+winubu+name+str(listofobjects[name]),"wb")
				listofobjects[name] += 1
				pickle.dump(mydict,f)
				f.close()