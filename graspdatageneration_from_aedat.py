import myread as load
# import filtertd as flt
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

	if(leftx>=5):
		leftx=leftx-5
	if(rightx<=122):
		rightx+=5
	if(bottomy>=5):
		bottomy=bottomy-5
	if(topy<=122):
		topy+=5
	return leftx,rightx,bottomy,topy



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
	newimage=cv2.resize(image,(64,64),interpolation = cv2.INTER_CUBIC)
	return newimage





####################################################



mypath="testingnewdata"
mytraingraspdata= "testingnew1500graspdata24thjuneonly1064bicubic"

myfilespath = glob.glob(mypath + '/**/*.aedat', recursive=True)
listofobjects={}
listofgrasp= {}
winubu='\\'
smoothdict={}
smoothdict[1]=[0.2,0.7,0.1,0,0,0]
smoothdict[2]=[0.7,0.2,0.1,0,0,0]
smoothdict[3]=[0,0,0,0.2,0.7,0.1]
smoothdict[4]=[0,0,0,0.7,0.2,0.1]
smoothdict[5]=[0,0,0,0.1,0.2,0.7]
smoothdict[6]=[0.1,0.2,0.7,0,0,0]

if (not(os.path.exists(mytraingraspdata))):
	os.makedirs(mytraingraspdata)

##########################################


for filepath in myfilespath:

	print(filepath)

	filename = os.path.basename(filepath)
	tmp = os.path.splitext(filename)[0]
	grasptype= tmp[5:6]
	objecttype=tmp[6:-10]

	objecttype=objecttype.strip("0123456789")

	tmp=grasptype+objecttype

	if(not(tmp in listofgrasp.keys())):
		listofgrasp[tmp]=0

	# output_filt = flt.filtertd(filepath)
	output_orig = load.xypt(filepath)
	output_orig['t'] = np.array(output_orig['t'])
	output_orig['t'] = output_orig['t']-output_orig['t'][0]

	stdx= 20
	stdy= 20
	threshreg= 0.3
	start=3
	group=1500


####### File selected and now creating images out of it 
	for counter in range(start,mt.floor(len(output_orig['t'])/group)):

		image_orig = [[0 for i in range(128)] for j in range(128)]    ############## Image constructed from original data for events corresponding to last timestamp of filtered events selected

		for i in range((counter-1)*group,(counter)*group):
			image_orig[output_orig['y'][i]][output_orig['x'][i]] = 1


		comx,comy=getcom(image_orig)
		total_events=0

		for i in range(128):
			for j in range(128):
				total_events+=image_orig[i][j]

		thresh = 0.8
		threshed=total_events/(128*128)

		noofevents=0    ########## number of events present in current bounding box
		noofeventsreq= int(thresh*total_events)  #####  number of events that should be present in the bounding box
	
		upper_bound=0.95*total_events
		leftx,rightx,bottomy,topy=createboundingbox(image_orig,noofeventsreq,threshed,comx,comy,upper_bound)
		

		image_orig=np.array(image_orig)
		image_orig=image_orig.astype(np.double)
		# cv2.rectangle(image_orig,(leftx,bottomy),(rightx,topy),(100,100,100),1)
		# cv2.imshow('image',image_orig)
		# cv2.waitKey(0)

		croppedimage= crop(image_orig,leftx,rightx,bottomy,topy)
		sizex=rightx-leftx
		sizey=topy-bottomy


		noofeventsorig=0.0
		for i in range(sizex):
			for j in range(sizey):
				noofeventsorig+=croppedimage[j][i]

		multiplierx=mt.floor(sizex/4)
		multipliery=mt.floor(sizey/4)

		for i in range(1,4):
			for j in range(2):
				if(j==0):
					trainimage=[[croppedimage[p][q] for q in range(i*multiplierx)] for p in range(sizey)]
				else:
					trainimage=[[croppedimage[p][q] for q in range(i*multiplierx,sizex)] for p in range(sizey)]

				trainimage=np.array(trainimage)
				noofeventsreg=np.sum(trainimage)
				overlap = noofeventsreg/noofeventsorig

				if(overlap>0.9):

					trainimage=padwithzeros(trainimage)
					trainimage=preprocess3(trainimage)

					mydict={}
					mydict['data']=trainimage
					mydict['label']=smoothdict[int(grasptype)]		
					f = open(mytraingraspdata+winubu+grasptype+objecttype+str(listofgrasp[tmp]),"wb")
					listofgrasp[tmp] += 1
					pickle.dump(mydict,f)
					f.close()

		for i in range(1,4):
			for j in range(2):
				if(j==0):
					trainimage=[[croppedimage[p][q] for q in range(sizex)] for p in range(i*multipliery)]
				else:
					trainimage=[[croppedimage[p][q] for q in range(sizex)] for p in range(i*multipliery,sizey)]
				trainimage=np.array(trainimage)
				noofeventsreg=np.sum(trainimage)
				overlap = noofeventsreg/noofeventsorig

				
				if(overlap>0.9):

					trainimage=padwithzeros(trainimage)
					trainimage=preprocess3(trainimage)


					mydict={}
					mydict['data']=trainimage
					mydict['label']=smoothdict[int(grasptype)]
					f = open(mytraingraspdata+winubu+grasptype+objecttype+str(listofgrasp[tmp]),"wb")
					listofgrasp[tmp] += 1
					pickle.dump(mydict,f)
					f.close()


		trainingimage=padwithzeros(croppedimage)
		trainingimage=preprocess3(trainingimage)
		# cv2.imshow('image',trainingimage)
		# cv2.waitKey(0)


		mydict={}
		mydict['data']=trainingimage
		mydict['label']=smoothdict[int(grasptype)]			
		f = open(mytraingraspdata+winubu+grasptype+objecttype+str(listofgrasp[tmp]),"wb")
		listofgrasp[tmp] += 1
		pickle.dump(mydict,f)
		f.close()