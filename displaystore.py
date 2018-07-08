import cv2
import myread as mrt
import filtertd as flt
import numpy as np
import math as mt
import scipy.ndimage.filters
import glob
import os
import pickle

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


def display(grasp,name):

	mylist = ["pos1f_sp1", "pos1b_sp1", "pos1f_sp2", "pos1b_sp2","pos1f_sp3", "pos1b_sp3",
	"pos2f_sp1", "pos2b_sp1", "pos2f_sp2", "pos2b_sp2", "pos2f_sp3", "pos2b_sp3", 
	"pos3f_sp1", "pos3b_sp1", "pos3f_sp2", "pos3b_sp2", "pos3f_sp3", "pos3b_sp3", 
	"pos4f_sp1", "pos4b_sp1", "pos4f_sp2", "pos4b_sp2", "pos4f_sp3", "pos4b_sp3",
	"pos5f_sp1", "pos5b_sp1", "pos5f_sp2", "pos5b_sp2", "pos5f_sp3", "pos5b_sp3",
	"pos6f_sp1", "pos6b_sp1", "pos6f_sp2", "pos6b_sp2", "pos6f_sp3", "pos6b_sp3",
	"pos7f_sp1", "pos7b_sp1", "pos7f_sp2", "pos7b_sp2", "pos7f_sp3", "pos7b_sp3",
	"pos8f_sp1", "pos8b_sp1", "pos8f_sp2", "pos8b_sp2", "pos8f_sp3", "pos8b_sp3", 
	"pos9f_sp1", "pos9b_sp1", "pos9f_sp2", "pos9b_sp2", "pos9f_sp3", "pos9b_sp3"]

	index=0

	while(index<len(mylist)):
		print(mylist[index])
		filepath="trainingnewdata\\"+"Grasp"+grasp+name+mylist[index]+".aedat"
		output_orig = mrt.xypt(filepath)
		output_orig['t'] = np.array(output_orig['t'])
		output_orig['t'] = output_orig['t']-output_orig['t'][0]


		start=3
		group=1500

	####### File selected and now creating images out of it 
		for counter in range(start,min(30,mt.floor(len(output_orig['t'])/group))):

			image_orig = [[0 for i in range(128)] for j in range(128)]    ############## Image constructed from original data for events corresponding to last timestamp of filtered events selected

			
			for i in range((counter-1)*group,(counter)*group):
				image_orig[output_orig['y'][i]][output_orig['x'][i]] = 1



			image_orig=np.array(image_orig)
			image_orig=image_orig.astype(np.double)
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
			cv2.rectangle(image_orig,(leftx,bottomy),(rightx,topy),(100,100,100),1)
			cv2.imshow('image',image_orig)
			cv2.waitKey(0)

		index+=2




# display("3","Screwdriver1")


mypath= "testingnew1500graspdata24thjuneonly1064bicubic"
myfilespath = glob.glob(mypath+'/*', recursive=True)
winubu='\\'

##########################################


for filepath in myfilespath:
	filename = os.path.basename(filepath)
	print(filename)
	file= open(filepath,'rb')
	object_file=pickle.load(file)
	file.close()
	cv2.imshow('image',object_file['data'])
	cv2.waitKey(0)