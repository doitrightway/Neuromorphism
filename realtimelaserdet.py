import myread as load
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import math as mt
import time


def checkneighbourhood(centerx,centery,state,prev,coordx,coordy,polarity,tim):
	for u in [-1,0,1]:
		for v in [-1,0,1]:
			if(centerx+u>=0 and centerx+u<128 and centery+v>=0 and centery+v<128):
				if(polarity*state[centery+v][centerx+u]==-1 and tim-prev[centery+v][centerx+u]<30000):
					return True
	return False

def laserdet(output):
	
	output['t']=np.array(output['t'])
	output['t']=output['t']-output['t'][0]

	images = [[0 for i in range(128)] for j in range(128)]
	state = [[0 for i in range(128)] for j in range(128)]
	prev = [[0 for i in range(128)] for j in range(128)]
	

	#################
	for i in range(len(output['t'])):
		coordx=output['x'][i]
		coordy=output['y'][i]
		polarity = output['p'][i]
		tim=output['t'][i]
		for u in [-1,0,1]:
			for v in [-1,0,1]:
				if(coordx+u>=0 and coordx+u<128 and coordy+v>=0 and coordy+v<128):
					if(checkneighbourhood(coordx+u,coordy+v,state,prev,coordx,coordy,polarity,tim)):
						images[coordy+v][coordx+u]+=1
		
		state[coordy][coordx]=polarity
		prev[coordy][coordx]=tim



	images=np.array(images)
	maxfindarray=images[64:128,30:90]
	maxi = np.amax(maxfindarray)
	number=0
	sumx=0
	sumy=0
	for i in range(64):
		for j in range(60):
			if(maxfindarray[i][j]==maxi):
				sumx+=j
				sumy+=i
				number+=1
	return mt.floor(sumx/number),mt.floor(sumy/number)+1