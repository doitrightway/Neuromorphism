import myread as load
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

def filtertd(file):

	output = load.xypt(file)

	x=[]
	y=[]
	t=[]
	p=[]

	time_maintain= [[(np.inf)*0 for i in range(128)] for j in range(128)]

	output['t']= np.array(output['t'])
	output['t']=output['t']-output['t'][0]

	thresh = 50000

	for i in range(len(output['p'])):
		coord_x=output['x'][i]
		coord_y=output['y'][i]
		mini=time_maintain[coord_x][coord_y]
		for u in [-1,0,1]:
			for v in [-1,0,1]:
				if(coord_x+u >=0 and coord_x+u<128 and coord_y+v>=0 and coord_y+v < 128):
					mini=max(mini,time_maintain[coord_x+u][coord_y+v])

		if(np.isnan(mini) or output['t'][i]-mini < thresh):
			time_maintain[coord_x][coord_y]=output['t'][i]
			if not np.isnan(mini):
				x.append(coord_x)
				y.append(coord_y)
				p.append(output['p'][i])
				t.append(output['t'][i])

	output={}
	output['x']=x
	output['y']=y
	output['t']=t
	output['p']=p
	return output