import keras
from keras.models import load_model,model_from_json
import os
from os import listdir
import numpy as np
import pickle
from keras.optimizers import SGD
import keras.backend as K
from keras import metrics
import myread as load
import cv2
import math as mt



def customloss(y_true,y_pred):
	mult=K.max(y_true,axis=-1)
	multi=K.expand_dims(mult,-1)
	multi=K.repeat_elements(multi,18,-1)
	y_pred= y_pred*multi
	y_true = K.clip(y_true, K.epsilon(), 1)
	div=K.max(y_true,axis=-1)
	y_pred = K.clip(y_pred, K.epsilon(), 1)
	return (K.sum(y_true*K.abs(K.log(y_pred/y_true)), axis=-1))/div


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


def crop(image,leftx,rightx,bottomy,topy):
	newimage=[[image[i][j] for j in range(leftx,rightx)] for i in range(bottomy,topy)]
	return newimage


def preprocess3(image):
	image=np.array(image)
	newimage=cv2.resize(image,(64,64),interpolation = cv2.INTER_CUBIC)
	return newimage



# testingfolder="testingnew1500graspdata24thjuneonly1064bicubic"
# testlist = listdir(testingfolder)
# x_test= np.zeros((len(testlist),64,64,1))
# y_test=np.zeros((len(testlist),6))
# winubu='\\'
# dim=64

# for i in range(len(testlist)):
# 	file= open(testingfolder+winubu+testlist[i],'rb')
# 	object_file=pickle.load(file)
# 	file.close()
# 	x_test[i]=object_file['data'].reshape(dim,dim,1)
# 	y_test[i]=object_file['label']

def custommetric(y_true,y_pred):
	return metrics.top_k_categorical_accuracy(y_pred, y_true, k=2)



# keras.losses.customloss = customloss
# sgd = SGD(lr=0.01, decay=.003, momentum=0.9, nesterov=True)

model = load_model('objectnessnetworkiou_2objects_nearest.08.hdf5')
# model.compile(loss=customloss, optimizer=sgd, metrics=['accuracy'])

model_json = model.to_json()
with open("objectnessnetworkiou_2objects_nearest.08.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("objectnessnetworkiou_2objects_nearest.08.h5")
print("Saved model to disk")


# json_file= open("graspnetwork.05.json",'r')
# loaded_model_json= json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("graspnetwork.05.h5")
# print("Loaded model from disk")
# loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[custommetric])

# score, accuracy= loaded_model.evaluate(x_test, y_test, batch_size=32)

# print(score,accuracy)


####################################


# mypath= "trainingnew1500graspdata24thjuneonly1064bicubic"
# myfilespath = glob.glob(mypath+'/*', recursive=True)

# for filepath in myfilespath:

# 	filename = os.path.basename(filepath)
# 	print(filename)
# 	output_orig = load.xypt(filepath)
# 	output_orig['t'] = np.array(output_orig['t'])
# 	output_orig['t'] = output_orig['t']-output_orig['t'][0]
# 	start=3
# 	group =1500

# 	for counter in range(start,mt.floor(len(output_orig['t'])/group)):
# 		image_orig = [[0 for i in range(128)] for j in range(128)]    ############## Image constructed from original data for events corresponding to last timestamp of filtered events selected

# 		for i in range((counter-1)*group,(counter)*group):
# 			image_orig[output_orig['y'][i]][output_orig['x'][i]] = 1

# 		image_orig=np.array(image_orig)
# 		image_orig=image_orig.astype(np.double)
# 		for run in range(5):
# 			r = cv2.selectROI(image_orig)
# 			bottomy=int(r[1])
# 			topy=int(r[1]+r[3])
# 			leftx=int(r[0])
# 			rightx=int(r[0]+r[2])
# 			if(leftx>=5):
# 				leftx=leftx-5
# 			if(rightx<=122):
# 				rightx+=5
# 			if(bottomy>=5):
# 				bottomy=bottomy-5
# 			if(topy<=122):
# 				topy+=5

# 			croppedimage=crop(image_orig,leftx,rightx,bottomy,topy)
# 			trainingimage=padwithzeros(croppedimage)
# 			trainingimage=preprocess3(trainingimage)
# 			cv2.imshow('image',trainingimage)
# 			cv2.waitKey(0)

# 			xte= np.empty((1,dim,dim,1))
# 			xte[0]=trainingimage.reshape(64,64,1)

# 			inp = loaded_model.input                                           # input placeholder
# 			outputs = [layer.output for layer in loaded_model.layers]          # all layer outputs
# 			functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
# 			# Testing
# 			test = xte
# 			layer_outs = [func([test]) for func in functors]
# 			print(layer_outs[-1][0][0])