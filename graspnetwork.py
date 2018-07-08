import cv2
import os
from os import listdir
import scipy.io
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, initializers
from keras.optimizers import SGD
from keras.utils import plot_model
import pickle
import tensorflow as tf
import math as mt
import keras.backend as K
import os.path
import random
from keras.models import model_from_json
import Datagengrasp


if __name__ ==  '__main__':


	trainingfolder="trainingnew1500graspdata24thjuneonly1064bicubic"
	winubu='\\'  ############# for changing from windows to ubuntu or mac
	filelist = listdir(trainingfolder)
	dim=64

	L=len(filelist)
	noofsamples=L
	mylist = random.sample(range(L),noofsamples)
	

	testingfolder="testingnew1500graspdata24thjuneonly1064bicubic"
	testlist = listdir(testingfolder)
	x_test= np.zeros((len(testlist),64,64,1))
	y_test=np.zeros((len(testlist),6))

	for i in range(len(testlist)):
		file= open(testingfolder+winubu+testlist[i],'rb')
		object_file=pickle.load(file)
		file.close()
		x_test[i]=object_file['data'].reshape(dim,dim,1)
		y_test[i]=np.array(object_file['label'])

	training_data= Datagengrasp.DataGenerator(filelist,mylist,trainingfolder,winubu)

	model = Sequential()

	model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = (64,64,1), kernel_initializer=initializers.RandomNormal(stddev=0.001)))
	model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.1))
	model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001)))
	model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.1))

	model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001)))

	model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.1))

	model.add(Conv2D(256, (3,3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.001)))

	#model.add(Dropout(0.1))
	model.add(BatchNormalization())

	model.add(Flatten())
	model.add(Dense(6, activation = 'softmax',kernel_initializer=initializers.RandomNormal(stddev=0.001)))

	# plot_model(model, to_file='Model_0.1_6x6.png',show_shapes = True)

	checkpoint = keras.callbacks.ModelCheckpoint('graspnetwork.{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

	sgd = SGD(lr=0.01, decay=.003, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	model.fit_generator(generator=training_data ,epochs=10,validation_data = (x_test,y_test), callbacks=[checkpoint],use_multiprocessing=True)
	score, accuracy= model.evaluate(x_test, y_test, batch_size=32)
	print (score)
	print (accuracy)