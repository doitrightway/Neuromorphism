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
import objectnessgenerator


if __name__ ==  '__main__':


	trainingfolder="trainingobjectnessdataiou_random_boundingboxes"
	winubu='\\'
	filelist = listdir(trainingfolder)
	dim=64

	L=len(filelist)
	noofsamples=L
	mylist = random.sample(range(L),noofsamples)

	testingfolder="testingobjectnessdataiou_random_boundingboxes"
	testlist = listdir(testingfolder)
	x_test= np.zeros((len(testlist),64,64,1))
	y_test=np.zeros((len(testlist),2))

	for i in range(len(testlist)):
		file= open(testingfolder+winubu+testlist[i],'rb')
		object_file=pickle.load(file)
		file.close()
		data=np.array(object_file['data'])
		x_test[i]=data.reshape(dim,dim,1)
		one_hot_labels = keras.utils.to_categorical(object_file['label'], num_classes=2)
		y_test[i]=one_hot_labels


	training_data= objectnessgenerator.DataGenerator(filelist,mylist,trainingfolder,winubu,dim)


	model = Sequential()


	model.add(Conv2D(32,(6,6),activation = 'relu', input_shape = (64,64,1), kernel_initializer=initializers.RandomNormal(stddev=0.001)))
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

	model.add(Flatten())
	model.add(Dense(2, activation = 'softmax',kernel_initializer=initializers.RandomNormal(stddev=0.001)))


	checkpoint = keras.callbacks.ModelCheckpoint('objectnessnetworkiou_random_boundingboxes.{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

	sgd = SGD(lr=0.01, decay=.003, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	model.fit_generator(generator=training_data ,epochs=10,validation_data = (x_test, y_test), callbacks=[checkpoint],use_multiprocessing=True)
	score, accuracy= model.evaluate(x_test, y_test, batch_size=32)
	print (score)
	print (accuracy)