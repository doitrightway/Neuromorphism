import numpy as np
import keras
import os
import pickle

class DataGenerator(keras.utils.Sequence):
	def __init__(self,files,traininglist,trainingfolder, winubu, dim=64, batch_size=32,shuffle=True):
		self.files=files
		self.traininglist=traininglist
		self.trainingfolder=trainingfolder
		self.winubu=winubu
		self.dim=dim
		self.batch_size=batch_size
		self.shuffle=shuffle
		self.featurewise_center=False,
		self.featurewise_std_normalization=False,
		self.width_shift_range=0.15,
		self.height_shift_range=0.15,
		# self.zoom_range= 0.1,
		# self.horizontal_flip=True,
		self.on_epoch_end()

	def __len__(self):
		return int(np.floor(len(self.traininglist)/self.batch_size))

	def __getitem__(self,index):
		indexes= self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		list_ids= [self.traininglist[k] for k in indexes]

		X, y= self.__data_generation(list_ids)

		return X, y

	def on_epoch_end(self):
		self.indexes=np.arange(len(self.traininglist))
		if self.shuffle== True:
			np.random.shuffle(self.indexes)

	def __data_generation(self,list_ids):

		X= np.empty((self.batch_size,self.dim,self.dim,1))
		y= np.empty((self.batch_size,6))

		for i,ID in enumerate(list_ids):
			file= open(self.trainingfolder+self.winubu+self.files[ID],'rb')
			object_file=pickle.load(file)
			file.close()
			X[i]=object_file['data'].reshape(self.dim,self.dim,1)
			y[i]=np.array(object_file['label'])

		return X,y