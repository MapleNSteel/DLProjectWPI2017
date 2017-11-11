import hickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from keras import regularizers

import matplotlib.pyplot as plt

numberOfLabels=5

def createModel():
	model = Sequential()
	model.add(Convolution2D(16, 3, 3, dim_ordering="th", input_shape=(3, 320, 180), activation='relu'))
	model.add(Convolution2D(16, 3, 3, dim_ordering="th", activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Convolution2D(32, 3, 3, dim_ordering="th", activation='relu'))
	model.add(Convolution2D(32, 3, 3, dim_ordering="th", activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))	
	model.add(BatchNormalization())
	model.add(Convolution2D(16, 3, 3, dim_ordering="th", activation='relu'))
	model.add(Convolution2D(16, 3, 3, dim_ordering="th", activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l1(1e-4)))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l1(1e-4)))
	model.add(Dense(5, activation='softmax',kernel_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l1(1e-4)))

	return model

def compileModel(model, X_train, Y_train, X_valid, Y_valid):
	#Hyper-parameters
	epochs = 100
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	print(model.summary())

	#Training model
	history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), nb_epoch=epochs, batch_size=16)

	#Final result
	scores = model.evaluate(X_test, Y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	return history, scores

def loadData():
	X=hickle.load("Database/database.hf5")
	X=X.reshape(np.shape(X)[0],3,320,180)
	indices=np.arange(np.shape(X)[0])
	np.random.shuffle(indices)
	X=X[indices,:,:,:]
	Y=hickle.load("Database/labels.hf5")
	Y=Y[indices,:]

	#split:70-30

	X_train=X[0:int(np.shape(X)[0]*0.7),:,:,:]
	X_valid=X[int(np.shape(X)[0]*0.7):,:,:,:]
	Y_train=Y[0:int(np.shape(Y)[0]*0.7),:]
	Y_valid=Y[int(np.shape(Y)[0]*0.7):,:]

	return X_train,X_valid,Y_train,Y_valid

def plotStat(history, scores):
	plt.plot(history.history['acc'],'r')
	plt.plot(history.history['valid'],'b')

def main():
	X_train,X_valid,Y_train,Y_valid=loadData()
	model=createModel()
	history,scores=compileModel(model, X_train, Y_train, X_valid, Y_valid)
	plotStat(history, scores)
	
	model.save('Models/model.h5')
if __name__ == '__main__':
	main()
