import hickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
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
batch_size=64

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def createModel():
	model = Sequential()
	model.add(Convolution2D(16, 3, 3, dim_ordering="th", input_shape=(3, 160, 90), activation='relu'))
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
	numEpochs = 20
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	print(model.summary())

	#Training model
	history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), validation_data=(X_valid, Y_valid),steps_per_epoch=X_train.shape[0]/batch_size,
                        epochs=numEpochs,)

	#Final result
	scores = model.evaluate(X_valid, Y_valid, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	return history, scores

def loadData():
	X=hickle.load("Database/database.hf5")
	X=X.reshape(np.shape(X)[0],3,160,90)
	indices=np.arange(np.shape(X)[0])
	np.random.shuffle(indices)
	X=X[indices,:,:,:]
	Y=hickle.load("Database/labels.hf5")
	Y=Y[indices,:]

	#split:70-30
	split=0.7

	X_train=X[0:int(np.shape(X)[0]*split),:,:,:]
	X_valid=X[int(np.shape(X)[0]*split):,:,:,:]
	Y_train=Y[0:int(np.shape(Y)[0]*split),:]
	Y_valid=Y[int(np.shape(Y)[0]*split):,:]

	print("Number of training images:"+str(np.shape(X_train)[0]))
	print("Number of validation images:"+str(np.shape(X_valid)[0]))

	global datagen

	datagen.fit(X_train)

	return X_train,X_valid,Y_train,Y_valid

def plotStat(history, scores):
	plt.plot(history.history['acc'],'r')
	plt.plot(history.history['valid'],'b')

def main():
	X_train,X_valid,Y_train,Y_valid=loadData()
	model=createModel()
	history,scores=compileModel(model, X_train, Y_train, X_valid, Y_valid)
	#plotStat(history, scores)

	hickle.dump(history.history, 'history.hf5', mode='w')
	hickle.dump(scores, 'scores.hf5', mode='w')
	
	model.save('Models/model.h5')
if __name__ == '__main__':
	main()
