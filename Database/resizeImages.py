import numpy as np
import cv2
import sys
import os
num=0
def getImageFiles(directory):
	imageList=[]
	for file in os.listdir(directory):
	    if file.endswith(".jpg"):
	        imageList.append(os.path.join(directory, file))
	return imageList

def resizeImage(address):
	frame=cv2.imread(address)
	frame= cv2.resize(frame, (180,320), interpolation = cv2.INTER_CUBIC)
	cv2.imwrite(address,frame)
def main():

	directories=[name for name in os.listdir(".") if os.path.isdir(name)]
	print(directories)
	for directory in directories:
		addresses=getImageFiles(directory)
		for address in addresses:
			print(address)
			frames=resizeImage(address)
		

if __name__ == '__main__':
	main()
	
