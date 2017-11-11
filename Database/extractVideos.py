import numpy as np
import cv2
import sys
import os
num=0
def getImageFiles(directory):
	videoList=[]
	for file in os.listdir(directory):
	    if file.endswith(".mp4"):
	        videoList.append(os.path.join(directory, file))
	return videoList

def readFrames(address):
	print(address)
	global num
	frames=[]
	cap = cv2.VideoCapture(address)
	while(cap.isOpened()):
  		ret, frame = cap.read()
		if(not ret):
			break
		cv2.imwrite(sys.argv[1]+str(num)+'.jpg',frame)
		num=num+1
	return np.array(frames)

def main():

	addresses=getImageFiles(sys.argv[1])
	global num
	num=0
	for address in addresses:
		frames=readFrames(address)
		

if __name__ == '__main__':
	main()
	
