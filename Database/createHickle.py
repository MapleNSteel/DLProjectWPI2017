from tempfile import mkdtemp
import numpy as np
import cv2
import sys
import os
import hickle

def getImageFiles(directory):
	imageList=[]
	num=0
	for file in os.listdir(directory):
		if file.endswith(".jpg"):
			print(num)
			num=num+1
			frame=cv2.imread(os.path.join(directory, file))
			imageList.append(frame)

	frames=np.array(imageList)
	hickle.dump(frames, directory+'/'+directory+str(int(num/1000))+'.hf5', mode='w')
	imageList=[]

def main():

	addresses=['shubham','Arjun','kygandomi_frames','prakash', 'kiyer_frames']
	for address in addresses:
		print(address)
		getImageFiles(address)

if __name__ == '__main__':
	main()
