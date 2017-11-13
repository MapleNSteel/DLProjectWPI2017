import numpy as np
import sys
import os
import hickle
import cv2

arjun=hickle.load('Arjun7.hf5')
print("da")
prakash=hickle.load('prakash8.hf5')
print("da")
kiyer=hickle.load('kiyer_frames8.hf5')
print("da")
kygandomi=hickle.load('kygandomi_frames9.hf5')
print("da")
shubham=hickle.load('shubham7.hf5')
print("da")

database=np.concatenate((arjun,prakash,kiyer,kygandomi,shubham), axis=0)
labels=np.zeros((np.shape(arjun)[0]+np.shape(prakash)[0]+np.shape(kiyer)[0]+np.shape(kygandomi)[0]+np.shape(shubham)[0],5))
labels[0:np.shape(arjun)[0],:]=np.eye(5)[0,0:]
labels[np.shape(arjun)[0]:np.shape(prakash)[0],:]=np.eye(5)[0,0:]
labels[np.shape(prakash)[0]:np.shape(kiyer)[0],:]=np.eye(5)[0,0:]
labels[np.shape(kiyer)[0]:np.shape(kygandomi)[0],:]=np.eye(5)[0,0:]
labels[np.shape(kygandomi)[0]:np.shape(shubham)[0],:]=np.eye(5)[0,0:]

hickle.dump(database, 'database.hf5', mode='w')
hickle.dump(labels, 'labels.hf5', mode='w')
