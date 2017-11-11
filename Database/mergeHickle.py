import numpy as np
import sys
import os
import hickle

arjun=hickle.load('Arjun7.hf5')[0:3000,:,:,:]
print("da")
prakash=hickle.load('prakash8.hf5')[0:3000,:,:,:]
print("da")
kiyer=hickle.load('kiyer_frames8.hf5')[0:3000,:,:,:]
print("da")
kygandomi=hickle.load('kygandomi_frames9.hf5')[0:3000,:,:,:]
print("da")
shubham=hickle.load('shubham7.hf5')[0:3000,:,:,:]
print("da")

database=np.concatenate((arjun,prakash,kiyer,kygandomi,shubham), axis=0)
labels=np.zeros((15000,5))
labels[0:3000,:]=np.eye(5)[0,0:]
labels[3000:6000,:]=np.eye(5)[0,0:]
labels[6000:9000,:]=np.eye(5)[0,0:]
labels[9000:12000,:]=np.eye(5)[0,0:]
labels[12000:15000,:]=np.eye(5)[0,0:]

hickle.dump(database, 'database.hf5', mode='w')
