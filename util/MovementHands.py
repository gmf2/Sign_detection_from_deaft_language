# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:15:02 2019

@author: Gabriel
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:05:53 2019

@author: Gabriel
"""

# REQUIRED IMPORTS FROM STANDARD PACKAGES
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
#%matplotlib inline

import csv
import random

from os.path import join as pjoin
from glob import glob
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# PATHS
DATA_DIR = 'C:/Users/Gabriel/Documents/Universidad/UGENT/Asignaturas/Machine Learning/Competition/Phase 1/data'
POSE_DIR = 'C:/Users/Gabriel/Documents/Universidad/UGENT/Asignaturas/Machine Learning/Competition/Phase 1/data/pose'
POSE_TRAIN_DIR = 'C:/Users/Gabriel/Documents/Universidad/UGENT/Asignaturas/Machine Learning/Competition/Phase 1/data/pose/train'
EXTRACT_DIR='C:/Users/Gabriel/Documents/Universidad/UGENT/Asignaturas/Machine Learning/Competition/Phase 1/data/extract/train/'
PLOTS='C:/Users/Gabriel/Documents/Universidad/UGENT/Asignaturas/Machine Learning/Competition/Phase 1/data/movement/train/'

# IMPORTS FROM THE UTIL LIBRARY PROVIDED BY US
import util.submission as S
import util.vis as V
import util.metrics as M
#df = pd.DataFrame(datos)
for i,name in enumerate(listdir(EXTRACT_DIR)):
    #KeyPoints Extracted: Total=60
    #0-8 Pose
    #9-13 Nose
    #14-18 Mouth
    #19-20 Eyes
    #21-62 Hands
    indexFrame=0
    indexKeyPoint=0
    mat= np.load(pjoin(EXTRACT_DIR, name))
#    1 dimension: Number of keyPoints
#    2 dimension: Number of frames per sample
#    3 dimension: x and y per Key Point
    movement=np.zeros((43,mat.shape[0],2))

#    mat.append(np.load(pjoin(POSE_DIR, 'train/train_01181.npy')))  
    indexFrame=0
    for frame in mat:
        indexKeyPoint=0
        indexNewKeyPoint=0
        print(frame.shape)
        for keyPoint in frame:
            #Hands
            if indexKeyPoint in range(21,63):
#                print("range95.136" .format(indexKeyPoint))
                movement[indexNewKeyPoint][indexFrame][0]=keyPoint[0]
                movement[indexNewKeyPoint][indexFrame][1]=keyPoint[1]
                indexNewKeyPoint=indexNewKeyPoint+1
            indexKeyPoint=indexKeyPoint+1
        indexFrame=indexFrame+1
        
#    Save array in PLOTS
    #np.save(PLOTS+name,movement)
    indexKey=0
    for keyPoint in movement:
        #PLOT 1 key point in a 1 sample
        fig = plt.figure()
        
        #X labels
        plt.subplot(1,2,1)
        datax=np.zeros((keyPoint.shape[0]))
        for i in range(keyPoint.shape[0]):
            datax[i]=keyPoint[i][0]
        plt.plot(datax)
        plt.ylabel("X")
        plt.xlabel("Frame")
        plt.title('KeyPoint movement in X:'.format(indexKey))
        plt.grid(True)
        #plt.ax.set(xlim=(0, movement[0].shape[0]), ylim=(min(datax), max(datax)))
        
        #Y labels
        plt.subplot(1,2,2)
        datay=np.zeros((keyPoint.shape[0]))
        for i in range(keyPoint.shape[0]):
            datay[i]=keyPoint[i][1]
        plt.plot(datay)
        plt.ylabel("Y")
        plt.xlabel("Frame")
        plt.title('KeyPoint movement in Y'.format(indexKey))
        plt.grid(True)
        
        
        indexKey=indexKey+1
        print(indexKey)
        #plt.ax.set(xlim=(0, movement[0].shape[0]), ylim=(min(datay), max(datax)))
        
        

            
