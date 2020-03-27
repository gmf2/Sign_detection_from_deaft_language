import numpy as np
import math

# PATHS
#DATA_DIR: Folder where is located label.csv
#POSE_DIR= Folder that provide the pose train data
#PLOT_DIR =Folder that will store all the plots corresponding the displacement 
#           of the average keypoints of the hands per sample
DATA_DIR = '../data'
POSE_DIR = '../data/pose'
    

#HELP FUNCTION
#It calculates the deuclidean distances between 2 points
def calculateDistance(x1,y1,x2,y2):
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     return dist

#Total motion per sample.: It calculates the total difference between frames per sample
def analyze_motion(all_samples,keyPoint):
    #ArrayMotion
    #    1 dimension: Number of samples
    #    2 dimension: Total motion per sample
    array_motion=np.zeros((len(all_samples)))
    #IndexSample
    index_sample=0
    for i,name in enumerate(listdir(POSE_DIR+'/train')):
        #Initial index
        indexFrame=0
        
        mat= np.load(pjoin(POSE_DIR,'train', name))
        #Per frame in each sample
        indexFrame=0
        for indexFrame in range (0,mat.shape[0]):
            #Initial Index
            if indexFrame>=1:
                if mat[indexFrame-1][keyPoint][2]>0.4:
                #Calculate distance between this frame and the previous one
                    array_motion[index_sample]+=calculateDistance(mat[indexFrame-1][keyPoint][0],mat[indexFrame][keyPoint][0],mat[indexFrame-1][keyPoint][1],mat[indexFrame][keyPoint][1])
        index_sample+=1
    return array_motion
    
#Total motion per sample.: It calculates the total difference between frame 1 and FINAL frame per sample
def initial_Fin_Motion(all_samples,keyPoint):
    #ArrayMotion
    #    1 dimension: Number of samples
    #    2 dimension: Total motion per sample
    array_motion=np.zeros((len(all_samples)))
    #IndexSample
    index_sample=0
    for i,name in enumerate(listdir(POSE_DIR+'/train')):
        #Initial index
        mat= np.load(pjoin(POSE_DIR,'train', name))
        #Calculate distance between  frame 1 and the previous one
        array_motion[index_sample]=calculateDistance(mat[0][keyPoint][0],mat[mat.shape[0]-1][keyPoint][0],mat[0][keyPoint][1],mat[mat.shape[0]-1][keyPoint][1])
        index_sample+=1
    return array_motion