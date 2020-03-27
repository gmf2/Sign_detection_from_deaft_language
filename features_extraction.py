import numpy as np
import math
import pandas as pd
from collections import Counter




def hand_moving(samplearray):
    """
    Function that returns the hand that is moving. 1:left hand; 2:right hand; 3:both;4:undefined
    
    :param samplearray: sample array
    """
    ret_list = []
    #ArrayKeypoints
    keypoints_left_hand = [i for i in range(95, 116)]
    keypoints_right_hand = [i for i in range(116, 137)]
    #LenKeypoints
    len_keypoints_left_hand=len(keypoints_left_hand)
    len_keypoints_right_hand=len(keypoints_right_hand)
    for sample in samplearray:
        #Distances arrays
        dist_l_x = np.zeros((len(keypoints_left_hand)))
        dist_l_y = np.zeros((len(keypoints_left_hand)))
        dist_r_x = np.zeros((len(keypoints_right_hand)))
        dist_r_y = np.zeros((len(keypoints_right_hand)))
        for i in range(sample.shape[0]):
            #keypoints
            keypoints_left_hand = [i for i in range(95, 116)]
            keypoints_right_hand = [i for i in range(116, 137)]
            #index
            indexKeypoint_left=0
            indexKeypoint_right=0
            
            #left hand
            #Calculating the displacement per keypoint in one sample
            ind_left=95
            for keypoint_l in keypoints_left_hand:
                if (sample[i][ind_left][2] > 0.2 and sample[i-1][ind_left][2] > 0.2):#check confidence levels
                    dist_l_x[indexKeypoint_left] += abs(sample[i-1][keypoint_l][0] - sample[i][keypoint_l][0]) 
                    dist_l_y[indexKeypoint_left] += abs(sample[i-1][keypoint_l][1] - sample[i][keypoint_l][1]) 
                indexKeypoint_left+=1
                ind_left+=1
                
            #Right hand
            #Calculating the displacement per keypoint in one sample
            ind_right=116
            for keypoint_r in keypoints_right_hand:
                if (sample[i][ind_right][2] > 0.2 and sample[i-1][ind_right][2] > 0.2):#check confidence levels
                    dist_r_x[indexKeypoint_right] += abs(sample[i-1][keypoint_r][0] - sample[i][keypoint_r][0]) 
                    dist_r_y[indexKeypoint_right] += abs(sample[i-1][keypoint_r][1] - sample[i][keypoint_r][1]) 
                indexKeypoint_right+=1
                ind_right+=1     
        #left hand
        #Total mean displacement between all keypoints of one hand 
        #X
        mean_left_movement_x=0
        for j in range(len(dist_l_x)):
            mean_left_movement_x+=dist_l_x[j]
        mean_left_movement_x=mean_left_movement_x/len(keypoints_left_hand)

        #Y
        mean_left_movement_y=0
        for j in range(len(dist_l_y)):
            mean_left_movement_y+=dist_l_y[j]
        mean_left_movement_y=mean_left_movement_y/len(keypoints_left_hand)

        #Right hand
        #Total mean displacement between all keypoints of one hand 
        #X
        mean_right_movement_x=0
        for j in range(len(dist_r_x)):
            mean_right_movement_x+=dist_r_x[j]
        mean_right_movement_x=mean_right_movement_x/len(keypoints_right_hand)
        #Y
        mean_right_movement_y=0
        for j in range(len(dist_r_y)):
            mean_right_movement_y+=dist_r_y[j]
        mean_right_movement_y=mean_right_movement_y/len(keypoints_right_hand)  
 
        #Comparaison
        hand_l=0
        hand_r=0
        #Left Hand
        if mean_left_movement_x > 2 and mean_left_movement_y >2:
            hand_l=1
        #Right Hand
        if mean_right_movement_x > 2 and mean_right_movement_y >2:
            hand_r=1
        final=0
        #both
        if hand_l==1 and hand_r==1:
            final=3
        #left
        elif hand_l==1 and hand_r==0:
            final=1
        #right
        elif hand_l==0 and hand_r==1:
            final=2
        #Undefined
        else:
            final=4
        
        ret_list.append(final)
    #ret_list = order_features(ret_list)
    return ret_list 

def angles(sample, q0, q1, key1, key2, control): #control keypoint is the neck keypoint = 1
    """
    Function that calculates the mean angle between two keypoints. The vectors used for the calculations are respective to the
    neck.
    
    :param sample: the sample that is being featured
    :param q0: begin of the quarter frame
    :param q1: end of the quarter frame
    :param key1: number of the first keypoint
    :param key2: number of the second keypoint
    :param control: the number of the keypoint that is being used to measure the angles with. this is standard set to the neck which has keypoint number 1.
    """
    special_val_angle = math.pi/180*270
    ret_list = []
    median_angles = []
    std_angles = []
    angles = np.zeros(q1 - q0)
    for j in range(q0, q1): 
        key1_vector = sample[j][key1] - sample[j][control] #get x and y value
        key2_vector = sample[j][key2] - sample[j][control] #get x and y value
        if (np.linalg.norm(key1_vector) == 0): #Catch outliers keypoint 1
            k = j-1
            while (np.linalg.norm(key1_vector) == 0 and k>=0): #previous frames
                key1_vector = sample[k][key1] - sample[k][control]
                k -=1
            if (k == -1 and np.linalg.norm(key1_vector) == 0): #future frames
                k = j+1
                while (np.linalg.norm(key1_vector) == 0 and k<len(sample)):
                    key1_vector = sample[k][key1] - sample[k][control]
                    k+=1
        if (np.linalg.norm(key2_vector) == 0): #Catch outliers keypoint 2
            k = j-1
            while (np.linalg.norm(key2_vector) == 0 and k>=0): #previous frames
                key2_vector = sample[k][key2] - sample[k][control]
                k -=1
            if (k == -1 and np.linalg.norm(key2_vector) == 0): #future frames
                k = j+1
                while (np.linalg.norm(key2_vector) == 0 and k<len(sample)):
                    key2_vector = sample[k][key2] - sample[k][control]
                    k+=1
        key1_vector = key1_vector[:2]/np.linalg.norm(key1_vector[:2])
        key2_vector = key2_vector[:2]/np.linalg.norm(key2_vector[:2])
        angle = np.arccos(np.clip(np.dot(key1_vector, key2_vector), -1.0, 1.0))
        angles[j-q0] = angle
    if math.isnan(np.median(angles)):
        ret_list.append(special_val_angle)
    else:
        ret_list.append(np.median(angles))
    if math.isnan(np.std(angles)):
        ret_list.append(special_val_angle)
    else:
        ret_list.append(np.std(angles))
    
    return ret_list

def angle_quarters(samplearray, key1, key2, control):
    """
    Function that calculates the angle between the 2 keypoints which used the control keypoint to measure the angles with.
    
    :param samplearray: the array with all the samples in
    :param key1: the first keypoint that is being used
    :param key2: the second keypoint that is being used
    :param control: the keypoint that is being used to measure the angles against
    """
    samplearray = normalise_frames(samplearray)
    angle_list = []
    std_list = []
    for sample in samplearray:
        q0 = 0
        q1 = sample.shape[0] // 4
        q2 = sample.shape[0] // 2
        q3 = sample.shape[0] // 4 * 3
        q4 = sample.shape[0]
        lijst = [q0, q1, q2, q3, q4]
        angle_quarter_list = []
        std_quarter_list = []
        for quarter in range(1, 5): #use all the quarters
            angle, std = angles(sample, lijst[quarter - 1], lijst[quarter], key1, key2, control)
            angle_quarter_list.append(angle)
            std_quarter_list.append(std)
        angle_list.append(angle_quarter_list)
        std_list.append(std_quarter_list)
    #print(angle_list)
    #print("===========Ordering features==========")
    feat_array=[]
    feat_array2=[]
    for quarter in range(1, 5):
        temp=[]
        temp2=[]
        for i in range(len(angle_list)):
            temp.append(angle_list[i][quarter-1])
            temp2.append(std_list[i][quarter-1])
        feat_array.append(temp)
        feat_array2.append(temp2)
    #print(feat_array)
    #print("======================================")
    return feat_array, feat_array2 

def thumb_pink_switch(samplearray, finger1, finger2):
    # Function checks if thumb and pink change positions
    ret_list = []
 
    for sample in samplearray:   
        switched = False
        smaller = False
        count = 0
        for i in range(sample.shape[0]):
            confident = True
            if (sample[i][finger1][2] < 0.01 or sample[i][finger2][2] < 0.01):
                confident = False
            if (i == 0): #initalisation of function
                if (sample[i][finger1][0] < sample[i][finger2][0]):
                    smaller = True
            elif ((sample[i][finger1][0] < sample[i][finger2][0]) and not smaller and confident): # other direction
                #switched = True
                count+=1
            elif ((sample[i][finger1][0] > sample[i][finger2][0]) and smaller and confident): # other direction
                #switched = True
                count+=1
        if (count > 2):
            switched = True
        if (switched):
            ret_list.append(1.0)
        else:
            ret_list.append(0.0)
    return ret_list

def open_close_hands(samplearray, finger1, finger2):
    #function checks how many times hands touch each other
    ret_list = []
    for sample in samplearray:
        closed = False
        count = 0
        for i in range(sample.shape[0]):
            confident = True
            if (sample[i][finger1][2] < 0.1 or sample[i][finger2][2] < 0.1): #check confidence levels
                confident = False
            if (i == 0):
                if ((sample[i][finger1][0] - sample[i][finger2][0]) < 0 and ((abs(sample[i][finger1][1] - sample[i][finger2][1]) < 2))): #initalisation of function
                    closed = True
            elif (((sample[i][finger1][0] - sample[i][finger2][0]) < 0) and not closed and confident and ((abs(sample[i][finger1][1] - sample[i][finger2][1]) < 2))):
                count += 1
                closed = True
            elif (((sample[i][finger1][0] - sample[i][finger2][0]) > 0) and closed and confident and ((abs(sample[i][finger1][1] - sample[i][finger2][1]) < 2))):
                count += 1
                closed = False
        ret_list.append(count)
    return ret_list

def hands_passing_vertical(samplearray, finger1, finger2):
    #function checks how many times hands pass each other vertically
    ret_list = []
    for sample in samplearray:
        above = False
        count = 0
        for i in range(sample.shape[0]):
            confident = True
            if (sample[i][finger1][2] < 0.1 or sample[i][finger2][2] < 0.1): #check confidence levels
                confident = False
            if (i == 0):
                if ((sample[i][finger1][1] - sample[i][finger2][1]) < 0): #initalisation of function
                    above = True
            elif (((sample[i][finger1][1] - sample[i][finger2][1]) < 0) and not above and confident):
                count += 1
                above = True
            elif (((sample[i][finger1][1] - sample[i][finger2][1]) > 0) and above and confident):
                count += 1
                above = False
        ret_list.append(count)
    return ret_list

#HELP FUNCTION
def order_features(values):
    feat_array = []
    #print(len(values[0][0]), len(values), len(values[0]))
    for i in range(len(values)): #for every quarter
        temp2 = []
        for u in range(len(values[0][0])): #for every feature in the quarter
            temp = []
            for e in range(len(values[0])): #extract the right feature data 
                temp.append(values[i][e][u])   
            temp2.append(temp)
        feat_array.append(temp2)
    #print(feat_array)
    return feat_array

def mirror(sample):
    #mir = np.arange(sample.shape[0]*137*3).reshape(sample.shape[0], 137, 3) #create an empty copy
    mir=np.zeros((sample.shape[0],137,3))
    for i in range(sample.shape[0]):
        for e in range(137):
            mir[i][e][0] = float(455) - sample[i][e][0] #mirror the x-coordinates cause input images consist of 455 by 256 pixels
            mir[i][e][1] = sample[i][e][1]
            mir[i][e][2] = sample[i][e][2]
            
    return mir #return the mirrored array

def update_minmax_left(min_lx, max_lx, min_ly, max_ly, value_x, value_y):
        if (value_x < min_lx): min_lx = value_x
        if (value_x > max_lx): max_lx = value_x
        if (value_y < min_ly): min_ly = value_y
        if (value_y > max_ly): max_ly = value_y
            
        return (min_lx, max_lx, min_ly, max_ly)
    
def update_minmax_right(min_rx, max_rx, min_ry, max_ry, value_x, value_y):
        if (value_x < min_rx): min_rx = value_x
        if (value_x > max_rx): max_rx = value_x
        if (value_y < min_ry): min_ry = value_y
        if (value_y > max_ry): max_ry = value_y
            
        return (min_rx, max_rx, min_ry, max_ry)
    
 #Function that gets the features in an ordered way
#ret[0] contains features of the 1st quarter of the frames
#ret[1] of the 2nd quarter and so on
#inside ret[0] (1st quarter) are the left hand x - left hand y - right hand x - right hand y values
def avg_reach_features(samplearray):
    #print("debugging")
    values = avg_reach_extraction(samplearray)
    values = order_features(values)
    return values

#HELP FUNCTION
def avg_reach_extraction(samplearray):
    ret_list = []
    for e in range(1, 5):
        ret_list.append(avg_reach_sample(samplearray, e))
        #print(avg_reach_sample(samplearray, e))
    return ret_list #contains all the features for all the quarters

#HELP FUNCTION
def avg_reach_sample(samplearray, quarter):
    ret_list = []
    for sample in samplearray:
        q0 = 1
        q1 = sample.shape[0] // 4
        q2 = sample.shape[0] // 2
        q3 = sample.shape[0] // 4 * 3
        q4 = sample.shape[0]
        lijst = [q0, q1, q2, q3, q4]
        ret_list.append(avg_reach_quarter(sample, lijst[quarter - 1], lijst[quarter]))
    return ret_list #contains lx, ly, rx, ry for each sample for quarter 1
    
#HELP FUNCTION
def avg_reach_quarter(sample, q1, q2):
    #input images of 455 by 256 pixels
    min_left_y = 256
    min_left_x = 455
    min_right_y = 256
    min_right_x = 455
    max_left_x = 0
    max_left_y = 0
    max_right_x = 0
    max_right_y = 0
    
    for i in range(q1, q2): #iterate over all the frames within the sample
        
        ###Calculate the average of the keypoints of the hands
        avg_left_hand_x = sum([sample[i][e][0] for e in range(95, 116)])/21
        avg_right_hand_x = sum([sample[i][e][0] for e in range(116, 137)])/21
        avg_left_hand_y = sum([sample[i][e][1] for e in range(95, 116)])/21
        avg_right_hand_y = sum([sample[i][e][1] for e in range(116, 137)])/21
        
        ###update the minima and maxima
        (min_left_x, max_left_x, min_left_y, max_left_y) = update_minmax_left(min_left_x, max_left_x, min_left_y, max_left_y, avg_left_hand_x, avg_left_hand_y)
        (min_right_x, max_right_x, min_right_y, max_right_y) = update_minmax_right(min_right_x, max_right_x, min_right_y, max_right_y, avg_right_hand_x, avg_right_hand_y)
  
    return (max_left_x - min_left_x, max_left_y - min_left_y, max_right_x - min_right_x, max_right_y - min_right_y)  
   
    
    
#NOT BEING USED ANYMORE   
def avg_reach_left_horizontal(samplearray):
    ret_list = []
    for sample in samplearray:
        ans = avg_reach(sample)
        ret_list.append(ans[1] - ans[0])
    return ret_list
def avg_reach_right_horizontal(samplearray):
    ret_list = []
    for sample in samplearray:
        ans = avg_reach(sample)
        ret_list.append(ans[5] - ans[4])
    return ret_list
def avg_reach_left_vertical(samplearray):
    ret_list = []
    for sample in samplearray:
        ans = avg_reach(sample)
        ret_list.append(ans[3] - ans[2])
    return ret_list
def avg_reach_right_vertical(samplearray):
    ret_list = []
    for sample in samplearray:
        ans = avg_reach(sample)
        ret_list.append(ans[7] - ans[6])
    return ret_list
    
#help function
def avg_reach(sample):
    #input images of 455 by 256 pixels
    min_left_y = 256
    min_left_x = 455
    min_right_y = 256
    min_right_x = 455
    max_left_x = 0
    max_left_y = 0
    max_right_x = 0
    max_right_y = 0
    
    for i in range(sample.shape[0]): #iterate over all the frames within the sample
        
        ###Calculate the average of the keypoints of the hands
        avg_left_hand_x = sum([sample[i][e][0] for e in range(95, 116)])/21
        avg_right_hand_x = sum([sample[i][e][0] for e in range(116, 137)])/21
        avg_left_hand_y = sum([sample[i][e][1] for e in range(95, 116)])/21
        avg_right_hand_y = sum([sample[i][e][1] for e in range(116, 137)])/21
        
        ###update the minima and maxima
        (min_left_x, max_left_x, min_left_y, max_left_y) = update_minmax_left(min_left_x, max_left_x, min_left_y, max_left_y, avg_left_hand_x, avg_left_hand_y)
        (min_right_x, max_right_x, min_right_y, max_right_y) = update_minmax_right(min_right_x, max_right_x, min_right_y, max_right_y, avg_right_hand_x, avg_right_hand_y)
  
    return (min_left_x, max_left_x, min_left_y, max_left_y, min_right_x, max_right_x, min_right_y, max_right_y)





def total_reach_left_horizontal(samplearray):
    ret_list = []
    for sample in samplearray:
        ans = total_reach(sample)
        ret_list.append(ans[0])
    return ret_list
def total_reach_right_horizontal(samplearray):
    ret_list = []
    for sample in samplearray:
        ans = total_reach(sample)
        ret_list.append(ans[2])
    return ret_list
def total_reach_left_vertical(samplearray):
    ret_list = []
    for sample in samplearray:
        ans = total_reach(sample)
        ret_list.append(ans[1])
    return ret_list
def total_reach_right_vertical(samplearray):
    ret_list = []
    for sample in samplearray:
        ans = total_reach(sample)
        ret_list.append(ans[3])
    return ret_list


def total_reach(sample):
    coo_list = []
    dif_list = []
    #input images of 455 by 256 pixels
    min_left_y = 256
    min_left_x = 455
    min_right_y = 256
    min_right_x = 455
    max_left_x = 0
    max_left_y = 0
    max_right_x = 0
    max_right_y = 0
    for i in range(sample.shape[0]):
        #calculate the min and maxima of the fingers
        left_point_l = min([sample[i][e][0] for e in range(95, 116)])
        right_point_l = max([sample[i][e][0] for e in range(95, 116)])
        bottom_point_l = min([sample[i][e][1] for e in range(95, 116)])
        highest_point_l = max([sample[i][e][1] for e in range(95, 116)])
        
        left_point_r = min([sample[i][e][0] for e in range(116, 137)])
        right_point_r = max([sample[i][e][0] for e in range(116, 137)])
        bottom_point_r = min([sample[i][e][1] for e in range(116, 137)])
        highest_point_r = max([sample[i][e][1] for e in range(116, 137)])
        
        (min_left_x, max_left_x, min_left_y, max_left_y) = update_minmax_left(min_left_x, max_left_x, min_left_y, max_left_y, left_point_l, bottom_point_l)
        (min_left_x, max_left_x, min_left_y, max_left_y) = update_minmax_left(min_left_x, max_left_x, min_left_y, max_left_y, right_point_l, highest_point_l)
        (min_right_x, max_right_x, min_right_y, max_right_y) = update_minmax_right(min_right_x, max_right_x, min_right_y, max_right_y, left_point_r, bottom_point_r)   
        (min_right_x, max_right_x, min_right_y, max_right_y) = update_minmax_right(min_right_x, max_right_x, min_right_y, max_right_y, right_point_r, highest_point_r)
        
        #add these to the arrays
        points = (min_left_x, max_left_x, min_left_y, max_left_y, min_right_x, max_right_x, min_right_y, max_right_y)
        coo_list.append(points)
        
        differences = (points[1] - points[0], points[3] - points[2], points[5] - points[4], points[7] - points[6])
        dif_list.append(differences) 
    
    return differences

def normalise_frames(samplearray): #makes sure that every sample has at least 4 frames
    ret = []
    for sample in samplearray:
        if (sample.shape[0] == 1):
            sample = np.concatenate((sample, sample), axis=0)
            sample = np.concatenate((sample, sample), axis=0)
        elif (sample.shape[0] == 2):
            p0 = np.expand_dims(sample[0], axis=0)
            p1 = np.expand_dims(sample[1], axis=0)
            part1 = np.concatenate((p0, sample), axis=0)
            part2 = np.concatenate((part1, p1), axis=0)
            sample = part2
        elif (sample.shape[0] == 3):
            p2 = np.expand_dims(sample[2], axis = 0)
            sample = np.concatenate((sample, p2), axis=0)
        ret.append(sample)
    ret = np.array(ret)
    return ret # return the samples

def get_frames(samplearray):
    ret_list = [sample.shape[0] for sample in samplearray]
    return ret_list

##HELP FUNCTION
def keypoint_dist2(sample, key_l, key_r, q0, q1): 
    #print(q0, q1, sample.shape[0])
    ret_list = []   
    dist_l_x = 0
    dist_l_y = 0
    dist_r_x = 0
    dist_r_y = 0
    #ret_list.append(keypoint_dist(sample, key_l, key_r, q0, q1, q2, q3, q4)[0])
    for i in range(q0, q1):
        #print(i)
        if (sample[i][key_l][2] > 0.2 and sample[i-1][key_l][2] > 0.2):#check confidence levels
            dist_l_x += abs(sample[i-1][key_l][0] - sample[i][key_l][0]) 
            dist_l_y += abs(sample[i-1][key_l][1] - sample[i][key_l][1]) 
        if (sample[i][key_r][2] > 0.2 and sample[i-1][key_r][2] > 0.2):#same
            dist_r_x += abs(sample[i-1][key_r][0] - sample[i][key_r][0]) 
            dist_r_y += abs(sample[i-1][key_r][1] - sample[i][key_r][1])     
    div = q1 - q0
    if ((q1 - q0) == 0): div = 1
    distances = (dist_l_x / div, dist_l_y / div, dist_r_x / div, dist_r_y / div)
    
    ret_list.append(distances)
    #print(ret_list)
    return ret_list

##HELP FUNCTION
def keypoint_distance_sample(samplearray, key_l, key_r, quarter):
    ret_list = []
    for sample in samplearray:
        q0 = 1
        q1 = sample.shape[0] // 4
        q2 = sample.shape[0] // 2
        q3 = sample.shape[0] // 4 * 3
        q4 = sample.shape[0]
        lijst = [q0, q1, q2, q3, q4]
        ret_list.append(keypoint_dist2(sample, key_l, key_r, lijst[quarter - 1], lijst[quarter])[0])
    return ret_list #contains lx, ly, rx, ry for each sample for quarter 1

##HELP FUNCTION
def keypoint_distance(samplearray, key_l, key_r):
    ret_list = []
    for e in range(1, 5):
        ret_list.append(keypoint_distance_sample(samplearray, key_l, key_r, e))
    return ret_list #contains all the features for all the quarters

#Function that gets the features in an ordered way
#ret[0] contains features of the 1st quarter of the frames
#ret[1] of the 2nd quarter and so on
#inside ret[0] (1st quarter) are the left hand x - left hand y - right hand x - right hand y values
def keypoint_distance_features(samplearray, k1, k2):
    values = keypoint_distance(samplearray, k1, k2)
    values = order_features(values)
    return values
	





def keypoint_dist_l_x(samplearray, key_l, key_r):
    ret_list = []
    for sample in samplearray:
        ret_list.append(keypoint_dist(sample, key_l, key_r)[0])
    return ret_list
def keypoint_dist_l_y(samplearray, key_l, key_r):
    ret_list = []
    for sample in samplearray:
        ret_list.append(keypoint_dist(sample, key_l, key_r)[1])
    return ret_list
def keypoint_dist_r_x(samplearray, key_l, key_r):
    ret_list = []
    for sample in samplearray:
        ret_list.append(keypoint_dist(sample, key_l, key_r)[2])
    return ret_list
def keypoint_dist_r_y(samplearray, key_l, key_r):
    ret_list = []
    for sample in samplearray:
        ret_list.append(keypoint_dist(sample, key_l, key_r)[3])
    return ret_list


def keypoint_dist(sample, key_l, key_r):
    dist_l_x = 0
    dist_l_y = 0
    dist_r_x = 0
    dist_r_y = 0
    for i in range(1, sample.shape[0]):
        if (sample[i][key_l][2] > 0.2 and sample[i-1][key_l][2] > 0.2):
            dist_l_x += abs(sample[i-1][key_l][0] - sample[i][key_l][0]) 
            dist_l_y += abs(sample[i-1][key_l][1] - sample[i][key_l][1]) 
        if (sample[i][key_r][2] > 0.2 and sample[i-1][key_r][2] > 0.2):
            dist_r_x += abs(sample[i-1][key_r][0] - sample[i][key_r][0]) 
            dist_r_y += abs(sample[i-1][key_r][1] - sample[i][key_r][1]) 
    distances = (dist_l_x, dist_l_y, dist_r_x, dist_r_y)
    return distances

#HELP FUNCTION
#It calculates the deuclidean distances between 2 points
def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

#Total motion per sample.: It calculates the total difference between frame 1 and FINAL frame per sample
#Start: initial percentage
#Final: final percentage
def initial_Fin_Motion(all_samples,keyPoint,start,end):
    #ArrayMotion
    #    1 dimension: Number of samples
    #    2 dimension: Total motion per sample
    array_motion=np.zeros((len(all_samples)))
    #IndexSample
    index_sample=0
    for sample in all_samples:
        #Initial index
        #Calculate distance between  frame 1 and the previous one
#        print(index_sample)
        #print('sample:{}'.format(sample.shape[0]))
        #print('end:{}'.format(int(sample.shape[0]*end-1)))
        array_motion[index_sample]=calculateDistance(sample[int(sample.shape[0]*start)][keyPoint][0],sample[int(sample.shape[0]*start)][keyPoint][1],sample[math.ceil(sample.shape[0]*end-1)][keyPoint][0],sample[math.ceil(sample.shape[0]*end-1)][keyPoint][1])
        index_sample+=1
    return array_motion

#function that returns the mean and stdev of the distance between two keypoints over all frames
def keypoints_dist_mean_std(all_samples, keypoint_a, keypoint_b):
    all_samples = normalise_frames(all_samples)
    feature_list_mean = [[], [], [], []]
    feature_list_std = [[], [], [], []]
    for sample in all_samples:
        temp = []
        for frame in sample:
            #temp_x.append(abs(frame[keypoint_a][0]-frame[keypoint_b][0])) #x coord kp a - x coord kp b
            #temp_y.append(abs(frame[keypoint_a][1]-frame[keypoint_b][1])) #y coord kp a - y coord kp b
            if (frame[keypoint_a][2] == 0 or frame[keypoint_b][2] == 0):
                temp.append(float('nan'))
            else:
                temp.append(calculateDistance(frame[keypoint_a][0],frame[keypoint_a][1],frame[keypoint_b][0],frame[keypoint_b][1])**2)
        #feature_list_mean.append(np.mean(temp))
        #feature_list_std.append(np.std(temp))
        for quarter in range(4):
            start = quarter * len(sample) // 4
            stop = (quarter + 1) * len(sample) // 4
            if isNaNarray(temp[start:stop]):
                feature_list_mean[quarter].append((1000))
                feature_list_std[quarter].append((50))
            else:
                feature_list_mean[quarter].append(np.nanmean(temp[start:stop]))
                feature_list_std[quarter].append(np.nanstd(temp[start:stop]))
    return feature_list_mean, feature_list_std


##stdev1: It doesn't return better results!!
def stdev_displacement(sample, key_l, key_r, q0, q1): 
    #print(q0, q1, sample.shape[0])
    ret_list= []
    array_left_x=[]
    array_left_y=[]
    array_right_x=[]
    array_right_y=[]
    stdev_left_x=0
    stdev_left_y=0
    stdev_right_x=0
    stdev_right_y=0
    #ret_list.append(keypoint_dist(sample, key_l, key_r, q0, q1, q2, q3, q4)[0])
    for i in range(q0, q1):
        #print(i)
        if (sample[i][key_l][2] > 0.2 and sample[i-1][key_l][2] > 0.2):#check confidence levels
            array_left_x.append(abs(sample[i-1][key_l][0] - sample[i][key_l][0]))
            array_left_y.append(abs(sample[i-1][key_l][1] - sample[i][key_l][1]))
        if (sample[i][key_r][2] > 0.2 and sample[i-1][key_r][2] > 0.2):#same
            array_right_x.append(abs(sample[i-1][key_r][0] - sample[i][key_r][0]))
            array_right_y.append(abs(sample[i-1][key_r][1] - sample[i][key_r][1])) 
            
    stdev_left_x=np.std(array_left_x)
    if math.isnan(stdev_left_x):
        stdev_left_x=0
        
    stdev_left_y=np.std(array_left_y)
    if math.isnan(stdev_left_y):
        stdev_left_y=0
        
    stdev_right_x=np.std(array_right_x)
    if math.isnan(stdev_right_x):
        stdev_right_x=0
        
    stdev_right_y=np.std(array_right_y)
    if math.isnan(stdev_right_y):
        stdev_right_y=0
        
    stdevs=(stdev_left_x,stdev_left_y,stdev_right_x,stdev_right_y)
    ret_list.append(stdevs)
    #print(ret_list)
    return ret_list

##HELP FUNCTION
def stdev_distance_sample(samplearray, key_l, key_r, quarter):
    ret_list = []
    for sample in samplearray:
        q0 = 1
        q1 = sample.shape[0] // 4
        q2 = sample.shape[0] // 2
        q3 = sample.shape[0] // 4 * 3
        q4 = sample.shape[0]
        lijst = [q0, q1, q2, q3, q4]
        ret_list.append(stdev_displacement(sample, key_l, key_r, lijst[quarter - 1], lijst[quarter])[0])
    return ret_list #contains lx, ly, rx, ry for each sample for quarter 1

##HELP FUNCTION
def stdev_keypoint_distance(samplearray, key_l, key_r):
    ret_list = []
    for e in range(1, 5):
        ret_list.append(stdev_distance_sample(samplearray, key_l, key_r, e))
    return ret_list #contains all the features for all the quarters
#Function that gets the features in an ordered way
#ret[0] contains features of the 1st quarter of the frames
#ret[1] of the 2nd quarter and so on
#inside ret[0] (1st quarter) are the left hand x - left hand y - right hand x - right hand y values
def stdev_keypoint_distance_features(samplearray, k1, k2):
    values = stdev_keypoint_distance(samplearray, k1, k2)
    values = order_features(values)
    return values


#DISTANCES BETWEEN INDEX,THUMB AND CHAIN NOSE EYES MOUTH
#HELP FUNCTION
def min_dist_extraction(samplearray,keypoint_a, keypoint_b):
    ret_list = []
    for e in range(1, 5):
        ret_list.append(quarters_min_distance(samplearray, e,keypoint_a, keypoint_b))
        #print(avg_reach_sample(samplearray, e))
    return ret_list #contains all the features for all the quarters
def quarters_min_distance(samplearray, quarter,keypoint_a, keypoint_b):
    ret_list = []
    for sample in samplearray:
        q0 = 1
        q1 = sample.shape[0] // 4
        q2 = sample.shape[0] // 2
        q3 = sample.shape[0] // 4 * 3
        q4 = sample.shape[0] 
        #q5 = sample.shape[0] // 8 * 5
        #q6 = sample.shape[0] // 8 * 6
        #q7 = sample.shape[0] // 8 * 7
        #q8 = sample.shape[0] // 8
        lijst = [q0, q1, q2, q3, q4]
        ret_list.append(min_distance(sample, lijst[quarter - 1], lijst[quarter],keypoint_a, keypoint_b))
    return ret_list #contains lx, ly, rx, ry for each sample for quarter 1

def min_distance(sample,q1,q2, keypoint_a, keypoint_b):
    minDistance=456
    for i in range(q1, q2):
        #Maximum number of pixels
        distance=calculateDistance(sample[i][keypoint_a][0],sample[i][keypoint_a][1],sample[i][keypoint_b][0],sample[i][keypoint_b][1])
        if distance<minDistance:
            minDistance=distance
    if np.isnan(minDistance):
        minDistance=0
    return minDistance

#FUNCTION THAT RETURNS MINIMAL DISTANCE BETWEEN KEYPOINTS PER QUARTER
def fing_nose_eyes_chin(all_samples,keypoint_a, keypoint_b):
    values=min_dist_extraction(all_samples, keypoint_a, keypoint_b)
    return values

#HAND TOP DIRECTION
def normalise_frames_to8(sample,frames): #makes sure that every sample has at least 4 frames
    ret = []
    #Frames Needed
    frames_need=8-frames
    #print('needed:{}'.format(frames_need))
    #Last Frame to use as expansion
    frame_last=sample[frames-1]
    #print('frame_last:{}'.format(frame_last.shape))
    #Frames needed in a loop
    #print('antes:{}'.format(sample.shape))
    for i in range(frames_need):
        p0 = np.expand_dims(frame_last, axis=0)
        #print('p0:{}'.format(p0.shape))
        sample = np.concatenate((sample,p0), axis=0)
        #print('i:{}'.format(i))
    #print('después:{}'.format(sample.shape))
    #Create sample array with at least 8 frames
    ret=sample
    #print(ret.shape)
    ret = np.array(ret)
    return ret # return the samples


def finger_top_dir(sample,q1,q2):
    #left
    kp_right_center_hand=116
    kp_right_top=112
    kp_right_top2=116
    left_dir=[]
    #top
    kp_left_center_hand=95
    kp_left_top=128
    kp_left_top2=132
    right_dir=[]
    #print('q1:{}'.format(q1))
    #print('q2:{}'.format(q2))
    
    if q1==q2:
        #LEFT HAND
        right_l=0
        down_l=0
        left_l=0
        #down=1
        #print('ycenter:{}'.format(sample[i][kp_left_center_hand][1]))
        #print('ytop2:{}'.format(sample[i][kp_left_top][1]))
        #print('down:{}'.format(sample[i][kp_left_center_hand][1]-sample[i][kp_left_top][1]>8))
        if (sample[q1][kp_left_top][1]-sample[q1][kp_left_center_hand][1])>8:
            down_l=1
        #left=2
        #print('xcenter:{}'.format(sample[i][kp_left_center_hand][0]))
        #print('xtop2:{}'.format(sample[i][kp_left_top][0]))
       # print('left:{}'.format(sample[i][kp_left_center_hand][0]-sample[i][kp_left_top][0]>8))
        if (sample[q1][kp_left_center_hand][0]-sample[q1][kp_left_top][0])>8:
            left_l=1
        #right=3
        #print('xcenter:{}'.format(sample[i][kp_left_center_hand][0]))
        #print('xtop2:{}'.format(sample[i][kp_left_top][0]))
        #print('right:{}'.format((sample[i][kp_left_top][0]-sample[i][kp_left_center_hand][0])>8))
        if (sample[q1][kp_left_top][0]-sample[q1][kp_left_center_hand][0])>8:
            right_l=1       
        #print(right_l)
        #print(down_l)
        #print(left_l)
        dir_left=6
        #print('cond:{}'.format(down_l==1 and left_l==1 and right_l==0))
        #Down Right
        if (down_l==1 and right_l==1 and left_l==0):
            #print('ola1')
            dir_left=4
        #Down Left
            
        elif (down_l==1 and left_l==1 and right_l==0):
            #print('ola2')
            dir_left=5
        #Undefined
        elif (left_l==1 and right_l==1):
            #print('ola3')
            dir_left=0
        #Down
        elif (down_l==1):
            #print('ola4')
            dir_left=1
        #Left
        elif(left_l==1):
            #print('ola5')
            dir_left=2
        #Right
        elif(right_l==1):
            #print('ola6')
            dir_left=3
        
        #No Direction
        if np.isnan(dir_left):
            dir_left=6
        #Array directions
        left_dir.append(dir_left)
        #RIGHT HAND
        right_r=0
        down_r=0
        left_r=0
        #down=1
        if  (sample[q1][kp_right_top][1]-sample[q1][kp_right_center_hand][1])>8:
            down_r=1
        #left
        if (sample[q1][kp_right_center_hand][0]-sample[q1][kp_right_top][0])>8:
            left_r=1
        #right
        if (sample[q1][kp_right_top][0]-sample[q1][kp_right_center_hand][0])>8:
            right_r=1
            
        dir_right=6
        #Down Right
        if (down_r==1 and right_r==1 and left_r==0):
            dir_right=4
        #Down Left
        elif (down_r==1 and left_r==1 and right_r==0):
            dir_right=5
        #Undefined
        elif (left_r==1 and right_r==1 ):
            dir_right=0
        #Down
        elif (down_r==1):
            dir_right=1
        #Left
        elif(left_r==1):
            dir_right=2
        #Right
        elif(right_r==1):
            dir_right=3
        
        #No Direction
        if np.isnan(dir_right):
            dir_right=6
        #Arrays directions
        right_dir.append(dir_right)
    else:
        for i in range(q1, q2): #iterate over all the frames within the sample
            #LEFT HAND
            right_l=0
            down_l=0
            left_l=0
            #down=1
            #print('ycenter:{}'.format(sample[i][kp_left_center_hand][1]))
            #print('ytop2:{}'.format(sample[i][kp_left_top][1]))
            #print('down:{}'.format(sample[i][kp_left_center_hand][1]-sample[i][kp_left_top][1]>8))
            if (sample[i][kp_left_top][1]-sample[i][kp_left_center_hand][1])>8:
                down_l=1
            #left=2
            #print('xcenter:{}'.format(sample[i][kp_left_center_hand][0]))
            #print('xtop2:{}'.format(sample[i][kp_left_top][0]))
           # print('left:{}'.format(sample[i][kp_left_center_hand][0]-sample[i][kp_left_top][0]>8))
            if (sample[i][kp_left_center_hand][0]-sample[i][kp_left_top][0])>8:
                left_l=1
            #right=3
            #print('xcenter:{}'.format(sample[i][kp_left_center_hand][0]))
            #print('xtop2:{}'.format(sample[i][kp_left_top][0]))
            #print('right:{}'.format((sample[i][kp_left_top][0]-sample[i][kp_left_center_hand][0])>8))
            if (sample[i][kp_left_top][0]-sample[i][kp_left_center_hand][0])>8:
                right_l=1       
            #print(right_l)
            #print(down_l)
            #print(left_l)
            dir_left=6
            #print('cond:{}'.format(down_l==1 and left_l==1 and right_l==0))
            #Down Right
            if (down_l==1 and right_l==1 and left_l==0):
                #print('ola1')
                dir_left=4
            #Down Left

            elif (down_l==1 and left_l==1 and right_l==0):
                #print('ola2')
                dir_left=5
            #Undefined
            elif (left_l==1 and right_l==1):
                #print('ola3')
                dir_left=0
            #Down
            elif (down_l==1):
                #print('ola4')
                dir_left=1
            #Left
            elif(left_l==1):
                #print('ola5')
                dir_left=2
            #Right
            elif(right_l==1):
                #print('ola6')
                dir_left=3

            #No Direction
            if np.isnan(dir_left):
                dir_left=6
            #Array directions
            left_dir.append(dir_left)
            #RIGHT HAND
            right_r=0
            down_r=0
            left_r=0
            #down=1
            if  (sample[i][kp_right_top][1]-sample[i][kp_right_center_hand][1])>8:
                down_r=1
            #left
            if (sample[i][kp_right_center_hand][0]-sample[i][kp_right_top][0])>8:
                left_r=1
            #right
            if (sample[i][kp_right_top][0]-sample[i][kp_right_center_hand][0])>8:
                right_r=1

            dir_right=6
            #Down Right
            if (down_r==1 and right_r==1 and left_r==0):
                dir_right=4
            #Down Left
            elif (down_r==1 and left_r==1 and right_r==0):
                dir_right=5
            #Undefined
            elif (left_r==1 and right_r==1 ):
                dir_right=0
            #Down
            elif (down_r==1):
                dir_right=1
            #Left
            elif(left_r==1):
                dir_right=2
            #Right
            elif(right_r==1):
                dir_right=3

            #No Direction
            if np.isnan(dir_right):
                dir_right=6
            #Arrays directions
            right_dir.append(dir_right)
    #print('left_dir:{}'.format(left_dir))
    #print('right_dir:{}'.format(right_dir))
    #FIND THE MOST DIRECTION REPEATED
    if left_dir==[]:
        left_dir.append(0)
    if right_dir==[]:
        right_dir.append(0)    

    dir_left_mode=Counter(left_dir).most_common()[0][0]
    dir_right_mode=Counter(right_dir).most_common()[0][0]
    
    if np.isnan(dir_left_mode):
        dir_left_mode=6.0
    if np.isnan(dir_right_mode):
        dir_right_mode=6.0
    return dir_left_mode,dir_right_mode
    
#HELP FUNCTION
def split_quarters(samplearray, quarter):
    ret_list = []
    for sample in samplearray:
        #print('1:{}'.format(sample.shape[0]))
        if sample.shape[0]<8:
            sample=normalise_frames_to8(sample,sample.shape[0])
        #print('2:{}'.format(sample.shape[0]))
        q0 = 0
        q1 = sample.shape[0] // 8
        q2 = sample.shape[0] // 4
        q3 = sample.shape[0] // 8 * 3
        q4 = sample.shape[0] // 2
        q5 = sample.shape[0] // 8 * 5
        q6 = sample.shape[0] // 8 * 6
        q7 = sample.shape[0] // 8 * 7
        q8 = sample.shape[0]  
        lijst = [q0, q1, q2, q3, q4,q5,q6,q7,q8]
        ret_list.append(finger_top_dir(sample, lijst[quarter - 1], lijst[quarter]))
    return ret_list #contains lx, ly, rx, ry for each sample for quarter 1
#HELP FUNCTION
def finger_dir_extraction(samplearray):
    ret_list = []
    for e in range(1, 9):
        ret_list.append(split_quarters(samplearray, e))
        #print(avg_reach_sample(samplearray, e))
    return ret_list #contains all the features for all the quarters

def hand_dir(samplearray):
    #print("debugging")
    values = finger_dir_extraction(samplearray)
    values = order_features(values)
    return values

def fractional_move_time(samplearray, keypoint, xy, tolerance=10):
    """
    Calculate the (fractional) time a keypoint moves up/right or down/left or stays at the same position. Returns the (fractional) time of up/right movement, down/left movement and no movement.

    :param samplearray: the array with all the samples
    :param keypoint: the keypoint to track the movement
    :param xy: boolean which indicates to track either left/right (True) or up/down (False) movement
    """
    no_mov = []
    ur_mov = []
    dl_mov = []
    for sample in samplearray:
        no_movement = 0
        up_right_movement = 0
        down_left_movement = 0
        numbof_frames = len(sample)
        for i in range(1,numbof_frames):
            difference = sample[i][keypoint][0 if xy else 1] - sample[i-1][keypoint][0 if xy else 1] #calculate distance difference of 2 points in up/down direction between two adjacent frames
            if (abs(difference) <= 1):
                no_movement += 1
            elif (difference < 0):
                up_right_movement += 1
            else:
                down_left_movement += 1
        if numbof_frames > 1:
            numbof_frames -= 1
        no_mov.append(no_movement/(numbof_frames))
        ur_mov.append(up_right_movement/(numbof_frames))
        dl_mov.append(down_left_movement/(numbof_frames))
    return no_mov, ur_mov, dl_mov

def mean_stdev_absolute_position(samplearray, keypoint):
    """
    Calculate the mean and stdev of absolute (x,y) position of keypoint

    :param samplearray: the array with all the samples
    :param keypoint: the keypoint
    """
    mean_x_res = []
    stdev_x_res = [] 
    mean_y_res = []
    stdev_y_res = [] 
    for i in range(len(samplearray)):
        temp_x = np.zeros(len(samplearray[i]))
        temp_y = np.zeros(len(samplearray[i]))
        for j in range(len(samplearray[i])):
            temp_x[j] = samplearray[i][j][keypoint][0]
            temp_y[j] = samplearray[i][j][keypoint][1]
        mean_x_res.append(np.mean(temp_x))
        stdev_x_res.append(np.std(temp_x))
        mean_y_res.append(np.mean(temp_y))
        stdev_y_res.append(np.std(temp_y))
    return mean_x_res, stdev_x_res, mean_y_res, stdev_y_res

def elbow_angle(samplearray, keypoint_hand, keypoint_elbow, keypoint_shoulder):
    """
    Calculate the median and stdev of the angle between 2 vectors which represent the direction shoulder-elbow and elbow-hand

    :param samplearray: the array with all the samples
    :param keypoint_hand: pose keypoint of the hand
    :param keypoint_elbow: pose keypoint of the elbow
    :param keypoint_shoulder: pose keypoint of the shoulder
    """
    median_angle = []
    stdev_angle = []
    for sample in samplearray:
        angles = np.zeros(len(sample))
        for j in range(len(sample)):
            vector_elbow_shoulder = sample[j][keypoint_shoulder] - sample[j][keypoint_elbow]
            if np.linalg.norm(vector_elbow_shoulder[:2]) == 0: # catch outliers where x,y,c == 0,0,0
                k = j-1
                while (np.linalg.norm(vector_elbow_shoulder[:2]) == 0 and k>=0): # try to use previous frames
                    vector_elbow_shoulder = sample[k][keypoint_shoulder] - sample[k][keypoint_elbow]
                    k -= 1
                if (k==-1 and np.linalg.norm(vector_elbow_shoulder[:2]) == 0):
                    k = j+1
                    while (np.linalg.norm(vector_elbow_shoulder[:2]) == 0 and k<len(sample)): #try to use future frames
                        vector_elbow_shoulder = sample[k][keypoint_shoulder] - sample[k][keypoint_elbow]
                        k += 1
            vector_elbow_shoulder = vector_elbow_shoulder[:2] / np.linalg.norm(vector_elbow_shoulder[:2])

            vector_hand_elbow = sample[j][keypoint_hand] - sample[j][keypoint_elbow]
            if np.linalg.norm(vector_hand_elbow[:2]) == 0: # catch outliers where x,y,c == 0,0,0
                k = j-1
                while (np.linalg.norm(vector_hand_elbow[:2]) == 0 and k>=0): # try to use previous frames
                    vector_hand_elbow = sample[k][keypoint_hand] - sample[k][keypoint_elbow]
                    k -= 1
                if (k==-1 and np.linalg.norm(vector_hand_elbow[:2]) == 0):
                    k = j+1
                    while (np.linalg.norm(vector_hand_elbow[:2]) == 0 and k<len(sample)): #try to use future frames
                        vector_hand_elbow = sample[k][keypoint_hand] - sample[k][keypoint_elbow]
                        k += 1
            vector_hand_elbow = vector_hand_elbow[:2] / np.linalg.norm(vector_hand_elbow[:2])

            angle = np.arccos(np.clip(np.dot(vector_elbow_shoulder, vector_hand_elbow), -1.0, 1.0))
            angles[j] = angle
        median_angle.append(np.median(angles))
        stdev_angle.append(np.std(angles))
    return median_angle, stdev_angle
                        
def mean_stdev_distance_keypoints(samplearray, keypoint_l, keypoint_r):
    """
    Calculate the mean and stdev of the horizontal and vertical distance between two keypoints

    :param samplearray: the array with all the samples
    :param keypoint_l: the first keypoint
    :param keypoint_r: the second keypoint
    """
    ratio = []
    x_stdev = []
    y_stdev = []
    for i in range(len(samplearray)):
        sample = samplearray[i]
        ratio_temp = []
        x_temp = []
        y_temp = []
        for frame in sample:
            x = abs(frame[keypoint_l][0] - frame[keypoint_r][0])
            y = abs(frame[keypoint_l][1] - frame[keypoint_r][1])
            x_temp.append(x)
            y_temp.append(y)
            if y != 0:
                ratio_temp.append(x/y)
            else:
                ratio_temp.append(float('nan'))
        ratio.append(np.nanmean(ratio_temp))
        x_stdev.append(np.nanstd(x_temp))
        y_stdev.append(np.nanstd(y_temp))
    return ratio, x_stdev, y_stdev

def curvature_angle_hand(samplearray, hand):
    """
    Calculate the median and stddev of curvature angle of the hand

    :param samplearray: the array with all the samples
    :param hand: boolean which indicates if we want to compute the curvature angle of the left (True) hand or right (False) hand
    """
    special_val_angle = math.pi/180*270 #If no hand is present, put the angle on a special (not occuring) value: 270 degrees
    mean_confidence_fingertops = []
    stdev_confidence_fingertops = []
    median_angle = []
    stdev_angle = []
    keypoint_hand = 7 if hand else 4
    keypoint_elbow = 6 if hand else 3
    for i in range(len(samplearray)):
        sample = samplearray[i]
        angles = np.zeros(len(sample))
        confidences = np.zeros(len(sample))
        for j in range(len(sample)):
            #First determine median (x,y) position of the finger tops 
            if hand:
                start = 103
                stop = 116
            else:
                start = 124
                stop = 137
            coords_median_fingers = calc_coords_median_fingertops(sample, start, stop, j)
            vector_hand_fingers = coords_median_fingers - sample[j][keypoint_hand][:2]
            if (np.linalg.norm(vector_hand_fingers) == 0): # catch outliers where x,y,c == 0,0,0
                k = j-1
                while (np.linalg.norm(vector_hand_fingers) == 0 and k>=0): # try to use previous frames
                    coords_median_fingers = calc_coords_median_fingertops(sample, start, stop, k)
                    vector_hand_fingers = coords_median_fingers - sample[k][keypoint_hand][:2]
                    k -= 1
                if (k==-1 and np.linalg.norm(vector_hand_fingers) == 0):
                    k = j+1
                    while (np.linalg.norm(vector_hand_fingers) == 0 and k<len(sample)): #try to use future frames
                        coords_median_fingers = calc_coords_median_fingertops(sample, start, stop, k)
                        vector_hand_fingers = coords_median_fingers - sample[k][keypoint_hand][:2]
                        k += 1
            temp_norm1 = np.linalg.norm(vector_hand_fingers) 
            vector_hand_fingers = vector_hand_fingers / np.linalg.norm(vector_hand_fingers)

            vector_hand_elbow = sample[j][keypoint_elbow][:2] - sample[j][keypoint_hand][:2]
            if (np.linalg.norm(vector_hand_elbow) == 0): # catch outliers where x,y,c == 0,0,0
                k = j-1
                while (np.linalg.norm(vector_hand_elbow) == 0 and k>=0): # try to use previous frames
                    vector_hand_elbow = sample[k][keypoint_elbow][:2] - sample[k][keypoint_hand][:2]
                    k -= 1
                if (k==-1 and np.linalg.norm(vector_hand_elbow) == 0):
                    k = j+1
                    while (np.linalg.norm(vector_hand_elbow) == 0 and k<len(sample)): #try to use future frames
                        vector_hand_elbow = sample[k][keypoint_elbow][:2] - sample[k][keypoint_hand][:2]
                        k += 1
            temp_norm2 = np.linalg.norm(vector_hand_elbow)
            vector_hand_elbow = vector_hand_elbow / np.linalg.norm(vector_hand_elbow)

            if temp_norm1== 0 or temp_norm2 == 0:
                angle = special_val_angle #One hand is missing in each frame of the sample
            else:
                angle = np.arccos(np.clip(np.dot(vector_hand_fingers, vector_hand_elbow), -1.0, 1.0))
            angles[j] = angle

            confidences[j] = calc_mean_confidence_fingertops(sample, start, stop, j)
        mean_confidence_fingertops.append(np.mean(confidences))
        stdev_confidence_fingertops.append(np.std(confidences))
        median_angle.append(np.median(angles))
        stdev_angle.append(np.std(angles))
    #print("-- finished --")
    return median_angle, stdev_angle, mean_confidence_fingertops, stdev_confidence_fingertops

def calc_mean_confidence_fingertops(sample, start, stop, j):
    """
    Help function to calculate the mean of the confidence of the fingertops of a hand. Used in the function curvature_angle_hand

    :param sample: a sample of the dataset
    :param start: keypoint of first fingertop
    :param stop: keypoint of last fingertop
    :param j: frame index
    """
    confidence_fingertops = []
    for keypoint_fingertop in range(start, stop, 4):
        confidence_fingertops.append(sample[j][keypoint_fingertop][2])
    return np.mean(np.array(confidence_fingertops))

def calc_coords_median_fingertops(sample, start, stop, j):
    """
    Help function to calculate the median coordinates of the fingertops of a hand. Used in the function curvature_angle_hand

    :param sample: a sample of the dataset
    :param start: keypoint of first fingertop
    :param stop: keypoint of last fingertop
    :param j: frame index
    """
    x_fingertops = []
    y_fingertops = []
    for keypoint_fingertop in range(start, stop, 4):
        x_fingertops.append(sample[j][keypoint_fingertop][0])
        y_fingertops.append(sample[j][keypoint_fingertop][1])
    return np.array([np.median(np.array(x_fingertops)), np.median(np.array(y_fingertops))])

def fractional_time_finger_curved_downwards(samplearray):
    """
    Calculate (seperately) the (fractional) time that each finger is curved downwards during a sample

    :param samplearray: the array with all the samples
    """
    curved_downwards_99 = []
    curved_downwards_103 = []
    curved_downwards_107 = []
    curved_downwards_111 = []
    curved_downwards_115 = []
    curved_downwards_120 = []
    curved_downwards_124 = []
    curved_downwards_128 = []
    curved_downwards_132 = []
    curved_downwards_136 = []
    for sample in samplearray:
        numoftimes_dw_99 = 0
        numoftimes_dw_103 = 0
        numoftimes_dw_107 = 0
        numoftimes_dw_111 = 0
        numoftimes_dw_115 = 0
        numoftimes_dw_120 = 0
        numoftimes_dw_124 = 0
        numoftimes_dw_128 = 0
        numoftimes_dw_132 = 0
        numoftimes_dw_136 = 0
        numbof_frames = len(sample)
        for i in range(numbof_frames):
            if sample[i][99][1] > sample[i][98][1]:
                numoftimes_dw_99 += 1
            if sample[i][103][1] > sample[i][102][1]:
                numoftimes_dw_103 += 1
            if sample[i][107][1] > sample[i][106][1]:
                numoftimes_dw_107 += 1
            if sample[i][111][1] > sample[i][110][1]:
                numoftimes_dw_111 += 1
            if sample[i][115][1] > sample[i][114][1]:
                numoftimes_dw_115 += 1
            if sample[i][120][1] > sample[i][119][1]:
                numoftimes_dw_120 += 1
            if sample[i][124][1] > sample[i][123][1]:
                numoftimes_dw_124 += 1
            if sample[i][128][1] > sample[i][127][1]:
                numoftimes_dw_128 += 1
            if sample[i][132][1] > sample[i][131][1]:
                numoftimes_dw_132 += 1
            if sample[i][136][1] > sample[i][135][1]:
                numoftimes_dw_136 += 1
        curved_downwards_99.append(numoftimes_dw_99/(numbof_frames))
        curved_downwards_103.append(numoftimes_dw_103/(numbof_frames))
        curved_downwards_107.append(numoftimes_dw_107/(numbof_frames))
        curved_downwards_111.append(numoftimes_dw_111/(numbof_frames))
        curved_downwards_115.append(numoftimes_dw_115/(numbof_frames))
        curved_downwards_120.append(numoftimes_dw_120/(numbof_frames))
        curved_downwards_124.append(numoftimes_dw_124/(numbof_frames))
        curved_downwards_128.append(numoftimes_dw_128/(numbof_frames))
        curved_downwards_132.append(numoftimes_dw_132/(numbof_frames))
        curved_downwards_136.append(numoftimes_dw_136/(numbof_frames))
    return curved_downwards_99, curved_downwards_103, curved_downwards_107, curved_downwards_111, curved_downwards_115, curved_downwards_120, curved_downwards_124, curved_downwards_128, curved_downwards_132, curved_downwards_136
            
def fractional_time_fingers_stretched(samplearray, tolerance=1.5):
    """
    Calculates the (fractional) time that each finger is stretched during a sample

    :param samplearray: the array with all the samples
    :param tolerance: the tolerance to determine whether the finger is stretched or not, a higher tolerance implies that a finger is more likely to be determined as stretched
    Default value of tolerance is 4, this is determined on a sample such that the fractional time equals 1 in a sample where the fingers is (almost) always stretched
    """
    keypoints_fingertops = [99, 103, 107, 111, 115, 120, 124, 128, 132, 136]
    string_keypoints_fingertops = ["99", "103", "107", "111", "115", "120", "124", "128", "132", "136"]
    time_stretched = {"99":[], "103":[], "107":[], "111":[], "115":[], "120":[], "124":[], "128":[], "132":[], "136":[]}
    mean_confidence_fingers = {"99":[], "103":[], "107":[], "111":[], "115":[], "120":[], "124":[], "128":[], "132":[], "136":[]}
    for i in range(len(samplearray)):
        sample = samplearray[i]
        time_counter = {"99":0, "103":0, "107":0, "111":0, "115":0, "120":0, "124":0, "128":0, "132":0, "136":0}
        for j in range(len(sample)):
            mean_conf = {"99":[], "103":[], "107":[], "111":[], "115":[], "120":[], "124":[], "128":[], "132":[], "136":[]}
            for kp_fingertop in keypoints_fingertops:
                x1 = sample[j][kp_fingertop-3][0]
                y1 = sample[j][kp_fingertop-3][1]
                x3 = sample[j][kp_fingertop-2][0]
                y3 = sample[j][kp_fingertop-2][1]
                x2 = sample[j][kp_fingertop][0]
                y2 = sample[j][kp_fingertop][1]
                mean_conf[str(kp_fingertop)].append(np.mean(np.array([sample[j][kp_fingertop-3][2], sample[j][kp_fingertop-2][2], sample[j][kp_fingertop][2]])))
                if (x2-x1 != 0 and y2-y1 != 0 and x3-x1 != 0 and y3-y1 != 0):
                    alpha_x = (x3-x1)/(x2-x1)
                    alpha_y = (y3-y1)/(y2-y1)
                    if abs(alpha_x - alpha_y) < tolerance: #then we assume that the finger is stretched
                        time_counter[str(kp_fingertop)] += 1
        for i in string_keypoints_fingertops:
            mean_confidence_fingers[i].append(np.mean(np.array(mean_conf[i])))
            time_stretched[i].append(time_counter[i]/len(sample))
    return time_stretched["99"], time_stretched["103"], time_stretched["107"], time_stretched["111"], time_stretched["115"], time_stretched["120"], time_stretched["124"], time_stretched["128"], time_stretched["132"], time_stretched["136"], mean_confidence_fingers["99"], mean_confidence_fingers["103"], mean_confidence_fingers["107"], mean_confidence_fingers["111"], mean_confidence_fingers["115"], mean_confidence_fingers["120"], mean_confidence_fingers["124"], mean_confidence_fingers["128"], mean_confidence_fingers["132"], mean_confidence_fingers["136"]

def handpalm_orientation(samplearray, threshold=2):
    """
    Returns the orientation of the handpalm. Function returns -1 if handpalm is downwards oriented and 1 if upwards oriented

    :param samplearray: the array with all the samples
    :param threshold: value that determines how many fingers that we need to make a good decision
    """
    keypoints_fingertops_left_hand = [103, 107, 111, 115]
    keypoints_fingertops_right_hand = [124, 128, 132, 136]
    up_down_right = []
    up_down_left = []
    for sample in samplearray:
        up_down_sample_right = []
        up_down_sample_left = []
        for j in range(len(sample)):
            if sum([handpalm_upward_right(sample, j, kp_finger) for kp_finger in keypoints_fingertops_right_hand]) >= threshold:
                up_down_sample_right.append(1)
            else:
                up_down_sample_right.append(-1)
            if sum([handpalm_upward_left(sample, j, kp_finger) for kp_finger in keypoints_fingertops_left_hand]) >= threshold:
                up_down_sample_left.append(1)
            else:
                up_down_sample_left.append(-1)
        up_down_right.append(np.sign(sum(up_down_sample_right)))
        up_down_left.append(np.sign(sum(up_down_sample_left)))
    return up_down_left, up_down_right
            

def handpalm_upward_right(sample, j, kp_finger):
    """

    """
    xy_top = sample[j][kp_finger][:2]
    xy_mid1 = sample[j][kp_finger-1][:2]
    xy_mid2 = sample[j][kp_finger-2][:2]
    xy_bot = sample[j][kp_finger-3][:2]
    if xy_top[0] >= xy_bot[0] and (xy_top[1] <= xy_mid1[1] or xy_top[1] <= xy_mid2[1]):
        return True
    else:
        return False

def handpalm_upward_left(sample, j, kp_finger):
    """

    """
    xy_top = sample[j][kp_finger][:2]
    xy_mid1 = sample[j][kp_finger-1][:2]
    xy_mid2 = sample[j][kp_finger-2][:2]
    xy_bot = sample[j][kp_finger-3][:2]
    if xy_top[0] <= xy_bot[0] and (xy_top[1] <= xy_mid1[1] or xy_top[1] <= xy_mid2[1]):
        return True
    else:
        return False

def fractional_time_left_hand_above_right(samplearray, tolerance=2.5):
    """
    Calculate the (fractional) time that the left hand is above the right hand, or they are at the same height

    :param samplearray: the array with all the samples
    :param tolerance: tolerance used to determine if the hands are at the same height
    """
    time_left_above_right = []
    time_no_mov = []
    for sample in samplearray:
        no_mov = 0
        lar = 0
        for j in range(len(sample)):
            height_left_hand = sample[j][7][1]
            height_right_hand = sample[j][4][1]
            if (abs(height_left_hand-height_right_hand) < tolerance):
                no_mov += 1
            elif height_left_hand < height_right_hand:
                lar += 1
        time_left_above_right.append(lar/len(sample))
        time_no_mov.append(no_mov/len(sample))
    return time_left_above_right, time_no_mov

def closeness_hand(samplearray):
    """
    todo: divide into quarters and add median instead of mean
    """
    samplearray = normalise_frames(samplearray)

    left_shoulder = [[], [], [], []]
    right_shoulder = [[], [], [], []]
    left_centerpoint = [[], [], [], []]
    right_centerpoint = [[], [], [], []]
    left_centerpoint_face = [[], [], [], []]
    right_centerpoint_face = [[], [], [], []]
    ratio_shoulder = [[], [], [], []]
    ratio_centerpoint = [[], [], [], []]
    ratio_centerpoint_face = [[], [], [], []]
    
    for i in range(len(samplearray)):
        sample = samplearray[i]
        left_shoulder_temp = []
        right_shoulder_temp = []
        left_centerpoint_temp = []
        right_centerpoint_temp = []
        left_centerpoint_face_temp = []
        right_centerpoint_face_temp =[]
        ratio_shoulder_temp = []
        ratio_centerpoint_temp = []
        ratio_centerpoint_face_temp = []

        for j in range(len(sample)):
            coord_left_elbow = sample[j][6][:2]
            coord_right_elbow = sample[j][3][:2]
            coord_left_hand = sample[j][7][:2]
            coord_right_hand = sample[j][4][:2]
            coord_left_shoulder = sample[j][5][:2]
            coord_right_shoulder = sample[j][2][:2]
            coord_centerpoint = sample[j][1][:2]
            coord_centerpoint_face = sample[j][0][:2]

            dist_hand_elbow_left = calculateDistance(coord_left_elbow[0], coord_left_elbow[1], coord_left_hand[0], coord_left_hand[1])
            dist_hand_elbow_right = calculateDistance(coord_right_elbow[0], coord_right_elbow[1], coord_right_hand[0], coord_right_hand[1])
            dist_hand_shoulder_left = calculateDistance(coord_left_shoulder[0], coord_left_shoulder[1], coord_left_hand[0], coord_left_hand[1])
            dist_hand_shoulder_right = calculateDistance(coord_right_shoulder[0], coord_right_shoulder[1], coord_right_hand[0], coord_right_hand[1])
            dist_lefthand_centerpoint = calculateDistance(coord_centerpoint[0], coord_centerpoint[1], coord_left_hand[0], coord_left_hand[1])
            dist_righthand_centerpoint = calculateDistance(coord_centerpoint[0], coord_centerpoint[1], coord_right_hand[0], coord_right_hand[1])
            dist_lefthand_centerpoint_face = calculateDistance(coord_centerpoint_face[0], coord_centerpoint_face[1], coord_left_hand[0], coord_left_hand[1])
            dist_righthand_centerpoint_face = calculateDistance(coord_centerpoint_face[0], coord_centerpoint_face[1], coord_right_hand[0], coord_right_hand[1])

    
            closeness_left_shoulder = dist_hand_elbow_left / dist_hand_shoulder_left if dist_hand_shoulder_left > 0 else 1
            closeness_right_shoulder = dist_hand_elbow_right / dist_hand_shoulder_right if dist_hand_shoulder_right > 0 else 1
            closeness_left_centerpoint = dist_hand_elbow_left / dist_lefthand_centerpoint if dist_lefthand_centerpoint > 0 else 1
            closeness_right_centerpoint = dist_hand_elbow_right / dist_righthand_centerpoint if dist_righthand_centerpoint > 0 else 1
            closeness_left_centerpoint_face = dist_hand_elbow_left / dist_lefthand_centerpoint_face if dist_lefthand_centerpoint_face > 0 else 1
            closeness_right_centerpoint_face = dist_hand_elbow_right / dist_righthand_centerpoint_face if dist_righthand_centerpoint_face > 0 else 1

            left_shoulder_temp.append(closeness_left_shoulder)
            right_shoulder_temp.append(closeness_right_shoulder)
            left_centerpoint_temp.append(closeness_left_centerpoint)
            right_centerpoint_temp.append(closeness_right_centerpoint)
            left_centerpoint_face_temp.append(closeness_left_centerpoint_face)
            right_centerpoint_face_temp.append(closeness_right_centerpoint_face)
            if closeness_right_shoulder == 0:
                ratio_shoulder_temp.append(float('nan'))
            else:
                ratio_shoulder_temp.append(closeness_left_shoulder/closeness_right_shoulder)
            if closeness_right_centerpoint == 0:
                ratio_centerpoint_temp.append(float('nan'))
            else:
                ratio_centerpoint_temp.append(closeness_left_centerpoint/closeness_right_centerpoint)
            if closeness_right_centerpoint_face == 0:
                ratio_centerpoint_face_temp.append(float('nan'))
            else:
                ratio_centerpoint_face_temp.append(closeness_left_centerpoint_face/closeness_right_centerpoint_face)
        
        for quarter in range(4):
            start = quarter * len(sample) // 4
            stop = (quarter + 1) * len(sample) // 4
            left_shoulder[quarter].append(np.mean(left_shoulder_temp[start:stop]))
            right_shoulder[quarter].append(np.mean(right_shoulder_temp[start:stop]))
            left_centerpoint[quarter].append(np.mean(left_centerpoint_temp[start:stop]))
            right_centerpoint[quarter].append(np.mean(right_centerpoint_temp[start:stop]))
            left_centerpoint_face[quarter].append(np.mean(left_centerpoint_face_temp[start:stop]))
            right_centerpoint_face[quarter].append(np.mean(right_centerpoint_face_temp[start:stop]))
            if isNaNarray(ratio_shoulder_temp[start:stop]):
                ratio_shoulder[quarter].append(1)
            else:
                ratio_shoulder[quarter].append(np.nanmean(ratio_shoulder_temp[start:stop]))
            if isNaNarray(ratio_centerpoint_temp[start:stop]):
                ratio_centerpoint[quarter].append(1)
            else:
                ratio_centerpoint[quarter].append(np.nanmean(ratio_centerpoint_temp[start:stop]))
            if isNaNarray(ratio_centerpoint_face_temp[start:stop]):
                ratio_centerpoint_face[quarter].append(1)
            else:
                ratio_centerpoint_face[quarter].append(np.nanmean(ratio_centerpoint_face_temp[start:stop]))
    
    return left_shoulder, right_shoulder, left_centerpoint, right_centerpoint, left_centerpoint_face, right_centerpoint_face, ratio_shoulder, ratio_centerpoint, ratio_centerpoint_face

def index_to_face_dist(samplearray):
    samplearray = normalise_frames(samplearray)
    l_index_chin = [[], [], [], []]
    r_index_chin = [[], [], [], []]
    r_index_r_eye = [[], [], [], []]
    l_index_l_eye = [[], [], [], []]
    fraction = [[], [], [], []]
    for i in range(len(samplearray)):
        sample = samplearray[i]
        l_index_chin_temp = []
        r_index_chin_temp = []
        r_index_r_eye_dist_temp = []
        l_index_l_eye_dist_temp = []
        fraction_temp = []
        for j in range(len(sample)):
            confidence_chin_keypoint = sample[j][33][2]
            confidence_right_eye = sample[j][93][2]
            confidence_left_eye = sample[j][94][2]
            if confidence_chin_keypoint > 0:
                l_index_chin_dist = calculateDistance(sample[j][33][0], sample[j][33][1], sample[j][103][0], sample[j][103][1])
                r_index_chin_dist = calculateDistance(sample[j][33][0], sample[j][33][1], sample[j][124][0], sample[j][124][1])
            else:
                l_index_chin_dist = calculateDistance(sample[j][0][0], sample[j][0][1], sample[j][103][0], sample[j][103][1])
                r_index_chin_dist = calculateDistance(sample[j][0][0], sample[j][0][1], sample[j][124][0], sample[j][124][1])
            if confidence_right_eye > 0:
                r_index_r_eye_dist = calculateDistance(sample[j][93][0], sample[j][93][1], sample[j][124][0], sample[j][124][1])
            else:
                r_index_r_eye_dist = calculateDistance(sample[j][17][0], sample[j][17][1], sample[j][124][0], sample[j][124][1])
            if confidence_left_eye > 0:
                l_index_l_eye_dist = calculateDistance(sample[j][94][0], sample[j][94][1], sample[j][103][0], sample[j][103][1])
            else:
                l_index_l_eye_dist = calculateDistance(sample[j][18][0], sample[j][18][1], sample[j][103][0], sample[j][103][1])
            fraction_val = l_index_chin_dist / r_index_chin_dist

            l_index_chin_temp.append(l_index_chin_dist)
            r_index_chin_temp.append(r_index_chin_dist)
            r_index_r_eye_dist_temp.append(r_index_r_eye_dist)
            l_index_l_eye_dist_temp.append(l_index_l_eye_dist)
            fraction_temp.append(fraction_val)
        for quarter in range(4):
            start = quarter * len(sample) // 4
            stop = (quarter + 1) * len(sample) // 4
            l_index_chin[quarter].append(np.median(l_index_chin_temp[start:stop]))
            r_index_chin[quarter].append(np.median(r_index_chin_temp[start:stop]))
            r_index_r_eye[quarter].append(np.median(r_index_r_eye_dist_temp[start:stop]))
            l_index_l_eye[quarter].append(np.median(l_index_l_eye_dist_temp[start:stop]))
            fraction[quarter].append(np.median(fraction_temp[start:stop]))
    return l_index_chin, r_index_chin, r_index_r_eye, l_index_l_eye, fraction

def only_index_pink_middle_stretched(samplearray):
    """
    Feature that returns a value that indicates if the index and pink finger of the left/right hand is stretched while the other fingers are not. The higher this number, the higher the probability that this is the case

    :param samplearray: the array with all the samples
    """
    left_i = []
    right_i = []
    left_p = []
    right_p = []
    left_m = []
    right_m = []
    for i in range(len(samplearray)):
        sample = samplearray[i]
        left_i_temp = []
        right_i_temp = []
        left_p_temp = []
        right_p_temp = []
        left_m_temp = []
        right_m_temp = []
        for j in range(len(sample)):
            if sample[j][95][2] > 0 and sample[j][103][2] > 0:
                l_index = calculateDistance(sample[j][95][0], sample[j][95][1], sample[j][103][0], sample[j][103][1])
            else:
                l_index = float('nan')
            if sample[j][95][2] > 0 and sample[j][111][2] > 0:
                l_ring = calculateDistance(sample[j][95][0], sample[j][95][1], sample[j][111][0], sample[j][111][1])
            else:
                l_ring = float('nan')
            if sample[j][116][2] > 0 and sample[j][124][2] > 0:
                r_index = calculateDistance(sample[j][116][0], sample[j][116][1], sample[j][124][0], sample[j][124][1])
            else:
                r_index = float('nan')
            if sample[j][116][2] > 0 and sample[j][132][2] > 0:
                r_ring = calculateDistance(sample[j][116][0], sample[j][116][1], sample[j][132][0], sample[j][132][1])
            else:
                r_ring = float('nan')
            if sample[j][116][2] > 0 and sample[j][136][2] > 0:
                r_pink = calculateDistance(sample[j][116][0], sample[j][116][1], sample[j][136][0], sample[j][136][1])
            else:
                r_pink = float('nan')
            if sample[j][95][2] > 0 and sample[j][115][2] > 0:
                l_pink = calculateDistance(sample[j][95][0], sample[j][95][1], sample[j][115][0], sample[j][115][1])
            else:
                l_pink = float('nan')
            if sample[j][95][2] > 0 and sample[j][107][2] > 0:
                l_middle = calculateDistance(sample[j][95][0], sample[j][95][1], sample[j][107][0], sample[j][107][1])
            else:
                l_middle = float('nan')
            if sample[j][116][2] > 0 and sample[j][128][2] > 0:
                r_middle = calculateDistance(sample[j][116][0], sample[j][116][1], sample[j][128][0], sample[j][128][1])
            else:
                r_middle = float('nan')
            
            left_i_temp.append(l_index**2 / l_ring)
            right_i_temp.append(r_index**3 / r_ring) #to the power 3 because the right hand is more important in the gestures
            left_p_temp.append(l_pink**2 / l_ring)
            right_p_temp.append(r_pink**3 / r_ring) #to the power 3 because the right hand is more important in the gestures
            left_m_temp.append(l_middle**2 / l_ring)
            right_m_temp.append(r_middle**3 / r_ring) #to the power 3 because the right hand is more important in the gestures
        if isNaNarray(left_i_temp):
            left_i.append(1)
        else:
            left_i.append(np.nanmedian(left_i_temp))
        if isNaNarray(right_i_temp):
            right_i.append(1)
        else:
            right_i.append(np.nanmedian(right_i_temp))
        if isNaNarray(left_p_temp):
            left_p.append(1)
        else:
            left_p.append(np.nanmedian(left_p_temp))
        if isNaNarray(right_p_temp):
            right_p.append(1)
        else:
            right_p.append(np.nanmedian(right_p_temp))
        if isNaNarray(left_m_temp):
            left_m.append(1)
        else:
            left_m.append(np.nanmedian(left_m_temp))
        if isNaNarray(right_m_temp):
            right_m.append(1)
        else:
            right_m.append(np.nanmedian(right_m_temp))
    return left_i, right_i, left_p, right_p, left_m, right_m

def hand_stretched(samplearray):
    """
    Returns a value that determines if the hand (i.e. all the fingers except the thumb and index finger) is stretched, per quarter. If value > 1 the hand is not stretched, if value < 1 hand is stretched

    :param samplearray: the array with all the samples
    """
    samplearray = normalise_frames(samplearray)
    #left_fingers_keypoints = [107, 111, 115]
    left_fingers_keypoints = [107, 111]
    #right_fingers_keypoints = [128, 132, 136]
    right_fingers_keypoints = [128, 132]
    left = [[], [], [], []]
    right = [[], [], [], []]
    for i in range(len(samplearray)):
        sample = samplearray[i]
        values_left = []
        values_right = []
        for j in range(len(sample)):
            sum_left = 0
            sum_right = 0
            for kp in left_fingers_keypoints:
                if sample[j][kp][2] > 0 and sample[j][95][2] > 0 and sample[j][kp-2][2] > 0:
                    dist_top = calculateDistance(sample[j][kp][0], sample[j][kp][1], sample[j][95][0], sample[j][95][1])
                    dist_middle = calculateDistance(sample[j][kp-2][0], sample[j][kp-2][1], sample[j][95][0], sample[j][95][1])
                else:
                    dist_top, dist_middle = 1, 1

                sum_left += dist_middle / dist_top

            for kp in right_fingers_keypoints:
                if sample[j][kp][2] > 0 and sample[j][116][2] > 0 and sample[j][kp-2][2] > 0:
                    dist_top = calculateDistance(sample[j][kp][0], sample[j][kp][1], sample[j][116][0], sample[j][116][1])
                    dist_middle = calculateDistance(sample[j][kp-2][0], sample[j][kp-2][1], sample[j][116][0], sample[j][116][1])
                else:
                    dist_top, dist_middle = 1, 1

                sum_right += dist_middle / dist_top
            values_left.append(sum_left)
            values_right.append(sum_right)

        for quarter in range(4):
            start = quarter * len(sample) // 4
            stop = (quarter + 1) * len(sample) // 4
            left[quarter].append(np.nanmedian(values_left[start:stop]))
            right[quarter].append(np.nanmedian(values_right[start:stop]))
    return left, right
        
def mov_finger_rel_wrist(samplearray, tol_conf=0.10):
    """
    Function that calculates the relative movement of the fingers to the wrist
    """
    left = []
    right = []
    keypoints = [[103, 107, 111, 115], [124, 128, 132, 136]]
    for sample in samplearray:
        left_temp = []
        right_temp = []
        if len(sample) == 1:
            left.append(1)
            right.append(1)
        else:
            for j in range(1, len(sample)):
                wrist_left_displ = calculateDistance(sample[j][7][0], sample[j][7][1], sample[j-1][7][0], sample[j-1][7][1])
                wrist_right_displ = calculateDistance(sample[j][4][0], sample[j][4][1], sample[j-1][4][0], sample[j-1][4][1])
                for k in range(2):
                    temp = []
                    for kp in keypoints[k]:
                        if sample[j][kp][2] > tol_conf and sample[j-1][kp][2] > tol_conf:
                            temp.append(calculateDistance(sample[j][kp][0], sample[j][kp][1], sample[j-1][kp][0], sample[j-1][kp][1]))
                        else:
                            temp.append(float('nan'))
                    if k == 0:
                        displ_left_fingers = np.nanmedian(temp)
                    elif k == 1:
                        displ_right_fingers = np.nanmedian(temp)
                if math.isnan(displ_left_fingers):
                    left_temp.append(float('nan'))
                else:
                    left_temp.append((displ_left_fingers-wrist_left_displ)) #Maybe with abs?
                if math.isnan(displ_right_fingers):
                    right_temp.append(float('nan'))
                else:
                    right_temp.append((displ_right_fingers-wrist_right_displ)) #Maybe with abs?
            if isNaNarray(left_temp):
                left.append(1)
            else:
                left.append(np.nanmean(left_temp))
            if isNaNarray(right_temp):
                right.append(1)
            else:
                right.append(np.nanmean(right_temp))

    return left, right

def isNaNarray(array):
    """
    Help function to determine if an array contains only NaN's, returrns True if so, otherwise False

    :param array: the array to check
    """
    for val in array:
        if not(math.isnan(val)):
            return False
    return True

def fract_time_hands_close(samplearray, tolerance=40):
    """
    """
    keypoints = [[102, 105, 110, 114], [123, 127, 131, 135]]
    time = []
    for i in range(len(samplearray)):
        sample = samplearray[i]
        counter = 0
        for j in range(len(sample)):
            for k in range(2):
                temp = []
                for kp in keypoints[k]:
                    temp.append(sample[j][kp][:2])
                if k == 0:
                    left_coord = np.median(temp, axis=0)
                elif k == 1:
                    right_coord = np.median(temp, axis=0)
            if calculateDistance(left_coord[0], left_coord[1], right_coord[0], right_coord[1]) < tolerance:
                counter += 1
        time.append(counter/len(sample))
    return time

def relative_mov_wrists(samplearray):
    """
    """
    l = []
    r = []
    lrxx = []
    lryy = []
    lrxy = []
    lryx = []
    for sample in samplearray:
        l_ = []
        r_ = []
        lrxx_ = []
        lryy_ = []
        lrxy_ = []
        lryx_ = []
        for j in range(1, len(sample)):
            l_wrist_x, l_wrist_y = sample[j][7][:2] - sample[j-1][7][:2]
            r_wrist_x, r_wrist_y = sample[j][4][:2] - sample[j-1][4][:2]
            if l_wrist_x == 0:
                l_wrist_x = 1
            if r_wrist_x == 0:
                r_wrist_x = 1
            if l_wrist_y == 0:
                l_wrist_y = 1
            if r_wrist_y == 0:
                r_wrist_y = 1
            if sample[j][7][2] == 0:
                l_wrist_x, l_wrist_y = float('nan'), float('nan')
            if sample[j][4][2] == 0:
                r_wrist_x, r_wrist_y = float('nan'), float('nan')
            l_.append(l_wrist_x**5/l_wrist_y**5)
            r_.append(r_wrist_x**5/r_wrist_y**5)
            lrxx_.append(l_wrist_x/r_wrist_x)
            lryy_.append(l_wrist_y/r_wrist_y)
            lrxy_.append(l_wrist_x/r_wrist_y)
            lryx_.append(l_wrist_y/r_wrist_x)
        if isNaNarray(l_):
            l.append(1)
        else:
            l.append(np.nanmean(l_))
        if isNaNarray(r_):
            r.append(1)
        else:
            r.append(np.nanmean(r_))
        if isNaNarray(lrxx_):
            lrxx.append(1)
        else:
            lrxx.append(np.nanmean(lrxx_))
        if isNaNarray(lryy_):
            lryy.append(1)
        else:
            lryy.append(np.nanmean(lryy_))
        if isNaNarray(lrxy_):
            lrxy.append(1)
        else:
            lrxy.append(np.nanmean(lrxy_))
        if isNaNarray(lryx_):
            lryx.append(1)
        else:
            lryx.append(np.nanmean(lryx_))
    
    return l, r, lrxx, lryy, lrxy, lryx
        
def mean_dist_quarter(samplearray, keypoint1, keypoint2):
    """
    """
    samplearray = normalise_frames(samplearray)
    dist = [[], [], [], []]
    for sample in samplearray:
        dist_ = []
        for frame in sample:
            if frame[keypoint1][2] > 0.1 and frame[keypoint2][2] > 0.1:
                dist_.append(calculateDistance(frame[keypoint1][0], frame[keypoint1][1], frame[keypoint2][0], frame[keypoint2][1]))
            else:
                dist_.append(float('nan'))
        for quarter in range(4):
            start = quarter * len(sample) // 4
            stop = (quarter + 1) * len(sample) // 4
            if isNaNarray(dist_[start:stop]):
                dist[quarter].append(10)
            else:
                dist[quarter].append(np.nanmean(dist_[start:stop]))    
    
    return dist

def hand_direction(samplearray, keypoint, tolerance=3):
    """
    """
    samplearray = normalise_frames(samplearray)
    x = [[], [], [], []]
    y = [[], [], [], []]
    for sample in samplearray:
        val = []
        for j in range(1, len(sample)):
            if sample[j][keypoint][2] == 0 or sample[j-1][keypoint][2] == 0:
                vector = [float('nan'), float('nan')]
            elif calculateDistance(sample[j][keypoint][0], sample[j][keypoint][1], sample[j-1][keypoint][0], sample[j-1][keypoint][1]) < tolerance:
                vector = [0, 0]
            else:
                vector = sample[j][keypoint][:2] - sample[j-1][keypoint][:2]
            val.append(vector)
        val = np.array(val)
        for quarter in range(4):
            start = quarter * len(sample) // 4
            stop = (quarter + 1) * len(sample) // 4
            if isNaNarray(val[:,0][start:stop]):
                x[quarter].append(0)
            else:
                x[quarter].append(np.nanmean(val[:,0][start:stop]))
            if isNaNarray(val[:,1][start:stop]):
                y[quarter].append(0)
            else:
                y[quarter].append(np.nanmean(val[:,1][start:stop]))
            
    return x, y

            

    
def extract_features(samples_list):
    df_design_matrix = pd.DataFrame()
    
    #Hand in movement:
    #movement_hand=hand_moving(samples_list)
    #df_design_matrix['Hand or hands moving'] = movement_hand
    
    #
    for kp in [4, 7, 103, 111, 124, 132]:
        hd = hand_direction(samples_list, kp)
        for quarter in range(4):
            df_design_matrix['X direction of the keypoint {} in quarter {}'.format(kp, quarter)] = hd[0][quarter]
            df_design_matrix['Y direction of the keypoint {} in quarter {}'.format(kp, quarter)] = hd[1][quarter]
    #
    offset_left_hand = 95
    offset_right_hand = 116
    finger_keypoints_to_check = [0,4,8,12,16,20]
    couples_keypoints_to_check = []
    for i in range(len(finger_keypoints_to_check)):
        for j in range(i,len(finger_keypoints_to_check)):
            if (i != j):
                couples_keypoints_to_check.append((offset_left_hand+finger_keypoints_to_check[i], offset_left_hand+finger_keypoints_to_check[j]))
                couples_keypoints_to_check.append((offset_right_hand+finger_keypoints_to_check[i], offset_right_hand+finger_keypoints_to_check[j]))
    
    for couple in couples_keypoints_to_check:
        fing_mean_std = keypoints_dist_mean_std(samples_list, couple[0], couple[1])
        for quarter in range(4):
            df_design_matrix["Mean of distance between keypoint {} and {}, quarter {}".format(couple[0], couple[1], quarter)] = fing_mean_std[0][quarter]
            df_design_matrix["Std of distance beteween keypoint {} and {}, quarter {}".format(couple[0], couple[1], quarter)] = fing_mean_std[1][quarter]
    

    #Relative movement of the wrists
    rmw = relative_mov_wrists(samples_list)
    df_design_matrix['Relative move left wrist x / left wrist y'] = rmw[0]
    df_design_matrix['Relative move right wrist x / right wrist y'] = rmw[1]
    df_design_matrix['Relative move left wrist x / right wrist x'] = rmw[2]
    df_design_matrix['Relative move left wrist y / right wrist y'] = rmw[3]
    df_design_matrix['Relative move left wrist x / right wrist y'] = rmw[4]
    df_design_matrix['Relative move left wrist y / right wrist x'] = rmw[5]

    #Fractional time that left and right hand are close together (defined by tolerance value)
    df_design_matrix['Fractional time that the left and right hand are close together'] = fract_time_hands_close(samples_list)

    #Movement of fingers compared to movement of wrists
    mfrw_left, mfrw_right = mov_finger_rel_wrist(samples_list)
    df_design_matrix['Movement of fingers compared to movement of wrist, left hand'] = mfrw_left
    df_design_matrix['Movement of fingers compared to movement of wrist, right hand'] = mfrw_right

    #Feature to check if the hands are stretched. If value > 1 hand stretched, if < 1 not
    hs = hand_stretched(samples_list)
    for i in range(4):
        df_design_matrix['Left hand stretched, quarter {}'.format(i+1)] = hs[0][i]
        df_design_matrix['Right hand stretched, quarter {}'.format(i+1)] = hs[1][i]

    #Angles between 2 keypoints and a controlpoint (default keypoint 1)
    #Pink and ring finger left hand
    #Label 1
    #angle_fing_left_hand_l1, std_fing_left_hand_l1 = angle_quarters(samples_list, 115, 111,113)
    #df_design_matrix['Angle between pink and ring finger left hand label 1 Q1'] = angle_fing_left_hand_l1[0]
    #df_design_matrix['Angle between pink and ring finger left hand label 1 Q2'] = angle_fing_left_hand_l1[1]
    #df_design_matrix['Angle between pink and ring finger left hand label 1 Q3'] = angle_fing_left_hand_l1[2]
    #df_design_matrix['Angle between pink and ring finger left hand label 1 Q4'] = angle_fing_left_hand_l1[3]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 1 Q1'] = std_fing_left_hand_l1[0]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 1 Q2'] = std_fing_left_hand_l1[1]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 1 Q3'] = std_fing_left_hand_l1[2]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 1 Q4'] = std_fing_left_hand_l1[3]
    #Label 2
    #angle_fing_left_hand_l2, std_fing_left_hand_l2 = angle_quarters(samples_list, 113, 109,112)
    #df_design_matrix['Angle between pink and ring finger left hand label 2 Q1'] = angle_fing_left_hand_l2[0]
    #df_design_matrix['Angle between pink and ring finger left hand label 2 Q2'] = angle_fing_left_hand_l2[1]
    #df_design_matrix['Angle between pink and ring finger left hand label 2 Q3'] = angle_fing_left_hand_l2[2]
    #df_design_matrix['Angle between pink and ring finger left hand label 2 Q4'] = angle_fing_left_hand_l2[3]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 2 Q1'] = std_fing_left_hand_l2[0]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 2 Q2'] = std_fing_left_hand_l2[1]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 2 Q3'] = std_fing_left_hand_l2[2]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 2 Q4'] = std_fing_left_hand_l2[3]

    #Pink and ring fingers right hand
    #Label 1
    #angle_fing_right_hand_l1, std_fing_right_hand_l1 = angle_quarters(samples_list, 136, 132,134)
    #df_design_matrix['Angle between pink and ring finger right hand label 1 Q1'] = angle_fing_right_hand_l1[0]
    #df_design_matrix['Angle between pink and ring finger right hand label 1 Q2'] = angle_fing_right_hand_l1[1]
    #df_design_matrix['Angle between pink and ring finger right hand label 1 Q3'] = angle_fing_right_hand_l1[2]
    #df_design_matrix['Angle between pink and ring finger right hand label 1 Q4'] = angle_fing_right_hand_l1[3]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger right hand label 1 Q1'] = std_fing_right_hand_l1[0]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger right hand label 1 Q2'] = std_fing_right_hand_l1[1]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger right hand label 1 Q3'] = std_fing_right_hand_l1[2]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger right hand label 1 Q4'] = std_fing_right_hand_l1[3]
    #Label 2
    #angle_fing_right_hand_l2, std_fing_right_hand_l2 = angle_quarters(samples_list, 134, 130,133)
    #df_design_matrix['Angle between pink and ring finger left hand label 2 Q1'] = angle_fing_right_hand_l2[0]
    #df_design_matrix['Angle between pink and ring finger left hand label 2 Q2'] = angle_fing_right_hand_l2[1]
    #df_design_matrix['Angle between pink and ring finger left hand label 2 Q3'] = angle_fing_right_hand_l2[2]
    #df_design_matrix['Angle between pink and ring finger left hand label 2 Q4'] = angle_fing_right_hand_l2[3]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 2 Q1'] = std_fing_right_hand_l2[0]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 2 Q2'] = std_fing_right_hand_l2[1]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 2 Q3'] = std_fing_right_hand_l2[2]
    #df_design_matrix['Standard deviaton of the Angle between pink and ring finger left hand label 2 Q4'] = std_fing_right_hand_l2[3]

    #Wrists and neck(default)
    angle_wrists, std_wrists = angle_quarters(samples_list, 4, 7,1)
    df_design_matrix['Angle between wrists and neck(default) Q1'] = angle_wrists[0]
    df_design_matrix['Angle between wrists and neck(default) Q2'] = angle_wrists[1]
    df_design_matrix['Angle between wrists and neck(default) Q3'] = angle_wrists[2]
    df_design_matrix['Angle between wrists and neck(default) Q4'] = angle_wrists[3]
    df_design_matrix['Standard deviaton of the angle between wrists and neck(default) Q1'] = std_wrists[0]
    df_design_matrix['Standard deviaton of the angle between wrists and neck(default) Q2'] = std_wrists[1]
    df_design_matrix['Standard deviaton of the angle between wrists and neck(default) Q3'] = std_wrists[2]
    df_design_matrix['Standard deviaton of the angle between wrists and neck(default) Q4'] = std_wrists[3]
    
    #ellbows and neck(default)
    angle_ellbows, std_ellbows = angle_quarters(samples_list, 3, 6,1)
    df_design_matrix['Angle between ellbows and neck(default) Q1'] = angle_ellbows[0]
    df_design_matrix['Angle between ellbows and neck(default) Q2'] = angle_ellbows[1]
    df_design_matrix['Angle between ellbows and neck(default) Q3'] = angle_ellbows[2]
    df_design_matrix['Angle between ellbows and neck(default) Q4'] = angle_ellbows[3]
    df_design_matrix['Standard deviaton of the angle between ellbows and neck(default) Q1'] = std_ellbows[0]
    df_design_matrix['Standard deviaton of the angle between ellbows and neck(default) Q2'] = std_ellbows[1]
    df_design_matrix['Standard deviaton of the angle between ellbows and neck(default) Q3'] = std_ellbows[2]
    df_design_matrix['Standard deviaton of the angle between ellbows and neck(default) Q4'] = std_ellbows[3]
    
    #index fingers and neck(default)
    angle_fingers, std_fingers = angle_quarters(samples_list, 103, 124,1)
    df_design_matrix['Angle between fingers and neck(default) Q1'] = angle_fingers[0]
    df_design_matrix['Angle between fingers and neck(default) Q2'] = angle_fingers[1]
    df_design_matrix['Angle between fingers and neck(default) Q3'] = angle_fingers[2]
    df_design_matrix['Angle between fingers and neck(default) Q4'] = angle_fingers[3]
    df_design_matrix['Standard deviaton of the angle between fingers and neck(default) Q1'] = std_fingers[0]
    df_design_matrix['Standard deviaton of the angle between fingers and neck(default) Q2'] = std_fingers[1]
    df_design_matrix['Standard deviaton of the angle between fingers and neck(default) Q3'] = std_fingers[2]
    df_design_matrix['Standard deviaton of the angle between fingers and neck(default) Q4'] = std_fingers[3]
    
    #index fingers and left eye (body)
    angle_fingers_eye, std_fingers_eye = angle_quarters(samples_list, 103, 124, 0) #0 is a body keypoint resembling the middle of the face
    df_design_matrix['Angle between fingers and left eye Q1'] = angle_fingers_eye[0]
    df_design_matrix['Angle between fingers and left eye Q2'] = angle_fingers_eye[1]
    df_design_matrix['Angle between fingers and left eye Q3'] = angle_fingers_eye[2]
    df_design_matrix['Angle between fingers and left eye Q4'] = angle_fingers_eye[3]
    df_design_matrix['Standard deviaton of the angle between fingers and left eye Q1'] = std_fingers_eye[0]
    df_design_matrix['Standard deviaton of the angle between fingers and left eye Q2'] = std_fingers_eye[1]
    df_design_matrix['Standard deviaton of the angle between fingers and left eye Q3'] = std_fingers_eye[2]
    df_design_matrix['Standard deviaton of the angle between fingers and left eye Q4'] = std_fingers_eye[3]
    
    #Wrists and neck(default)
    angle_wrists_eye, std_wrists_eye = angle_quarters(samples_list, 4, 7, 0)
    df_design_matrix['Angle between wrists and left eye Q1'] = angle_wrists_eye[0]
    df_design_matrix['Angle between wrists and left eye Q2'] = angle_wrists_eye[1]
    df_design_matrix['Angle between wrists and left eye Q3'] = angle_wrists_eye[2]
    df_design_matrix['Angle between wrists and left eye Q4'] = angle_wrists_eye[3]
    df_design_matrix['Standard deviaton of the angle between wrists and left eye Q1'] = std_wrists_eye[0]
    df_design_matrix['Standard deviaton of the angle between wrists and left eye Q2'] = std_wrists_eye[1]
    df_design_matrix['Standard deviaton of the angle between wrists and left eye Q3'] = std_wrists_eye[2]
    df_design_matrix['Standard deviaton of the angle between wrists and left eye Q4'] = std_wrists_eye[3]

    #Feature that returns a value that indicates if the index finger, middle finger or pink of the left/right hand is stretched while the other fingers are not. The higher this number, the higher the probability that this is the case
    oips = only_index_pink_middle_stretched(samples_list)
    df_design_matrix['Only left index finger stretched value'] = oips[0]
    df_design_matrix['Only right index finger stretched value'] = oips[1]
    df_design_matrix['Only left pink stretched value'] = oips[2]
    df_design_matrix['Only right pink stretched value'] = oips[3]
    df_design_matrix['Only left middle finger stretched value'] = oips[4]
    df_design_matrix['Only right middle finger stretched value'] = oips[5]

    #Distances and fraction of distances between left / right index fingers and chin, left and right eye, where low confidence keypoints are replaced by higher confidence keypoints. Values are given as median per quarter
    index_dist = index_to_face_dist(samples_list)
    for i in range(4):
        df_design_matrix['Distance between left index and chin, quarter {}'.format(i +1)] = index_dist[0][i]
        df_design_matrix['Distance between right index and chin, quarter {}'.format(i +1)] = index_dist[1][i]
        df_design_matrix['Distance between right index and right eye, quarter {}'.format(i +1)] = index_dist[2][i]
        df_design_matrix['Distance between right index and left eye, quarter {}'.format(i +1)] = index_dist[3][i]
        df_design_matrix['Fraction of distances between left index and chin and right index and chin, quarter {}'.format(i +1)] = index_dist[4][i]
    
    #Closeness value of the left and right hands towards the shoulders, centerpoint (keypoint 1) and centerpoint of the face (keypoint 0)
    ch = closeness_hand(samples_list)
    for i in range(4):
        df_design_matrix['Closeness value left hand to left shoulder, quarter {}'.format(i +1)] = ch[0][i]
        df_design_matrix['Closeness value right hand to right shoulder, quarter {}'.format(i +1)] = ch[1][i]
        df_design_matrix['Closeness value left hand to centerpoint, quarter {}'.format(i +1)] = ch[2][i]
        df_design_matrix['Closeness value right hand to centerpoint, quarter {}'.format(i +1)] = ch[3][i]
        df_design_matrix['Closeness value left hand to centerpoint face, quarter {}'.format(i +1)] = ch[4][i]
        df_design_matrix['Closeness value right hand to centerpoint face, quarter {}'.format(i +1)] = ch[5][i]
        df_design_matrix['Ratio of closeness value left/right hand to left/right shoulder, quarter {}'.format(i +1)] = ch[6][i]
        df_design_matrix['Ratio of closeness value left/right hand to centerpoint, quarter {}'.format(i +1)] = ch[7][i]
        df_design_matrix['Ratio of closeness value left/right hand to centerpoint face, quarter {}'.format(i +1)] = ch[8][i]

    #(Fractional) time that left hand is above the right hand or they are at the same height
    ftlhar = fractional_time_left_hand_above_right(samples_list)
    df_design_matrix['Fractional time left hand above right hand'] = ftlhar[0]
    df_design_matrix['Fractional time hands at same height'] = ftlhar[1]


    #Handpalm orientation (upwards or downwards oriented) of left and right hand
    orientation = handpalm_orientation(samples_list)
    df_design_matrix['Orientation of handpalm left hand'] = orientation[0]
    df_design_matrix['Orientation of handpalm right hand'] = orientation[1]

    #(Fractional) time that each finger is stretched during a sample + mean confidence of finger keypoints
    time_stretched = fractional_time_fingers_stretched(samples_list)
    df_design_matrix['Fractional time finger with keypoint 99 stretched'] = time_stretched[0]
    df_design_matrix['Fractional time finger with keypoint 103 stretched'] = time_stretched[1]
    df_design_matrix['Fractional time finger with keypoint 107 stretched'] = time_stretched[2]
    df_design_matrix['Fractional time finger with keypoint 111 stretched'] = time_stretched[3]
    df_design_matrix['Fractional time finger with keypoint 115 stretched'] = time_stretched[4]
    df_design_matrix['Fractional time finger with keypoint 120 stretched'] = time_stretched[5]
    df_design_matrix['Fractional time finger with keypoint 124 stretched'] = time_stretched[6]
    df_design_matrix['Fractional time finger with keypoint 128 stretched'] = time_stretched[7]
    df_design_matrix['Fractional time finger with keypoint 132 stretched'] = time_stretched[8]
    df_design_matrix['Fractional time finger with keypoint 136 stretched'] = time_stretched[9]
    df_design_matrix['Mean confidence keypoints of finger with keypoint 99'] = time_stretched[10]
    df_design_matrix['Mean confidence keypoints of finger with keypoint 103'] = time_stretched[11]
    df_design_matrix['Mean confidence keypoints of finger with keypoint 107'] = time_stretched[12]
    df_design_matrix['Mean confidence keypoints of finger with keypoint 111'] = time_stretched[13]
    df_design_matrix['Mean confidence keypoints of finger with keypoint 115'] = time_stretched[14]
    df_design_matrix['Mean confidence keypoints of finger with keypoint 120'] = time_stretched[15]
    df_design_matrix['Mean confidence keypoints of finger with keypoint 124'] = time_stretched[16]
    df_design_matrix['Mean confidence keypoints of finger with keypoint 128'] = time_stretched[17]
    df_design_matrix['Mean confidence keypoints of finger with keypoint 132'] = time_stretched[18]
    df_design_matrix['Mean confidence keypoints of finger with keypoint 136'] = time_stretched[19]

    #(Fractional) time that each finger is curved downwards during a sample
    time_cd = fractional_time_finger_curved_downwards(samples_list)
    keypoints_fingertops = [99, 103, 107, 111, 115, 120, 124, 128, 132, 136]
    for i in range(10):
        df_design_matrix["Fractional time of downwards curved fingertop with keypoint {}".format(keypoints_fingertops[i])] = time_cd[i]

    #Median and stdev of curvature angle of the left and right hand
    curvature_angle_left_hand = curvature_angle_hand(samples_list, True)
    curvature_angle_right_hand = curvature_angle_hand(samples_list, False)
    df_design_matrix['Median of curvature angle left hand'] = curvature_angle_left_hand[0]
    df_design_matrix['Stdev of curvature angle left hand'] = curvature_angle_left_hand[1]
    df_design_matrix['Mean of confidence of fingertops left hand'] = curvature_angle_left_hand[2]
    df_design_matrix['Stdev of confidence of fingertops left hand'] = curvature_angle_left_hand[3]
    df_design_matrix['Median of curvature angle right hand'] = curvature_angle_right_hand[0]
    df_design_matrix['Stdev of curvature angle right hand'] = curvature_angle_right_hand[1]
    df_design_matrix['Mean of confidence of fingertops right hand'] = curvature_angle_right_hand[2]
    df_design_matrix['Stdev of confidence of fingertops right hand'] = curvature_angle_right_hand[3]

    #Mean and stdev of the horizontal and vertical distance between the left and right hand/elbow
    msd_hands = mean_stdev_distance_keypoints(samples_list, 7, 4)
    msd_elbows = mean_stdev_distance_keypoints(samples_list, 6, 3)
    df_design_matrix['Stdev of horizontal distance between hands'] = msd_hands[1]
    df_design_matrix['Stdev of vertical distance between hands'] = msd_hands[2]
    df_design_matrix['Ratio horizontal/vertical distance between hands'] = msd_hands[0]
    df_design_matrix['Stdev of horizontal distance between elbows'] = msd_elbows[1]
    df_design_matrix['Stdev of vertical distance between elbows'] = msd_elbows[2]
    df_design_matrix['Ratio horizontal/vertical distance between elbows'] = msd_elbows[0]

    #Median and stdev of the angle of the left and right elbow
    right_elbow_angle = elbow_angle(samples_list, 4, 3, 2)
    left_elbow_angle = elbow_angle(samples_list, 7, 6, 5)
    df_design_matrix['Median right elbow angle'] = right_elbow_angle[0]
    df_design_matrix['Stdev right elbow angle'] = right_elbow_angle[1]
    df_design_matrix['Median left elbow angle'] = left_elbow_angle[0]
    df_design_matrix['Stdev left elbow angle'] = left_elbow_angle[1]

    #Fractional move time left and right hand
    fmt_lefthand_lr = fractional_move_time(samples_list, 7, True)
    fmt_lefthand_ud = fractional_move_time(samples_list, 7, False)
    fmt_righthand_lr = fractional_move_time(samples_list, 4, True)
    fmt_righthand_ud = fractional_move_time(samples_list, 4, False)
    df_design_matrix['Fractional move time left hand left/right no mov'] = fmt_lefthand_lr[0]
    df_design_matrix['Fractional move time left hand left/right ur mov'] = fmt_lefthand_lr[1]
    df_design_matrix['Fractional move time left hand left/right dl mov'] = fmt_lefthand_lr[2]
    df_design_matrix['Fractional move time left hand up/down no mov'] = fmt_lefthand_ud[0]
    df_design_matrix['Fractional move time left hand up/down ur mov'] = fmt_lefthand_ud[1]
    df_design_matrix['Fractional move time left hand up/down dl mov'] = fmt_lefthand_ud[2]
    df_design_matrix['Fractional move time right hand left/right no mov'] = fmt_righthand_lr[0]
    df_design_matrix['Fractional move time right hand left/right ur mov'] = fmt_righthand_lr[1]
    df_design_matrix['Fractional move time right hand left/right dl mov'] = fmt_righthand_lr[2]
    df_design_matrix['Fractional move time right hand up/down no mov'] = fmt_righthand_ud[0]
    df_design_matrix['Fractional move time right hand up/down ur mov'] = fmt_righthand_ud[1]
    df_design_matrix['Fractional move time right hand up/down dl mov'] = fmt_righthand_ud[2]
     
    #Median and stdev of absolute position of hands and elbows
    msap_left_elbow = mean_stdev_absolute_position(samples_list, 6)
    msap_left_hand = mean_stdev_absolute_position(samples_list, 7)
    msap_right_elbow = mean_stdev_absolute_position(samples_list, 3)
    msap_right_hand = mean_stdev_absolute_position(samples_list, 4)
    df_design_matrix['Mean absolute x position left elbow'] = msap_left_elbow[0]
    df_design_matrix['Stdev absolute x position left elbow'] = msap_left_elbow[1]
    df_design_matrix['Mean absolute y position left elbow'] = msap_left_elbow[2]
    df_design_matrix['Stdev absolute y position left elbow'] = msap_left_elbow[3]
    df_design_matrix['Mean absolute x position left hand'] = msap_left_hand[0]
    df_design_matrix['Stdev absolute x position left hand'] = msap_left_hand[1]
    df_design_matrix['Mean absolute y position left hand'] = msap_left_hand[2]
    df_design_matrix['Stdev absolute y position left hand'] = msap_left_hand[3]
    df_design_matrix['Mean absolute x position right elbow'] = msap_right_elbow[0]
    df_design_matrix['Stdev absolute x position right elbow'] = msap_right_elbow[1]
    df_design_matrix['Mean absolute y position right elbow'] = msap_right_elbow[2]
    df_design_matrix['Stdev absolute y position right elbow'] = msap_right_elbow[3]
    df_design_matrix['Mean absolute x position right hand'] = msap_right_hand[0]
    df_design_matrix['Stdev absolute x position right hand'] = msap_right_hand[1]
    df_design_matrix['Mean absolute y position right hand'] = msap_right_hand[2]
    df_design_matrix['Stdev absolute y position right hand'] = msap_right_hand[3]
    
    #Switching
    df_design_matrix['Switching right thumb and pink'] = thumb_pink_switch(samples_list, 120, 136)
    df_design_matrix['Switching left thumb and pink'] = thumb_pink_switch(samples_list, 99, 115)
    df_design_matrix['Switching right fingers'] = thumb_pink_switch(samples_list, 124, 132)
    df_design_matrix['Switching left fingers'] = thumb_pink_switch(samples_list, 103, 111)

    #fingers touching horizontal+ vertically at the same time
    df_design_matrix['Count of the touching of the fingers pos 1'] = open_close_hands(samples_list, 99, 120)
    df_design_matrix['Count of the touching of the fingers pos 2'] = open_close_hands(samples_list, 100, 121)
    df_design_matrix['Count of the touching of the fingers pos 3'] = open_close_hands(samples_list, 101, 122)
    df_design_matrix['Count of the touching of the fingers pos 4'] = open_close_hands(samples_list, 102, 123)
    df_design_matrix['Count of the touching of the fingers pos 5'] = open_close_hands(samples_list, 103, 124)
    
    #hands crossing vertically only
    df_design_matrix['Count of crossing hands vertically finger 1'] = hands_passing_vertical(samples_list, 99, 120)
    df_design_matrix['Count of crossing hands vertically finger 2'] = hands_passing_vertical(samples_list, 103, 124)
    df_design_matrix['Count of crossing hands vertically finger 3'] = hands_passing_vertical(samples_list, 107, 128)
    df_design_matrix['Count of crossing hands vertically finger 4'] = hands_passing_vertical(samples_list, 111, 132)
    df_design_matrix['Count of crossing hands vertically finger 5'] = hands_passing_vertical(samples_list, 115, 136)
    
    #Fingers
    df_design_matrix['Tot reach fingers left hand horizontal'] = total_reach_left_horizontal(samples_list)
    df_design_matrix['Tot reach fingers left hand vertical'] = total_reach_left_vertical(samples_list)
    df_design_matrix['Tot reach fingers right hand horizontal'] = total_reach_right_horizontal(samples_list)
    df_design_matrix['Tot reach fingers right hand vertical'] = total_reach_right_vertical(samples_list)

    avg_reach_values = avg_reach_features(samples_list)
    df_design_matrix['Avg reach fingers left hand horizontal 1st Q'] = avg_reach_values[0][0]
    df_design_matrix['Avg reach fingers left hand vertical 1st Q'] = avg_reach_values[0][1]
    df_design_matrix['Avg reach fingers right hand horizontal 1st Q'] = avg_reach_values[0][2]
    df_design_matrix['Avg reach fingers right hand vertical 1st Q'] = avg_reach_values[0][3]

    df_design_matrix['Avg reach fingers left hand horizontal 2nd Q'] = avg_reach_values[1][0]
    df_design_matrix['Avg reach fingers left hand vertical 2nd Q'] = avg_reach_values[1][1]
    df_design_matrix['Avg reach fingers right hand horizontal 2nd Q'] = avg_reach_values[1][2]
    df_design_matrix['Avg reach fingers right hand vertical 2nd Q'] = avg_reach_values[1][3]

    df_design_matrix['Avg reach fingers left hand horizontal 3th Q'] = avg_reach_values[2][0]
    df_design_matrix['Avg reach fingers left hand vertical 3th Q'] = avg_reach_values[2][1]
    df_design_matrix['Avg reach fingers right hand horizontal 3th Q'] = avg_reach_values[2][2]
    df_design_matrix['Avg reach fingers right hand vertical 3th Q'] = avg_reach_values[2][3]

    df_design_matrix['Avg reach fingers left hand horizontal 4th Q'] = avg_reach_values[3][0]
    df_design_matrix['Avg reach fingers left hand vertical 4th Q'] = avg_reach_values[3][1]
    df_design_matrix['Avg reach fingers right hand horizontal 4th Q'] = avg_reach_values[3][2]
    df_design_matrix['Avg reach fingers right hand vertical 4th Q'] = avg_reach_values[3][3]

    #Frames
    #df_design_matrix['Num frames'] = get_frames(samples_list)
    #Wrist
    keypoint74_values = keypoint_distance_features(samples_list, 7, 4)
    df_design_matrix['Displacement left wrist horizontal 1st Q'] = keypoint74_values[0][0]
    df_design_matrix['Displacement left wrist vertical 1st Q'] = keypoint74_values[0][1]
    df_design_matrix['Displacement right wrist horizontal 1st Q'] = keypoint74_values[0][2]
    df_design_matrix['Displacement right wrist vertical 1st Q'] = keypoint74_values[0][3]
    df_design_matrix['Displacement left wrist horizontal 2nd Q'] = keypoint74_values[1][0]
    df_design_matrix['Displacement left wrist vertical 2nd Q'] = keypoint74_values[1][1]
    df_design_matrix['Displacement right wrist horizontal 2nd Q'] = keypoint74_values[1][2]
    df_design_matrix['Displacement right wrist vertical 2nd Q'] = keypoint74_values[1][3]
    df_design_matrix['Displacement left wrist horizontal 3th Q'] = keypoint74_values[2][0]
    df_design_matrix['Displacement left wrist vertical 3th Q'] = keypoint74_values[2][1]
    df_design_matrix['Displacement right wrist horizontal 3th Q'] = keypoint74_values[2][2]
    df_design_matrix['Displacement right wrist vertical 3th Q'] = keypoint74_values[2][3]
    df_design_matrix['Displacement left wrist horizontal 4th Q'] = keypoint74_values[3][0]
    df_design_matrix['Displacement left wrist vertical 4th Q'] = keypoint74_values[3][1]
    df_design_matrix['Displacement right wrist horizontal 4th Q'] = keypoint74_values[3][2]
    df_design_matrix['Displacement right wrist vertical 4th Q'] = keypoint74_values[3][3]
   
    #Elbows
    keypoint63_values = keypoint_distance_features(samples_list, 6, 3)
    df_design_matrix['Displacement left elbow horizontal 1st Q'] = keypoint63_values[0][0]
    df_design_matrix['Displacement left elbow vertical 1st Q'] = keypoint63_values[0][1]
    df_design_matrix['Displacement right elbow horizontal 1st Q'] = keypoint63_values[0][2]
    df_design_matrix['Displacement right elbow vertical 1st Q'] = keypoint63_values[0][3]
    df_design_matrix['Displacement left elbow horizontal 2nd Q'] = keypoint63_values[1][0]
    df_design_matrix['Displacement left elbow vertical 2nd Q'] = keypoint63_values[1][1]
    df_design_matrix['Displacement right elbow horizontal 2nd Q'] = keypoint63_values[1][2]
    df_design_matrix['Displacement right elbow vertical 2nd Q'] = keypoint63_values[1][3]
    df_design_matrix['Displacement left elbow horizontal 3th Q'] = keypoint63_values[2][0]
    df_design_matrix['Displacement left elbow vertical 3th Q'] = keypoint63_values[2][1]
    df_design_matrix['Displacement right elbow horizontal 3th Q'] = keypoint63_values[2][2]
    df_design_matrix['Displacement right elbow vertical 3th Q'] = keypoint63_values[2][3]
    df_design_matrix['Displacement left elbow horizontal 4th Q'] = keypoint63_values[3][0]
    df_design_matrix['Displacement left elbow vertical 4th Q'] = keypoint63_values[3][1]
    df_design_matrix['Displacement right elbow horizontal 4th Q'] = keypoint63_values[3][2]
    df_design_matrix['Displacement right elbow vertical 4th Q'] = keypoint63_values[3][3]
    #Shoulders
    keypoint52_values = keypoint_distance_features(samples_list, 5, 2)
    df_design_matrix['Displacement left shoulder horizontal 1st Q'] = keypoint52_values[0][0]
    df_design_matrix['Displacement left shoulder vertical 1st Q'] = keypoint52_values[0][1]
    df_design_matrix['Displacement right shoulder horizontal 1st Q'] = keypoint52_values[0][2]
    df_design_matrix['Displacement right shoulder vertical 1st Q'] = keypoint52_values[0][3]
    df_design_matrix['Displacement left shoulder horizontal 2nd Q'] = keypoint52_values[1][0]
    df_design_matrix['Displacement left shoulder vertical 2nd Q'] = keypoint52_values[1][1]
    df_design_matrix['Displacement right shoulder horizontal 2nd Q'] = keypoint52_values[1][2]
    df_design_matrix['Displacement right shoulder vertical 2nd Q'] = keypoint52_values[1][3]
    df_design_matrix['Displacement left shoulder horizontal 3th Q'] = keypoint52_values[2][0]
    df_design_matrix['Displacement left shoulder vertical 3th Q'] = keypoint52_values[2][1]
    df_design_matrix['Displacement right shoulder horizontal 3th Q'] = keypoint52_values[2][2]
    df_design_matrix['Displacement right shoulder vertical 3th Q'] = keypoint52_values[2][3]
    df_design_matrix['Displacement left shoulder horizontal 4th Q'] = keypoint52_values[3][0]
    df_design_matrix['Displacement left shoulder vertical 4th Q'] = keypoint52_values[3][1]
    df_design_matrix['Displacement right shoulder horizontal 4th Q'] = keypoint52_values[3][2]
    df_design_matrix['Displacement right shoulder vertical 4th Q'] = keypoint52_values[3][3]
    #Points of Mouth 60,64: Actually is 24+60/24+64
    #keypoint6064_values = keypoint_distance_features(samples_list, 81,89)
    #df_design_matrix['Displacement left point mouth 60,64 horizontal 1st Q'] = keypoint6064_values[0][0]
    #df_design_matrix['Displacement right point mouth 60,64 horizontal 1st Q'] = keypoint6064_values[0][2]
    #df_design_matrix['Displacement left point mouth 60,64 horizontal 2nd Q'] = keypoint6064_values[1][0]
    #df_design_matrix['Displacement right point mouth 60,64 horizontal 2nd Q'] = keypoint6064_values[1][2]
    #df_design_matrix['Displacement left point mouth 60,64 horizontal 3th Q'] = keypoint6064_values[2][0]
    #df_design_matrix['Displacement right point mouth 60,64 horizontal 3th Q'] = keypoint6064_values[2][2]
    #df_design_matrix['Displacement left point mouth 60,64 horizontal 4th Q'] = keypoint6064_values[3][0]
    #df_design_matrix['Displacement right point mouth 60,64 horizontal 4th Q'] = keypoint6064_values[3][2]
    #Actually is 24+62/24+66
    #Points of Mouth 62,66 (They are really close in vertical if the person does not open his/her mouth)
    #keypoint6266_values = keypoint_distance_features(samples_list, 87,91)
    #df_design_matrix['Displacement left point mouth 62,66 vertical 1st Q'] = keypoint6266_values[0][1]
    #df_design_matrix['Displacement right point mouth 62,66 vertical 1st Q'] = keypoint6266_values[0][3]
    #df_design_matrix['Displacement left point mouth 62,66 vertical 2nd Q'] = keypoint6266_values[1][1]
    #df_design_matrix['Displacement right point mouth 62,66 vertical 2nd Q'] = keypoint6266_values[1][3]
    #df_design_matrix['Displacement left point mouth 62,66 vertical 3th Q'] = keypoint6266_values[2][1]
    #df_design_matrix['Displacement right point mouth 62,66 vertical 3th Q'] = keypoint6266_values[2][3]
    #df_design_matrix['Displacement left point mouth 62,66 vertical 4th Q'] = keypoint6266_values[3][1]
    #df_design_matrix['Displacement right point mouth 62,66 vertical 4th Q'] = keypoint6266_values[3][3]
    #intial fin motion features
    #keypoints_to_check_init_fin_motion = [0, 8, 1, 4, 7, 2, 5, 3, 6, 64, 60, 62, 66]
    keypoints_to_check_init_fin_motion = [0, 8, 1, 4, 7, 2, 5, 3, 6]
    keypoints_to_check_init_fin_motion += [i for i in range(95, 137)]
    start_end_values = [(0, 0.25),(0.25,0.5),(0.5, 0.75),(0.75, 1)]
    for keypoint in keypoints_to_check_init_fin_motion:
        for start_end in start_end_values:
            df_design_matrix["Init Fin Motion keypoint {} start {} end {}".format(keypoint, start_end[0], start_end[1])] = initial_Fin_Motion(samples_list, keypoint, start_end[0], start_end[1])
    
    
   
    #LEFT HAND
    #Left Thumb 
    left_thumb=99
    #Left Index 
    left_index=103
    #RIGHT HAND
    #Right Thumb 
    right_thumb=120
    #Right Index 
    right_index=124
    #chin
    chin_keypoint=78
    #nose
    nose_keypoint=55
    #eyes
    left_eye=94
    right_eye=93
    #Mouth
    #UP
    mouth_point_62=87
    #DOWN
    mouth_point_66=91

    #left Thumb
    #Chin
    df_design_matrix['Minimal distance between left_thumb and chin 1st quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,chin_keypoint)[0]      
    df_design_matrix['Minimal distance between left_thumb and chin 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,chin_keypoint)[1]      
    df_design_matrix['Minimal distance between left_thumb and chin 3th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,chin_keypoint)[2]      
    df_design_matrix['Minimal distance between left_thumb and chin 4th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,chin_keypoint)[3]      
    #Nose
    df_design_matrix['Minimal distance between left_thumb and nose_keypoint 1st quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,nose_keypoint)[0]      
    df_design_matrix['Minimal distance between left_thumb and nose_keypoint 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,nose_keypoint)[1]      
    df_design_matrix['Minimal distance between left_thumb and nose_keypoint 3th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,nose_keypoint)[2]      
    df_design_matrix['Minimal distance between left_thumb and nose_keypoint 4th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,nose_keypoint)[3]      
    #Left eye
    df_design_matrix['Minimal distance between left_thumb and left_eye 1st quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,left_eye)[0]      
    df_design_matrix['Minimal distance between left_thumb and left_eye 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,left_eye)[1]      
    df_design_matrix['Minimal distance between left_thumb and left_eye 3th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,left_eye)[2]      
    df_design_matrix['Minimal distance between left_thumb and left_eye 4th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,left_eye)[3]      
    #Right eye
    df_design_matrix['Minimal distance between left_thumb and right_eye 1st quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,right_eye)[0]      
    df_design_matrix['Minimal distance between left_thumb and right_eye 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,right_eye)[1]      
    df_design_matrix['Minimal distance between left_thumb and right_eye 3th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,right_eye)[2]      
    df_design_matrix['Minimal distance between left_thumb and right_eye 4th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,right_eye)[3]      
    #Mouth 62
    df_design_matrix['Minimal distance between left_thumb and mouth_point_62 1st quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,mouth_point_62)[0]      
    df_design_matrix['Minimal distance between left_thumb and mouth_point_62 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,mouth_point_62)[1]      
    df_design_matrix['Minimal distance between left_thumb and mouth_point_62 3th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,mouth_point_62)[2]      
    df_design_matrix['Minimal distance between left_thumb and mouth_point_62 4th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,mouth_point_62)[3]      
    #Mouth 66
    df_design_matrix['Minimal distance between left_thumb and mouth_point_66 1st quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,mouth_point_66)[0]      
    df_design_matrix['Minimal distance between left_thumb and mouth_point_66 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,mouth_point_66)[1]      
    df_design_matrix['Minimal distance between left_thumb and mouth_point_66 3th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,mouth_point_66)[2]      
    df_design_matrix['Minimal distance between left_thumb and mouth_point_66 4th quarter'] = fing_nose_eyes_chin(samples_list,left_thumb,mouth_point_66)[3]      

    #Left index
    #Chin
    df_design_matrix['Minimal distance between left_index and chin 1st quarter'] = fing_nose_eyes_chin(samples_list,left_index,chin_keypoint)[0]      
    df_design_matrix['Minimal distance between left_index and chin 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_index,chin_keypoint)[1]      
    df_design_matrix['Minimal distance between left_index and chin 3th quarter'] = fing_nose_eyes_chin(samples_list,left_index,chin_keypoint)[2]      
    df_design_matrix['Minimal distance between left_index and chin 4th quarter'] = fing_nose_eyes_chin(samples_list,left_index,chin_keypoint)[3]      
    #Nose
    df_design_matrix['Minimal distance between left_index and nose_keypoint 1st quarter'] = fing_nose_eyes_chin(samples_list,left_index,nose_keypoint)[0]      
    df_design_matrix['Minimal distance between left_index and nose_keypoint 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_index,nose_keypoint)[1]      
    df_design_matrix['Minimal distance between left_index and nose_keypoint 3th quarter'] = fing_nose_eyes_chin(samples_list,left_index,nose_keypoint)[2]      
    df_design_matrix['Minimal distance between left_index and nose_keypoint 4th quarter'] = fing_nose_eyes_chin(samples_list,left_index,nose_keypoint)[3]      
    #Left eye
    df_design_matrix['Minimal distance between left_index and left_eye 1st quarter'] = fing_nose_eyes_chin(samples_list,left_index,left_eye)[0]      
    df_design_matrix['Minimal distance between left_index and left_eye 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_index,left_eye)[1]      
    df_design_matrix['Minimal distance between left_index and left_eye 3th quarter'] = fing_nose_eyes_chin(samples_list,left_index,left_eye)[2]      
    df_design_matrix['Minimal distance between left_index and left_eye 4th quarter'] = fing_nose_eyes_chin(samples_list,left_index,left_eye)[3]      
    #Right eye
    df_design_matrix['Minimal distance between left_index and right_eye 1st quarter'] = fing_nose_eyes_chin(samples_list,left_index,right_eye)[0]      
    df_design_matrix['Minimal distance between left_index and right_eye 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_index,right_eye)[1]    
    df_design_matrix['Minimal distance between left_index and right_eye 3th quarter'] = fing_nose_eyes_chin(samples_list,left_index,right_eye)[2]      
    df_design_matrix['Minimal distance between left_index and right_eye 4th quarter'] = fing_nose_eyes_chin(samples_list,left_index,right_eye)[3]      
    #Mouth 62
    df_design_matrix['Minimal distance between left_index and mouth_point_62 1st quarter'] = fing_nose_eyes_chin(samples_list,left_index,mouth_point_62)[0]      
    df_design_matrix['Minimal distance between left_index and mouth_point_62 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_index,mouth_point_62)[1]      
    df_design_matrix['Minimal distance between left_index and mouth_point_62 3th quarter'] = fing_nose_eyes_chin(samples_list,left_index,mouth_point_62)[2]      
    df_design_matrix['Minimal distance between left_index and mouth_point_62 4th quarter'] = fing_nose_eyes_chin(samples_list,left_index,mouth_point_62)[3]      
    #Mouth 66
    df_design_matrix['Minimal distance between left_index and mouth_point_66 1st quarter'] = fing_nose_eyes_chin(samples_list,left_index,mouth_point_66)[0]      
    df_design_matrix['Minimal distance between left_index and mouth_point_66 2nd quarter'] = fing_nose_eyes_chin(samples_list,left_index,mouth_point_66)[1]      
    df_design_matrix['Minimal distance between left_index and mouth_point_66 3th quarter'] = fing_nose_eyes_chin(samples_list,left_index,mouth_point_66)[2]      
    df_design_matrix['Minimal distance between left_index and mouth_point_66 4th quarter'] = fing_nose_eyes_chin(samples_list,left_index,mouth_point_66)[3]

    #Right Thumb
    #Chin
    df_design_matrix['Minimal distance between right_thumb and chin 1st quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,chin_keypoint)[0]      
    df_design_matrix['Minimal distance between right_thumb and chin 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,chin_keypoint)[1]      
    df_design_matrix['Minimal distance between right_thumb and chin 3th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,chin_keypoint)[2]      
    df_design_matrix['Minimal distance between right_thumb and chin 4th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,chin_keypoint)[3]      
    #Nose
    df_design_matrix['Minimal distance between right_thumb and nose_keypoint 1st quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,nose_keypoint)[0]      
    df_design_matrix['Minimal distance between right_thumb and nose_keypoint 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,nose_keypoint)[1]      
    df_design_matrix['Minimal distance between right_thumb and nose_keypoint 3th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,nose_keypoint)[2]      
    df_design_matrix['Minimal distance between right_thumb and nose_keypoint 4th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,nose_keypoint)[3]      
    #Left eye
    df_design_matrix['Minimal distance between right_thumb and left_eye 1st quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,left_eye)[0]      
    df_design_matrix['Minimal distance between right_thumb and left_eye 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,left_eye)[1]      
    df_design_matrix['Minimal distance between right_thumb and left_eye 3th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,left_eye)[2]      
    df_design_matrix['Minimal distance between right_thumb and left_eye 4th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,left_eye)[3]      
    #Right eye
    df_design_matrix['Minimal distance between right_thumb and right_eye 1st quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,right_eye)[0]      
    df_design_matrix['Minimal distance between right_thumb and right_eye 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,right_eye)[1]      
    df_design_matrix['Minimal distance between right_thumb and right_eye 3th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,right_eye)[2]      
    df_design_matrix['Minimal distance between right_thumb and right_eye 4th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,right_eye)[3]      
    #Mouth 62
    df_design_matrix['Minimal distance between right_thumb and mouth_point_62 1st quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,mouth_point_62)[0]      
    df_design_matrix['Minimal distance between right_thumb and mouth_point_62 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,mouth_point_62)[1]      
    df_design_matrix['Minimal distance between right_thumb and mouth_point_62 3th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,mouth_point_62)[2]      
    df_design_matrix['Minimal distance between right_thumb and mouth_point_62 4th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,mouth_point_62)[3]      
    #Mouth 66
    df_design_matrix['Minimal distance between right_thumb and mouth_point_66 1st quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,mouth_point_66)[0]      
    df_design_matrix['Minimal distance between right_thumb and mouth_point_66 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,mouth_point_66)[1]      
    df_design_matrix['Minimal distance between right_thumb and mouth_point_66 3th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,mouth_point_66)[2]      
    df_design_matrix['Minimal distance between right_thumb and mouth_point_66 4th quarter'] = fing_nose_eyes_chin(samples_list,right_thumb,mouth_point_66)[3]

    #Right index
    #Chin
    df_design_matrix['Minimal distance between right_index and chin 1st quarter'] = fing_nose_eyes_chin(samples_list,right_index,chin_keypoint)[0]      
    df_design_matrix['Minimal distance between right_index and chin 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_index,chin_keypoint)[1]      
    df_design_matrix['Minimal distance between right_index and chin 3th quarter'] = fing_nose_eyes_chin(samples_list,right_index,chin_keypoint)[2]      
    df_design_matrix['Minimal distance between right_index and chin 4th quarter'] = fing_nose_eyes_chin(samples_list,right_index,chin_keypoint)[3]      
    #Nose
    df_design_matrix['Minimal distance between right_index and nose_keypoint 1st quarter'] = fing_nose_eyes_chin(samples_list,right_index,nose_keypoint)[0]      
    df_design_matrix['Minimal distance between right_index and nose_keypoint 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_index,nose_keypoint)[1]      
    df_design_matrix['Minimal distance between right_index and nose_keypoint 3th quarter'] = fing_nose_eyes_chin(samples_list,right_index,nose_keypoint)[2]      
    df_design_matrix['Minimal distance between right_index and nose_keypoint 4th quarter'] = fing_nose_eyes_chin(samples_list,right_index,nose_keypoint)[3]      
    #Left eye
    df_design_matrix['Minimal distance between right_index and left_eye 1st quarter'] = fing_nose_eyes_chin(samples_list,right_index,left_eye)[0]      
    df_design_matrix['Minimal distance between right_index and left_eye 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_index,left_eye)[1]      
    df_design_matrix['Minimal distance between right_index and left_eye 3th quarter'] = fing_nose_eyes_chin(samples_list,right_index,left_eye)[2]      
    df_design_matrix['Minimal distance between right_index and left_eye 4th quarter'] = fing_nose_eyes_chin(samples_list,right_index,left_eye)[3]      
    #Right eye
    df_design_matrix['Minimal distance between right_index and right_eye 1st quarter'] = fing_nose_eyes_chin(samples_list,right_index,right_eye)[0]      
    df_design_matrix['Minimal distance between right_index and right_eye 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_index,right_eye)[1]    
    df_design_matrix['Minimal distance between right_index and right_eye 3th quarter'] = fing_nose_eyes_chin(samples_list,right_index,right_eye)[2]      
    df_design_matrix['Minimal distance between right_index and right_eye 4th quarter'] = fing_nose_eyes_chin(samples_list,right_index,right_eye)[3]      
    #Mouth 62
    df_design_matrix['Minimal distance between right_index and mouth_point_62 1st quarter'] = fing_nose_eyes_chin(samples_list,right_index,mouth_point_62)[0]      
    df_design_matrix['Minimal distance between right_index and mouth_point_62 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_index,mouth_point_62)[1]      
    df_design_matrix['Minimal distance between right_index and mouth_point_62 3th quarter'] = fing_nose_eyes_chin(samples_list,right_index,mouth_point_62)[2]      
    df_design_matrix['Minimal distance between right_index and mouth_point_62 4th quarter'] = fing_nose_eyes_chin(samples_list,right_index,mouth_point_62)[3]      
    #Mouth 66
    df_design_matrix['Minimal distance between right_index and mouth_point_66 1st quarter'] = fing_nose_eyes_chin(samples_list,right_index,mouth_point_66)[0]      
    df_design_matrix['Minimal distance between right_index and mouth_point_66 2nd quarter'] = fing_nose_eyes_chin(samples_list,right_index,mouth_point_66)[1]      
    df_design_matrix['Minimal distance between right_index and mouth_point_66 3th quarter'] = fing_nose_eyes_chin(samples_list,right_index,mouth_point_66)[2]      
    df_design_matrix['Minimal distance between right_index and mouth_point_66 4th quarter'] = fing_nose_eyes_chin(samples_list,right_index,mouth_point_66)[3]
    #HAND DIRECTIONS
    #1st quarter
    #df_design_matrix['Left Hand direction 1st quarter'] = hand_dir(samples_list)[0][0]      
    #df_design_matrix['Right Hand direction 1st quarter'] = hand_dir(samples_list)[0][1]  
    #2nd quarter
    #df_design_matrix['Left Hand direction 2nd quarter'] = hand_dir(samples_list)[0][0]      
    #df_design_matrix['Right Hand direction 2nd quarter'] = hand_dir(samples_list)[0][1]  
    #3th quarter
    #df_design_matrix['Left Hand direction 3th quarter'] = hand_dir(samples_list)[0][0]      
    #df_design_matrix['Right Hand direction 3th quarter'] = hand_dir(samples_list)[0][1]  
    #4th quarter
    #df_design_matrix['Left Hand direction 4th quarter'] = hand_dir(samples_list)[0][0]      
    #df_design_matrix['Right Hand direction 4th quarter'] = hand_dir(samples_list)[0][1]  
    #5th quarter
    #df_design_matrix['Left Hand direction 5th quarter'] = hand_dir(samples_list)[0][0]      
    #df_design_matrix['Right Hand direction 5th quarter'] = hand_dir(samples_list)[0][1]  
    #6th quarter
    #df_design_matrix['Left Hand direction 6th quarter'] = hand_dir(samples_list)[0][0]      
    #df_design_matrix['Right Hand direction 6th quarter'] = hand_dir(samples_list)[0][1]  
    #7th quarter
    #df_design_matrix['Left Hand direction 7th quarter'] = hand_dir(samples_list)[0][0]      
    #df_design_matrix['Right Hand direction 7th quarter'] = hand_dir(samples_list)[0][1]  
    #8th quarter
    #df_design_matrix['Left Hand direction 8th quarter'] = hand_dir(samples_list)[0][1]      
    #df_design_matrix['Right Hand direction 8th quarter'] = hand_dir(samples_list)[0][1] 
    return df_design_matrix
