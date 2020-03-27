import numpy as np
import random

def mirror(sample):
#    mir = np.arange(sample.shape[0]*137*3).reshape(sample.shape[0], 137, 3) #create an empty copy
    mir=np.zeros((sample.shape[0],137,3))

    for i in range(sample.shape[0]):
        for e in range(137):
            mir[i][e][0] = float(455) - sample[i][e][0] #mirror the x-coordinates cause input images consist of 455 by 256 pixels
            mir[i][e][1] = sample[i][e][1]
            mir[i][e][2] = sample[i][e][2]
            
    return mir #return the mirrored array

def move_left_hand(sample, x, y):
    """
    Augmentation function to move the left hand an amount determined by x and y
    
    """
    hand_left_offset = 25 + 70
    hand_left_length = 21
    output = np.copy(sample)
    #moving hand
    output[:, hand_left_offset:hand_left_offset+hand_left_length, 0] += x
    output[:, hand_left_offset:hand_left_offset+hand_left_length, 1] += y
    
    #wrist
    output[:, 7, 0] += x
    output[:, 7, 1] += y
    
    return output

def move_both_hands(sample):
    """
    Augmentation function to move both hands by a random amount of x and y
    
    """
    #Randomisation
    hor_max = 20
    hor_min = -20
    ver_min = -20
    ver_max = 20
    scaling_min = 1.1
    scaling_max = 1.5
    
    x = random.randint(hor_min, hor_max)
    y = random.randint(ver_min, ver_max)
    
    #Offsets
    hand_left_offset = 25 + 70
    hand_left_length = 21
    hand_right_offset = 25 + 70 + 21
    hand_right_length = 21
    output = np.copy(sample)
    
    #Moving hands
    output[:, hand_left_offset:hand_left_offset+hand_left_length, 0] += x
    output[:, hand_left_offset:hand_left_offset+hand_left_length, 1] += y
    output[:, hand_right_offset:hand_right_offset+hand_right_length, 0] += x
    output[:, hand_right_offset:hand_right_offset+hand_right_length, 1] += y
    
    #moving wrists
    output[:, 7, 0] += x
    output[:, 7, 1] += y
    output[:, 4, 0] += x
    output[:, 4, 1] += y

    return output

def move_right_hand(sample, x, y):
    """
    Augmentation function to move the right hand an amount determined by x and y
    
    """
    hand_right_offset = 25 + 70 + 21
    hand_right_length = 21
    output = np.copy(sample)
    #moving hand
    output[:, hand_right_offset:hand_right_offset+hand_right_length, 0] += x
    output[:, hand_right_offset:hand_right_offset+hand_right_length, 1] += y
    
    #wrist
    output[:, 4, 0] += x
    output[:, 4, 1] += y
    
    return output

def scale_person(sample, factor):
    """
    Augmentation function to scale the body of the person by a factor factor away from a calculated reference point
    
    """
    output = np.copy(sample)
    for frame in range(sample.shape[0]):
        ref_point = (output[frame, 1, 0] + output[frame, 8, 0] + output[frame, 11, 0])/3, (output[frame, 1, 1] + output[frame, 8, 1] + output[frame, 11, 1])/3
        output[frame, :, 0] = ref_point[0] + (output[frame, :, 0] - ref_point[0])*factor
        output[frame, :, 1] = ref_point[1] + (output[frame, :, 1] - ref_point[1])*factor
    
    return output

def augment_data(all_samples):
    extra_samples_shift_hands = []
    #extra_samples_mirror = []
    extra_samples_scaling = []
    for sample_index in range(len(all_samples)):
        sample = all_samples[sample_index]
        if sample_index%4 == 0:
            extra_sample_shift = move_left_hand(sample, -11, -10)
        elif sample_index%4 == 1:
            extra_sample_shift = move_right_hand(sample, 18, 13)
        elif sample_index%4 == 2:
            extra_sample_shift = move_left_hand(sample, 17, 15)
        else:
            extra_sample_shift = move_right_hand(sample, -16, -18)      
        extra_samples_shift_hands.append(extra_sample_shift)
        #extra_samples_mirror.append(mirror(sample))
        extra_samples_scaling.append(scale_person(sample, 1.56))
    return np.concatenate((all_samples, extra_samples_shift_hands, extra_samples_scaling))

def augment_data_random(all_samples, hor_min = -20, hor_max = 20, ver_min = -20, ver_max = 20, scaling_min = 1.1, scaling_max = 1.5):
    """
    Augmentation function that does random transformations on the samples instead of fixed by index
    
    The extra parameters determine the ranges shifting and scaling can take
    """
    extra_samples_shift_hands = []
    #extra_samples_mirror = []
    extra_samples_scaling = []
    for sample_index in range(len(all_samples)):
        sample = all_samples[sample_index]
    	#randomness
        hand_choice = random.randint(0, 1)
        shift_hor = random.randint(hor_min, hor_max)
        shift_ver = random.randint(ver_min, ver_max)
        scaling_factor = random.uniform(scaling_min, scaling_max)
        
        if hand_choice:
            extra_sample_shift = move_left_hand(sample, shift_hor, shift_ver)
        else:
            extra_sample_shift = move_right_hand(sample, shift_hor, shift_ver)
        extra_samples_shift_hands.append(extra_sample_shift)
        #extra_samples_mirror.append(mirror(sample))
        extra_samples_scaling.append(scale_person(sample, scaling_factor))
    return np.concatenate((all_samples, extra_samples_shift_hands, extra_samples_scaling))
