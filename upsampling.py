# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:25:09 2019

@author: Robbe Adriaens
"""

import numpy as np

#%load_ext autoreload
#%autoreload 1
import augmentation
import random


def keep_sample(sample):
    counter = 0
    no_hands_present = False
    for frame in sample:
        if (frame[4][0] == 0 or frame[7][0] == 0):
            counter += 1
        if (frame[4][0] == 0 and frame[7][0] == 0):
            no_hands_present = True
    if counter/len(sample) > 0.5 or no_hands_present:
        return False
    else:
        return True

def downsample(all_samples, all_labels, all_persons):
    """
    Functions that samples down untill every label has the same amount of samples
    
    :param all_samples: all the samples that need to be sampled down
    :param all_labels: the corresponding labels with the samples    
    """
    
    tot_label_list = [0 for i in range(18)]
    for label in all_labels:
        tot_label_list[int(label)] += 1
    max_labels = min(tot_label_list)
    new_samples = []
    new_labels = []
    new_persons = []
    new_labellist = [0 for i in range(18)]
    for i in range(len(all_samples)):
        sample = all_samples[i]
        if (new_labellist[int(all_labels[i])] < max_labels):
            new_samples.append(all_samples[i])
            new_labels.append(all_labels[i])
            new_labellist[int(all_labels[i])] += 1
            new_persons.append(all_persons[i])
    return new_samples, new_labels, new_labellist, new_persons

def prep_test_split(all_persons, all_personlabels):
    """
    Function that return some variables that help to split the data.
    
    :return return a person array and a array with how many samples each person has made for each label    
    """
    persons = []
    for i in all_persons:
        if i not in persons:
            persons.append(i)
    num_persons = []
    for i in persons:
        num_persons.append(np.count_nonzero(all_persons == i))
        
        
    label_list_all_persons = []
    for person in persons:
        label_list = []
        for sample in all_personlabels:
            if sample[1] == person:
                label_list.append(sample[0])
        tot_label_list = []
        for i in range(18):
            tot_label_list.append(0)
        for label in label_list:
            tot_label_list[int(label)] += 1
        label_list_all_persons.append(tot_label_list)
        
    return persons, label_list_all_persons

def get_tot_label_list(all_persons, all_personlabels):
    """
    Get the total_label_list that is being used when upsampling your data
    
    :return return the tot_label_list that is being used when you want to upsample your data
    
    """
    persons, label_list_all_persons = prep_test_split(all_persons, all_personlabels)
    tot_label_list = []
    for i in range(18):
        tot_label_list.append(0)
    for labellist in label_list_all_persons:
        for i in range(18):
            tot_label_list[i] += labellist[i]
    return tot_label_list

def upsample_help(all_samples, all_labels, all_persons, all_personlabels):
    """
    HELP function for the total upsampling function. This function checks whether or not upsampling is needed for
    samples with a low amount of frames
    Samples the labels with the lowest amount up to the number of labels from the most common type
    
    :param all_samples: samplearray
    :param all_labels:
    :return an array with the total new samples, new labels and a new tot_label_list
    """
    tot_label_list = get_tot_label_list(all_persons, all_personlabels)
    upsampled_samples = [sample for sample in all_samples]
    upsampled_labels = [label for label in all_labels]
    upsampled_persons = [person for person in all_persons]
    #print(len(all_samples), len(all_labels))
    #print(len(upsampled_samples), len(upsampled_labels))
    max_labels = max(tot_label_list)
    cur_label_list = [tot_label_list[e] for e in range(18)] #make a list that counts the amount of labels
    
    scaling_min = 1.1
    scaling_max = 1.5
    
    for i in range(len(all_samples)):
        if (cur_label_list[int(all_labels[i])] < max_labels): #if there are not enough samples yet for that label
            cur_label_list[int(all_labels[i])] += 1 #count the label
            
            sample = all_samples[i]

            extra_sample_shift = augmentation.move_both_hands(sample) 
            
            upsampled_samples.append(extra_sample_shift)
            upsampled_labels.append(all_labels[i])
            upsampled_persons.append(all_persons[i])
            
    for i in range(len(all_samples)):
        if (cur_label_list[int(all_labels[i])] < max_labels): #if there are not enough samples yet for that label
            cur_label_list[int(all_labels[i])] += 1 #count the label
            
            sample = all_samples[i]
            #Get the random value
            scaling_factor = random.uniform(scaling_min, scaling_max)

            extra_sample_shift = augmentation.scale_person(sample, scaling_factor)   
            upsampled_samples.append(extra_sample_shift)
            upsampled_labels.append(all_labels[i])
            upsampled_persons.append(all_persons[i])
            
            
    for i in range(len(all_samples)):
        if (cur_label_list[int(all_labels[i])] < max_labels): #if there are not enough samples yet for that label
            cur_label_list[int(all_labels[i])] += 1 #count the label
            
            sample = all_samples[i]

            extra_sample_shift = augmentation.move_both_hands(sample) #this is random so it does not matter that it is re-used
            
            upsampled_samples.append(extra_sample_shift)
            upsampled_labels.append(all_labels[i])
            upsampled_persons.append(all_persons[i])

    return (upsampled_samples, upsampled_labels, cur_label_list, upsampled_persons)

def upsample(all_samples, all_labels, all_persons, all_personlabels):
    """
    This function upsampled the lowest labels and when all the labels are even it does augmentation on the new sampleset
    """
    upsampled_samples, upsampled_labels, upsampled_label_list, upsampled_persons = upsample_help(all_samples, all_labels, all_persons, all_personlabels)#upsampling the lowest ones
    #print(len(upsampled_samples), len(all_samples), len(all_labels))
    
    """
    #This part is already balanced and uses data augmentation on every samples that is now available
    for i in range(len(upsampled_samples)): #add the scaling to every sample too
        upsampled_samples.append(augmentation.scale_person(upsampled_samples[i], 1.56))
        upsampled_labels.append(upsampled_labels[i])
        upsampled_persons.append(upsampled_persons[i])
        upsampled_label_list[int(upsampled_labels[i])] += 1
    """
    return (upsampled_samples, upsampled_labels, upsampled_label_list, upsampled_persons)