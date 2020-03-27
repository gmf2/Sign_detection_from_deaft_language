import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
import random

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def add_test(test_indices, samplearray, lablearray):
    test_samples = []
    test_labels = []
    for i in range(len(samplearray)):
        if i in test_indices:
            test_samples.append(samplearray[i])
            test_labels.append(lablearray[i])
    test_samples = np.array(test_samples)
    test_labels = np.array(test_labels)
    return (test_samples, test_labels)

def prep_test_split(all_persons, all_personlabels):
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
            #label_list.append(np.count_nonzero(all_personlabels[1] == sample[0]))
        #print(label_list)
        tot_label_list = []
        for i in range(18):
            tot_label_list.append(0)
        for label in label_list:
            tot_label_list[int(label)] += 1
        #print(tot_label_list)
        #plt.bar([i for i in range(18)], tot_label_list)
        #plt.xlabel("label id")
        #plt.ylabel("Amount of videos")
        #titel = ("videos created for person %s" , person)
        #plt.title(titel)
        #plt.savefig("Videos_per_person.png")
        #plt.show()
        label_list_all_persons.append(tot_label_list)
        #print("person done %s" % person)
        
    return persons, label_list_all_persons
        
#print(prep_test_split())

def get_tot_label_list():
    persons, label_list_all_persons = prep_test_split()
    tot_label_list = []
    for i in range(18):
        tot_label_list.append(0)
    for labellist in label_list_all_persons:
        for i in range(18):
            tot_label_list[i] += labellist[i]
    #print(tot_label_list, sum(tot_label_list))
    #print(all_labels[0])
    return tot_label_list

def upsample(all_samples, all_labels, tot_label_list):
    upsampled_samples = []
    upsampled_labels = []
    max_labels = max(tot_label_list)
    cur_label_list = [tot_label_list[e] for e in range(18)] #make a list that counts the amount of labels
    
    for i in range(len(all_samples)):
        if (cur_label_list[int(all_labels[i])] < max_labels): #if there are not enough samples yet for that label
            upsampled_samples.append(all_samples[i]) ##add the data
            upsampled_labels.append(all_labels[i]) #add the labels
            cur_label_list[int(all_labels[i])] += 2 #count the label + label for the mirroring
            
            upsampled_samples.append(mirror(all_samples[i])) # add mirrored version
            upsampled_labels.append(all_labels[i])
            
    upsampled_samples = np.array(upsampled_samples)
    upsampled_labels = np.array(upsampled_labels)
    print(cur_label_list)
    return (upsampled_samples, upsampled_labels, cur_label_list)



def balance_training(all_samples, all_labels, tot_label_list):
    new_training_samples = []
    new_training_labels = []
    max_labels = min(tot_label_list)
    cur_label_list = [0 for e in range(18)] #make a list that counts the amount of labels
    
    for i in range(len(all_labels)):
        if (cur_label_list[int(all_labels[i])] < max_labels): #if there are not enough samples yet for that label
            new_training_samples.append(all_samples[i]) ##add the data
            new_training_labels.append(all_labels[i]) #add the labels
            cur_label_list[int(all_labels[i])] += 1 #count the label
            
    new_training_samples = np.array(new_training_samples)
    new_training_labels = np.array(new_training_labels)
    return (new_training_samples, new_training_labels)

#upsampled_samples, upsampled_labels, upsampled_label_list = upsample(all_samples, all_labels, tot_label_list) ##upsampling
#balanced_samples, balanced_labels = balance_training(upsampled_samples, upsampled_labels, upsampled_label_list) #balance with upsampling
#balanced_samples, balanced_labels = balance_training(all_samples, all_labels, tot_label_list) #### balance without upsampling
    
def check_upper_balance(test_labels, maximum, new_labels):
    #print(test_labels, new_labels)
    for i in range(len(test_labels)):
        if (test_labels[i] + new_labels[i] > maximum):
            if (i != 0 and i != 1):
                return False
        for e in range(10):
            if (new_labels[e] > 10): return False
        zero_count = 0
        for e in range(10, 18):
            if (new_labels[e] < 1): zero_count += 1
        if (zero_count > 4): return False
    return True
def check_lower_balance(test_labels, minimum):
    for i in range(len(test_labels)):
        if (test_labels[i] < minimum):
            return False
    return True

def add_labels(test_indices, test_samples, test_labels, all_persons, samplearray, new_labels, person):
    #print("before adding: " , test_labels, "with ", new_labels)
    for i in range(len(samplearray)):
        if all_persons[i] == person: # if the right person is found
            #print("FOUND")
            test_samples.append(samplearray[i])
            test_indices.append(i)
    #adjust labels correctly
    for i in range(len(new_labels)):
        test_labels[i] += new_labels[i] 
    #print("after adding " , test_labels)
    return (test_samples, test_labels)

def add_training(test_indices, samplearray, lablearray):
    train_samples = []
    train_labels = []
    for i in range(len(samplearray)):
        if i not in test_indices:
            train_samples.append(samplearray[i])
            train_labels.append(lablearray[i])
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    return (train_samples, train_labels)


def test_group(samplearray, labelarray, label_array_per_person, persons, num_diff_labels, all_persons):
    sample_min = (len(samplearray) // num_diff_labels) // 2
    sample_max = sample_min + 10
    print(sample_min, sample_max)
    test_samples = []
    test_labels = []    
    train_samples = []
    train_labels = []
    test_indices = []
    for i in range(18):
        test_labels.append(0)
    used_persons = []
    balanced = False
    index = 0
    while (not balanced):
        for person in range(len(persons)):
            if (check_upper_balance(test_labels, sample_max, label_array_per_person[person])): #add the labels
                used_persons.append(persons[person]) #add person to used person list
                test_samples, test_labels = add_labels(test_indices, test_samples, test_labels, all_persons, samplearray, label_array_per_person[person], str(persons[person]))
        
        if (check_lower_balance(test_labels, sample_min)): 
            balanced = True
            break
        else:
            break
    #print(test_labels)
    #print(used_persons)
    train_samples, train_labels = add_training(test_indices, samplearray, labelarray)
    test_samples, test_labels = add_test(test_indices, samplearray, labelarray)
    return (test_samples, test_labels, train_samples, train_labels)
