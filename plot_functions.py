# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:19:03 2019

@author: Robbe Adriaens
"""

import matplotlib.pyplot as plt
import numpy as np
print("libraries ready")

plottie = 1117

x = [i for i in range(all_samples[plottie].shape[0])]
val_pose = 0
val_face = 0
val_hands = 0
val_left = 0
val_right = 0
pose_values = []
face_values = []
hands_values = []
left_values = []
right_values = []


for i in range(all_samples[plottie].shape[0]): #get number of frames
    avg_pose = sum([all_samples[plottie][i][e][2] for e in range(25)])/25
    avg_face = sum([all_samples[plottie][i][e][2] for e in range(25, 95)])/70
    avg_hands = sum([all_samples[plottie][i][e][2] for e in range(95, 137)])/42
    avg_left = sum([all_samples[plottie][i][e][2] for e in range(95, 116)])/21
    avg_right = sum([all_samples[plottie][i][e][2] for e in range(116, 137)])/21

    #print(avg_pose, avg_face, avg_hands)
    pose_values.append(avg_pose)
    face_values.append(avg_face)
    hands_values.append(avg_hands)
    left_values.append(avg_left)
    right_values.append(avg_right)
print("values extracted")


plt.plot(x, pose_values, label = 'Pose')
plt.plot(x, face_values, label = 'Face')
#plt.plot(x, hands_values, label = 'Hands')
plt.plot(x, left_values, label = 'Left Hand')
plt.plot(x, right_values, label = 'Right Hand')


plt.ylim(0, 1.0)
plt.ylabel('Confidence level ')
plt.xlabel('Framenumber')
plt.title('Confidence levels of keypoints')
plt.grid(False)
plt.legend()
plt.savefig("ConfidenceLevels"+str(plottie)+".png")
plt.show()

persons = []
for i in all_persons:
    if i not in persons:
        persons.append(i)
num_persons = []
for i in persons:
    num_persons.append(np.count_nonzero(all_persons == i))
    #print(np.count_nonzero(all_persons == i))
#print(num_persons)


plt.bar([i for i in range(59)], num_persons)
plt.xlabel("Person id")
plt.ylabel("Amount of videos")
plt.title('Videos created per person')
plt.savefig("Videos_per_person.png")
plt.show()


















