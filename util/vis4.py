import matplotlib.pyplot as plt
import numpy as np
import os, sys

a_hand = np.load('util/a_hand.npy')
a_face = np.load('util/a_face.npy')
a_body = np.load('util/a_body.npy')

def visualize(pose, sample_id, frame_id, person_id = 0, label = 0):
    """
    Plot a pose using matplotlib.

    :param pose: A numpy array of shape `(3, 137)`.
    """
    label_list = ['Done', 'To have', 'What', 'Same', 'Have to', 'To drive a car', 'Towards', '1', 'Too', 'To say', 'Good', '2', 'To Arrive', 'First', 'To see', 'Real', 'Yes', 'Not']
	 

    x = pose[:, 0]
    y = pose[:, 1]

    fig = plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.title("sample: {}, label: {} {}, person: {}, frame: {}".format(sample_id, label, label_list[int(label)], person_id, frame_id))
    _plot_bodypart(a_body, x, y, range(25), 0, 'red', 'body')
    _plot_bodypart(a_face, x, y, range(70), 25, 'blue', 'face')
    _plot_bodypart(a_hand, x, y, range(21), 95, 'green', 'l_hand')
    _plot_bodypart(a_hand, x, y, range(21), 116, 'magenta', 'r_hand')
    label_path = 'plots/'+str(label)+'/'
    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    path = 'plots/'+str(label)+'/'+str(sample_id)+'/'
    if not os.path.isdir(path):
        os.mkdir(path);
    plt.savefig(path+str(frame_id)+'.png')
    plt.close(fig)
    
def _plot_bodypart(adjacency, x, y, index_range, index_offset, color, pos):
    for i in index_range:
        for j in index_range:
            if i != j and adjacency[i, j] == 1:
                # The y coordinate is flipped, because the origin lies in the top left corner in OpenPose
                plt.plot([x[i + index_offset], x[j + index_offset]], [-1*y[i + index_offset], -1*y[j + index_offset]], c=color)
                