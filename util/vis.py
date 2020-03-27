import matplotlib.pyplot as plt
import numpy as np

a_hand = np.load('util/a_hand.npy')
a_face = np.load('util/a_face.npy')
a_body = np.load('util/a_body.npy')

def visualize(pose):
    """
    Plot a pose using matplotlib.

    :param pose: A numpy array of shape `(3, 137)`.
    """
    x = pose[:, 0]
    y = pose[:, 1]

    plt.figure()
    plt.xticks([])
    plt.yticks([])
    _plot_bodypart(a_body, x, y, range(25), 0, 'red')
    _plot_bodypart(a_face, x, y, range(70), 25, 'blue')
    _plot_bodypart(a_hand, x, y, range(21), 95, 'green')
    _plot_bodypart(a_hand, x, y, range(21), 116, 'magenta')
    
def _plot_bodypart(adjacency, x, y, index_range, index_offset, color):
    for i in index_range:
        for j in index_range:
            if i != j and adjacency[i, j] == 1:
                # The y coordinate is flipped, because the origin lies in the top left corner in OpenPose
                plt.plot([x[i + index_offset], x[j + index_offset]], [-1*y[i + index_offset], -1*y[j + index_offset]], c=color)