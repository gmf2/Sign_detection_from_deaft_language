import numpy as np

def centering(all_samples):
    """
    Function that centers (horizontally) all the samples and returns an array containing a centered version of all the samples

    :param all_samples: the array containing all the samples
    """
    center_frame_hor = 455/2
    center_frame_ver = 95
    #do translations based on the pose of the first frame
    for sample_index in range(len(all_samples)):
        hor_offset = all_samples[sample_index][0][1][0] - center_frame_hor #Horizontal centering, always take first frame, and keypoint 1 of the pose to perform recentering
        ver_offset = all_samples[sample_index][0][1][1] - center_frame_ver #Vertical centering, always take first frame, and keypoint 1 of the pose to perform recentering

        
        for frame_index in range(len(all_samples[sample_index])):
            for keypoint_index in range(len(all_samples[sample_index][frame_index])):
                #Horizontal centering
                if all_samples[sample_index][frame_index][keypoint_index][0] != 0.0:
                    all_samples[sample_index][frame_index][keypoint_index][0] -= hor_offset
                #Vertical centering
                if all_samples[sample_index][frame_index][keypoint_index][1] != 0.0:
                    all_samples[sample_index][frame_index][keypoint_index][1] -= ver_offset
    print("--- Centering finished ---")
    return all_samples

def scale(all_samples, desired_width_shoulders=70, desired_height_neck=35):
    """
    Function that scales (both horizontally and vertically) all the samples and returns an array containing a scaled version of all the samples

    :param all_samples: the array containing all the samples
    :param desired_width_shoulders: the width we want of the shoulders for each sample
    :param desired_height_neck: the width we want of the shoulders for each sample
    """
    for sample_index in range(len(all_samples)):
        width_shoulders = all_samples[sample_index][0][5][0] - all_samples[sample_index][0][2][0] #keypoint 5 is the left shoulder, keypoint 2 is the right shoulder
        if all_samples[sample_index][0][1][1] - all_samples[sample_index][0][0][1] > all_samples[sample_index][0][1][0] - all_samples[sample_index][0][0][0]:
            height_neck = all_samples[sample_index][0][1][1] - all_samples[sample_index][0][0][1]
        else:
            height_neck = all_samples[sample_index][0][1][0] - all_samples[sample_index][0][0][0]
        scale_factor_hor = desired_width_shoulders/width_shoulders
        scale_factor_ver = desired_height_neck/height_neck

        for frame_index in range(len(all_samples[sample_index])):
            for keypoint in range(137):
                all_samples[sample_index][frame_index][keypoint][0] *= scale_factor_hor
                all_samples[sample_index][frame_index][keypoint][1] *= scale_factor_ver        
    print("--- Scaling finished ---")
    return all_samples

def rotate(all_samples):
    """
    Function that rotates all the samples such that the line that connects both shoulders is horizontal

    :param all_samples: the array containing all the samples
    """
    for sample_index in range(len(all_samples)):
        vector1 = all_samples[sample_index][0][5][:2] - all_samples[sample_index][0][2][:2]
        vector2 = np.array([all_samples[sample_index][0][2][0]+10, all_samples[sample_index][0][2][1]]) - all_samples[sample_index][0][2][:2]
        vector1 =  vector1 / np.linalg.norm(vector1)
        vector2 =  vector2 / np.linalg.norm(vector2)
        if all_samples[sample_index][0][5][1] > all_samples[sample_index][0][2][1]:
            angle = -np.arccos(np.clip(np.dot(vector1, vector2), -1.0, 1.0))
        else:
            angle = np.arccos(np.clip(np.dot(vector1, vector2), -1.0, 1.0))
        for frame_index in range(len(all_samples[sample_index])):
            for keypoint in range(137):
                if all_samples[sample_index][frame_index][keypoint][0] != 0 and  all_samples[sample_index][frame_index][keypoint][1] != 0:
                    x = all_samples[sample_index][frame_index][keypoint][0]
                    y = all_samples[sample_index][frame_index][keypoint][1]
                    all_samples[sample_index][frame_index][keypoint][0] = x * np.cos(angle) - y * np.sin(angle)
                    all_samples[sample_index][frame_index][keypoint][1] = x * np.sin(angle) + y * np.cos(angle)
    print("--- Rotating finished ---")
    return all_samples
