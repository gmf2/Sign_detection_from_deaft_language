import numpy as np
import csv


# Source for ap@k and map@k code: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
# License for the original code:

# Copyright (c) 2012, Ben Hamner
# Author: Ben Hamner (ben@benhamner.com)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# The code below was modified to only require a single true label

def get_ordered_predictions(probabilities):
    """For every sample in X, return a list of predictions.
    Per sample, the predictions are ordered via their corresponding (descending) probability.

    :param probabilities predicted by your classifier.
    :param X: A numpy array of shape `(n_samples, n_features)`.
    :returns: A numpy array of shape `(n_samples, n_classes)`.
    """
    return np.flip(np.argsort(probabilities, axis=1), axis=1)

def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average precision at k for single predictions.
    Parameters
    ----------
    actual : int
             The true label
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score


def mapk(probabilities, actual, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k for multiple predictions.
    Parameters
    ----------
    actual : list
             A list of true labels
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    predicted = get_ordered_predictions(probabilities)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def top3_accuracy(probabilities, y):
    """Calculates the top-3 accuracy of an estimator, given an array or probabilities
    of shape `(n_samples, n_classes)` and an array of integer labels of shape
    `(n_samples,)`.

    :param p: An array of probabilities (model output).
    :param y: An array of integer labels (ground truth).
    """
    p = get_ordered_predictions(probabilities)
    n_samples = p.shape[0]
    # Get the top 3 predictions per sample.
    top_3 = p[:,:3]
    correct = 0
    for sample_index in range(n_samples):
        # The elementwise equality comparison returns an array of boolean values.
        # If any of them is `True`, the true label was in the top 3.
        if np.any(top_3[sample_index] == y[sample_index]):
            correct += 1
    return correct / n_samples

def create_submission(probabilities, path):
    """Create a submission file on the given path.

    :param p: The output of the `get_ordered_predictions` function (A numpy array of shape `(n_samples, n_classes)`).
    :param path: The path to the output CSV file.
    """
    p = get_ordered_predictions(probabilities)
    p = p.tolist()
    with open(path, 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['Id', 'Predicted'])
        for i in range(len(p)):
            writer.writerow([str(i), ' '.join([str(e) for e in p[i]])])


