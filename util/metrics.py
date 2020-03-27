import numpy as np

def top3_accuracy(p, y):
    """Calculates the top-3 accuracy of an estimator, given an array or probabilities
    of shape `(n_samples, n_classes)` and an array of integer labels of shape
    `(n_samples,)`.

    :param p: An array of probabilities (model output).
    :param y: An array of integer labels (ground truth).
    """
    n_samples = p.shape[0]
    # Get the last 3 indices of the sorted array (ascending order) per sample.
    top_3 = np.argsort(p, axis=1)[:, -3:]
    correct = 0
    for sample_index in range(n_samples):
        # The elementwise equality comparison returns an array of boolean values.
        # If any of them is `True`, the true label was in the top 3.
        if np.any(top_3[sample_index] == y[sample_index]):
            correct += 1
    return correct / n_samples