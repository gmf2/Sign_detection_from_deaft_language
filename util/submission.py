import csv

def create_submission(p, path):
    """Create a submission file on the given path.
    
    :param p: An array of probabilities of shape `(n_samples,)` (model output on test set).
    :param path: The path to the output CSV file.
    """
    with open(path, 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['Id','Predicted'])
        for i in range(p.shape[0]):
            writer.writerow([i, p[i]])