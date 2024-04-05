import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from utils import *
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import load_public_test_csv, load_train_sparse, load_valid_csv, \
    sparse_matrix_evaluate
from sklearn.neighbors import NearestNeighbors
import numpy as np
# Load the CSV files

def dist_all(v, X):
    """
    Compute the squared Euclidean distance between a user `v` (vector) and the
    user in the data matrix `X`.

    """

    diff = X - v
    sqdiff = np.square(diff)
    sumval = np.sum(sqdiff, axis = 1)
    return sumval

def predict_knn(v, X_train, t_train, k=30):
    """
    Returns a prediction using the k-NN

    Parameters:
        `v` - a numpy array (vector) representing an MNIST image, shape (784,)
        `X_train` - a data matrix representing a set of MNIST image, shape (N, 784)
        `t_train` - a vector of ground-truth labels, shape (N,)
        `k` - a positive integer 1 < k <= N, describing the number of closest images
              to consider as part of the knn algorithm

    Returns:
        A single number `i` between 0 and 9, representing the digit
    """
    # Step 1. compute the distances between v and every element of X
    dists = dist_all(v, X_train)

    # Step 2. find the indices of the k-nearest neighbours

    # Hint: You may wish to sort the distances in `dists`. But how should you
    # do this sorting while keeping track of the indices? You may find
    # the functions "enumerate" (or "zip"), and "sorted" helpful.
    # Alternatively, you may choose to use a function like "np.argsort"
    sorted_lst = sorted(enumerate(dists), key=lambda x: x[1])[:k]

    indices = [index for (index, value) in sorted_lst]

    # Step 3. find the most common target label amongst these indices

    ts = t_train[np.array(indices)]
    not_all_nan_columns = ~np.isnan(ts).all(axis=0)
    means = np.full(ts.shape[1], np.nan)
    valid_columns_means = np.nanmean(ts[:, not_all_nan_columns], axis=0)
    means[not_all_nan_columns] = valid_columns_means

    means = np.nan_to_num(means, nan=0.5)
    most_common_values = np.round(means).astype(int)
    prediction = most_common_values
    return prediction

def compute_accuracy(X_new, t_new, X_train, t_train, k=1):
    """
    Returns the accuracy (proportion of correct predictions) on the data set
    `X_new` and ground truth `t_new`.

    Parameters:
        `X_new` - a data matrix representing MNIST images that we would like to
                  make predictions for, shape (N', 784)
        `t_new` - a data matrix representing ground truth labels for images in X_new,
                  shape (N',)
        `X_train` - a data matrix representing a set of MNIST image in the training set,
                    shape (N, 784)
        `t_train` - a vector of ground-truth labels for images in X_train,
                    shape (N,)
        `k` - a positive integer 1 < k <= N, describing the number of closest images
              to consider as part of the knn algorithm

    Returns: the proportion of correct predictions (between 0 and 1)
    """

    num_predictions = 0
    num_correct = 0

    for i in range(X_new.shape[0]): # iterate over each image index in X_new
        v = X_new[i] # image vector
        t = t_new[i] # prediction target
        y = predict_knn(v, X_train, t_train, k=k)
        for j in range(len(y)):
            if y[j] != 0 and y[j] != 1:
                continue
            if t[j] != 0 and t[j] != 1:
                continue
            num_predictions += 1
            if y[j] == t[j]:
                num_correct += 1

    return num_correct / num_predictions

if __name__ == '__main__':
    train_data_path = '0.csv'
    train_target_path = 'question_correctness.csv'
    validation_data_path = '1.csv'
    valid_target_path = 'question_correctness1.csv'
    test_data_path = '2.csv'
    test_target_path = 'question_correctness2.csv'

    train_data = pd.read_csv(train_data_path)
    train_target_data = pd.read_csv(train_target_path)

    valid_data = pd.read_csv(validation_data_path)
    valid_target_data = pd.read_csv(valid_target_path)

    test_data = pd.read_csv(test_data_path)
    test_target_data = pd.read_csv(test_target_path)

    X_train = train_data.sort_values(by='user_id').drop(columns=['user_id']).values
    t_train = train_target_data.sort_values(by='user_id').drop(columns=['user_id']).values

    X_valid = valid_data.sort_values(by='user_id').drop(columns=['user_id']).values
    t_valid = valid_target_data.sort_values(by='user_id').drop(columns=['user_id']).values

    X_test = test_data.sort_values(by='user_id').drop(columns=['user_id']).values
    t_test = test_target_data.sort_values(by='user_id').drop(columns=['user_id']).values

    print('Train accuracy at k = 1 (should be 1): ')
    print(compute_accuracy(X_train, t_train, X_train=X_train, t_train=t_train, k=1))
    print('Validation accuracy at k = 250: ')
    print(compute_accuracy(X_valid, t_valid, X_train=X_train, t_train=t_train, k=250))
    print('Test accuracy at k = 250: ')
    print(compute_accuracy(X_test, t_test, X_train=X_train, t_train=t_train, k=250))



