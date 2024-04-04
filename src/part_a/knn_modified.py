import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from utils import *
import pandas as pd
from utils import load_public_test_csv, load_train_sparse, load_valid_csv, \
    sparse_matrix_evaluate
from sklearn.neighbors import NearestNeighbors
import numpy as np

def sparse_matrix_evaluate(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc

def main():
    sparse_matrix = pd.read_csv('0.csv')
    val_data = pd.read_csv('1.csv')
    test_data = pd.read_csv('2.csv')
    # Assuming you have sparse_matrix, val_data, and train_data already loaded
    k_values = [1, 6, 11, 16, 21, 26]
    # Step 2: Calculate Distances

    knn = NearestNeighbors(n_neighbors=11, metric='euclidean')  # Initialize KNN with k neighbors and Euclidean distance
    knn.fit(sparse_matrix)  # Fit KNN model on the sparse matrix

    # Step 3: Find Closest K Users
    distances, indices = knn.kneighbors(val_data)  # Find k nearest neighbors for each user in val_data

    # Step 4: Predict Labels
    predicted_labels = []

    for i in range(len(val_data)):
        nearest_users = sparse_matrix.iloc[indices[i]]  # Get the K nearest users from train_data
        accuracies = nearest_users.mean(axis=0)  # Calculate the mean accuracy for each question
        predicted_label = (accuracies > 0.5).astype(int)  # Assign label 1 if accuracy > 0.5, otherwise 0
        predicted_labels.append(predicted_label)

    predicted_labels = np.array(predicted_labels)

    # Now, predicted_labels contains the predicted labels for each user in val_data

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)


    user_validation_accuracies = []

    # Compute the validation accuracy for each k
    for k in k_values:
        print(f"Testing k = {k}")
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_validation_accuracies.append(acc)

    # Select the best k based on validation accuracy
    best_k = k_values[np.argmax(user_validation_accuracies)]
    print(f"Best k for user-based k-NN: {best_k}")

    # Report the test accuracy with the chosen k*
    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print(
        f"User-based Best k-NN Test Accuracy with k={best_k}: {test_accuracy}")


if __name__ == '__main__':
    main()