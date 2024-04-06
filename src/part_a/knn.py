import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from utils import *

from utils import load_public_test_csv, load_train_sparse, load_valid_csv, \
    sparse_matrix_evaluate


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


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Part A 1(b)
    # Transpose the matrix so that the questions become the features
    # Transpose the matrix so that questions become the features
    matrix_T = matrix.T

    # Create a KNN imputer instance with the specified number of neighbors
    nbrs = KNNImputer(n_neighbors=k)

    # Impute missing values based on question similarity
    imputed_matrix_T = nbrs.fit_transform(matrix_T)

    # Transpose back the imputed matrix to its original user-question format
    imputed_matrix = imputed_matrix_T.T

    # Initialize a counter for correct predictions
    correct_predictions = 0

    # Iterate over the validation data to predict and evaluate accuracy
    for user_id, question_id, is_correct in zip(valid_data['user_id'],
                                                valid_data['question_id'],
                                                valid_data['is_correct']):
        # Predicted correctness is the value in the imputed matrix at the user's
        # and question's position
        predicted_correctness = round(imputed_matrix[user_id, question_id])

        # Increment the counter if the prediction matches the actual correctness
        if predicted_correctness == is_correct:
            correct_predictions += 1

    # Calculate the accuracy as the ratio of correct predictions to total
    # predictions
    acc = correct_predictions / len(valid_data['user_id'])
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #

    # Part A 1(a)
    # Define k values to test
    k_values = [1, 6, 11, 16, 21, 26, 250]
    user_validation_accuracies = []

    # Compute the validation accuracy for each k
    for k in k_values:
        print(f"Testing k = {k}")
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_validation_accuracies.append(acc)

    # Plotting the validation accuracies
    plt.plot(k_values, user_validation_accuracies, marker='o')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy of different k for user-based k-NN')
    plt.show()

    # Select the best k based on validation accuracy
    best_k = k_values[np.argmax(user_validation_accuracies)]
    print(f"Best k for user-based k-NN: {best_k}")

    # Report the test accuracy with the chosen k*
    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print(
        f"User-based Best k-NN Test Accuracy with k={best_k}: {test_accuracy}")

    #
    # Part A 1(c):
    item_validation_accuracies = []

    for k in k_values:
        print(f"Testing k = {k}")
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        item_validation_accuracies.append(acc)

    # Plotting the validation accuracies
    plt.plot(k_values, item_validation_accuracies, marker='o')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy of different k for item-based k-NN')
    plt.show()

    # Select the best k based on validation accuracy
    best_k = k_values[np.argmax(item_validation_accuracies)]
    print(f"Best k for item-based k-NN: {best_k}")

    # Report the test accuracy with the chosen k*
    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print(
        f"Item-based Best k-NN Test Accuracy with k={best_k}: {test_accuracy}")


    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
    # Part A 1(b)
    print("Part A: 1(b): The underlying assumption on item-based collaborative"
          " filtering: the if question A is answered correctly (or incorrectly)"
          " by users in the same manner as question B, then the correctness of"
          " answers to question A can predict the correctness of answers to"
          " question B.")

    # Part A 1(d)
    print("Part A: 1(d): User-based collaborative filtering is better, "
          "since Test Accuracy of the best user-based k-NN with k = 11 is "
          "0.6841659610499576, which is bigger than the Test Accuracy of best "
          "item-based k-NN with k=26: 0.6528365791701948.")

    # Part A 1(e):
    # first limitation
    print("The initial limitation involves scalability issues. "
          "As the dataset expands, k-NN's effectiveness declines. Within the "
          "educational data realm, where there might be an enormous count of "
          "both students (users) and questions (items), kNN could prove to be "
          "unfeasible because it requires calculating the distance between "
          "every point pair for prediction. This problem is especially "
          "critical in real-time systems where quick feedback is essential")
    # second limitation
    print("The second major limitation is computational efficiency."
          "The requirement for the algorithm to retain the full training "
          "dataset for making predictions, coupled with the intensive "
          "computational demand of determining distances between a query "
          "point and every other point, especially when the question count "
          "is high, poses a significant challenge. Sparse data, characterized "
          "by a majority of missing entries, intensifies this issue, leading "
          "to the need for extra measures to handle the sparsity effectively.")
