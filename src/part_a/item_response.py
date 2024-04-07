from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for user_id, question_id, is_correct in zip(data['user_id'], data['question_id'], data['is_correct']):
        theta_i = theta[user_id]
        beta_j = beta[question_id]
        prob = sigmoid(theta_i - beta_j)
        log_lklihood += np.log(prob) * is_correct + np.log(1 - prob) * (1 - is_correct)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # grad_theta = np.zeros_like(theta)
    # grad_beta = np.zeros_like(beta)
    # for user_id, question_id, is_correct in zip(data['user_id'], data['question_id'], data['is_correct']):
    #     prob = sigmoid(theta[user_id] - beta[question_id])
    #     grad_theta[user_id] += is_correct - prob
    #     grad_beta[question_id] += -is_correct + prob
    # theta = theta + lr * grad_theta
    # beta = beta + lr * grad_beta
    grad_theta = np.zeros_like(theta)
    grad_beta = np.zeros_like(beta)

    for user_id, question_id, is_correct in zip(data['user_id'], data['question_id'], data['is_correct']):
        prob = sigmoid(theta[user_id] - beta[question_id])
        grad_theta[user_id] += (is_correct - prob)
    theta += lr * grad_theta
    grad_theta.fill(0)

    for user_id, question_id, is_correct in zip(data['user_id'], data['question_id'], data['is_correct']):
        prob = sigmoid(theta[user_id] - beta[question_id])
        grad_beta[question_id] += (-is_correct + prob)
    beta += lr * grad_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # 541 students 1773 questions
    theta = np.zeros(541 + 1)
    beta = np.zeros(1773 + 1)

    val_acc_lst = []

    train_lld_lst = []
    valid_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)

        train_lld_lst.append(-neg_lld)
        valid_lld_lst.append(-neg_log_likelihood(val_data, theta=theta, beta=beta))

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)



    plt.figure(figsize=(10, 6))
    plt.plot(train_lld_lst, label='Trailing Log Likelihood')

    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(valid_lld_lst, label='Validation Log Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


import matplotlib.pyplot as plt


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, val_acc_list = irt(train_data, val_data, 0.005, 100)

    print("Final validation accuracy: {}\nFinal test accuracy: {}".format(evaluate(data=val_data, theta=theta, beta=beta),evaluate(data=test_data, theta=theta, beta=beta)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j1 = 1525
    j2 = 1574
    j3 = 1030
    beta_j1 = beta[j1]
    beta_j2 = beta[j2]
    beta_j3 = beta[j3]
    print("the difficulty of j1 j2 j3 are", beta_j1, beta_j2, beta_j3)
    theta_range = np.linspace(-3, 3, 100)

    prob_j1 = sigmoid(theta_range - beta_j1)
    prob_j2 = sigmoid(theta_range - beta_j2)
    prob_j3 = sigmoid(theta_range - beta_j3)

    plt.plot(theta_range, prob_j1, label='Question $j_1$')
    plt.plot(theta_range, prob_j2, label='Question $j_2$')
    plt.plot(theta_range, prob_j3, label='Question $j_3$')

    plt.legend()

    plt.xlabel('Ability $\\theta$')
    plt.ylabel('Probability of Correct Response $p(c_{ij} = 1)$')
    plt.title('Probability of Correct Response vs. Ability')

    plt.grid(True)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
