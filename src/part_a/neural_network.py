from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = F.sigmoid(self.h(F.sigmoid(self.g(inputs))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


# def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
#     """ Train the neural network, where the objective also includes
#     a regularizer.
#
#     :param model: Module
#     :param lr: float
#     :param lamb: float
#     :param train_data: 2D FloatTensor
#     :param zero_train_data: 2D FloatTensor
#     :param valid_data: Dict
#     :param num_epoch: int
#     :return: None
#     """
#     # TODO: Add a regularizer to the cost function.
#
#     # Tell PyTorch you are training the model.
#     model.train()
#
#     # Define optimizers and loss function.
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#     num_student = train_data.shape[0]
#
#     train_losses = []
#     train_accuracies = []
#     valid_accuracies = []
#
#     for epoch in range(0, num_epoch):
#         train_loss = 0.
#         correct = 0
#         total = 0
#
#         for user_id in range(num_student):
#             inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
#             target = inputs.clone()
#
#             optimizer.zero_grad()
#             output = model(inputs)
#
#
#             # Mask the target to only compute the gradient of valid entries.
#             # nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
#             nan_mask = np.isnan(train_data[user_id].numpy())
#             target[0][nan_mask] = output[0][nan_mask]
#             # print(train_data)
#             # Training accuracy
#             guess = output[0][~nan_mask].data.numpy() >= 0.5
#             correct += np.sum(guess == train_data[user_id][~nan_mask].numpy())
#             total += np.sum(~nan_mask)
#
#             loss = torch.sum((output - target) ** 2.)
#             loss.backward()
#
#             train_loss += loss.item()
#             optimizer.step()
#
#
#         valid_acc = evaluate(model, zero_train_data, valid_data)
#         print("Epoch: {} \tTraining Cost: {:.6f}\t "
#               "Valid Acc: {}".format(epoch, train_loss, valid_acc))
#         # test_acc = evaluate(model, zero_train_data, test_data)
#         # print(test_acc)
#
#         train_accuracies.append(correct / float(total))
#         valid_accuracies.append(valid_acc)
#
#     return train_accuracies, valid_accuracies
#     ####################################################################
#     # END OF YOUR CODE                            #
#     ####################################################################

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    valid_accuracies = []


    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)

            # Regularization term: λ/2 * (||W1||^2 + ||W2||^2)
            # Get the weight norm from the model's method and add it to the loss
            reg_loss = lamb * model.get_weight_norm() / 2

            # The total loss is the sum of the prediction loss and the regularization loss
            total_loss = loss + reg_loss

            total_loss.backward()

            train_loss += total_loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        train_losses.append(train_loss)
        valid_accuracies.append(valid_acc)

    return train_losses, valid_accuracies

#     #####################################################################
#     #                       END OF YOUR CODE                            #
#     #####################################################################

def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # :                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    training_accuracies = {}
    validation_accuracies = {}
    test_accuracies = {}
    max = 0
    max_k = 0
    max_lr = 0
    max_num = 0
    test_acc = 0

    # for num_epoch in [5, 11, 15, 20]:
    #     #print("For num_epoch = {}".format(num_epoch))
    #     for lr in [0.01, 0.05, 0.1]:
    #         #print("For lr = {}".format(lr))
    #         for k in [10, 50, 100, 200, 500]:
    k = 50
    lr = 0.05
    num_epoch = 11
    max_lamb = 0
    for lamb in (0.001, 0.01, 0.1, 1):
        model = AutoEncoder(train_matrix.shape[1], k)

        # Set optimization hyperparameters.
        print("For k = {}".format(k))
        train_accs, valid_accs = train(model, lr, lamb, train_matrix, zero_train_matrix,
                                       valid_data, num_epoch)

        training_accuracies[k] = train_accs
        validation_accuracies[k] = valid_accs
        test_accuracies[k] = evaluate(model, zero_train_matrix, test_data)
        if validation_accuracies[k][-1] > max:
            max = validation_accuracies[k][-1]
            max_lamb = lamb
            # max_k = k
            # max_lr = lr
            # max_num = num_epoch
            test_acc = test_accuracies[k]
    print("Maximam validation accuracy is: {}".format(max))
    print("When test accuracy is: {}".format(test_acc))
    print("When lamb is: {}".format(max_lamb))
    # print("When k is: {}".format(max_k))
    # print("When lr is: {}".format(max_lr))
    # print("When number of epoch is: {}".format(max_num))


    # # Plotting
    # epochs = list(range(1, num_epoch + 1))
    # fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    #
    # # Plot training cost
    # for k, costs in training_accuracies.items():
    #     axes[0].plot(epochs, costs, marker='o', label=f'k={k}')
    # axes[0].set_title('Training Accuracy per Epoch')
    # axes[0].set_ylabel('Training Accuracy')
    # axes[0].legend()
    #
    # # Plot validation accuracy
    # for k, accuracies in validation_accuracies.items():
    #     axes[1].plot(epochs, accuracies, marker='o', label=f'k={k}')
    # axes[1].set_title('Validation Accuracy per Epoch')
    # axes[1].set_xlabel('Epoch')
    # axes[1].set_ylabel('Validation Accuracy')
    # axes[1].legend()
    #
    # plt.tight_layout()
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
