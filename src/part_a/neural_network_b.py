import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *

def load_data(base_path="../data"):
    # file_path = f"{base_path}/{file_name}"
    # data = pd.read_csv(file_path)
    #
    # X = data.drop(columns=['is_correct'])  # Features
    # y = data['is_correct']  # Target
    #
    # X_tensor = torch.tensor(X.values, dtype=torch.float32)
    # y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    #
    # dataset = TensorDataset(X_tensor, y_tensor)
    # loader = DataLoader(dataset, batch_size=32, shuffle=True)
    #
    # return loader
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
    def __init__(self, num_features, k=100):
        super(AutoEncoder, self).__init__()
        self.g = nn.Linear(num_features, k)
        self.h = nn.Linear(k, num_features)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        encoded = F.sigmoid(self.g(inputs))
        decoded = self.h(encoded)
        return F.sigmoid(decoded)

def train(model, train_loader, valid_loader, num_epoch, lr=0.05, lamb=0.001):
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # model.train()
    #
    # for epoch in range(0, num_epoch):
    #     train_loss = 0.
    #
    #     for features, labels in train_loader:
    #         optimizer.zero_grad()
    #         predictions = model(features)
    #         loss = torch.sum((predictions - labels) ** 2.)
    #         reg_loss = lamb * model.get_weight_norm() / 2
    #         total_loss = loss + reg_loss
    #         total_loss.backward()
    #
    #         train_loss += total_loss.item()
    #
    #         optimizer.step()
    #
    #     # avg_train_loss = total_loss / len(train_loader)
    #
    #     # Validation phase
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for features, labels in valid_loader:
    #             predictions = model(features)
    #             predicted = predictions.round()  # Assuming binary classification
    #             correct += (predicted == labels).sum().item()
    #             total += labels.size(0)
    #
    #     valid_acc = correct / total
    #
    #     print(f'Epoch {epoch}: Training Loss = {train_loss:.4f}, Validation Accuracy = {valid_acc:.4f}')
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_data = train_loader.shape[0]

    train_losses = []
    valid_accs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for i in range(num_data):
            inputs = Variable(train_loader[i]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            # nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            nan_mask = np.isnan(train_loader[i].numpy())
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


        valid_acc = evaluate(model, train_loader, valid_loader)
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)


        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        # test_acc = evaluate(model, zero_train_data, valid_data)
        # print(test_acc)

        train_losses.append(train_loss)
        valid_accs.append(valid_acc)

    return train_losses, valid_accs

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

    for i, u in enumerate(valid_data):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)
def main():
    train_loader = load_data('new_train_data.csv')
    valid_loader = load_data('new_valid_data.csv')

    num_features = next(iter(train_loader))[0].shape[1]
    model = AutoEncoder(num_features=num_features, k=50)

    train(model, train_loader, valid_loader, num_epoch=10, lr=0.05)

if __name__ == "__main__":
    main()
