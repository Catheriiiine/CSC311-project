import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def load_data(file_name, base_path="../data"):
    file_path = f"{base_path}/{file_name}"
    data = pd.read_csv(file_path)

    X = data.drop(columns=['is_correct'])  # Features
    y = data['is_correct']  # Target

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader

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
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for features, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = torch.sum((predictions - labels) ** 2.)
            reg_loss = lamb * model.get_weight_norm() / 2
            total_loss = loss + reg_loss
            total_loss.backward()

            train_loss += total_loss.item()

            optimizer.step()

        # avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in valid_loader:
                predictions = model(features)
                predicted = predictions.round()  # Assuming binary classification
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        valid_acc = correct / total

        print(f'Epoch {epoch}: Training Loss = {train_loss:.4f}, Validation Accuracy = {valid_acc:.4f}')

def main():
    train_loader = load_data('new_train_data.csv')
    valid_loader = load_data('new_valid_data.csv')

    num_features = next(iter(train_loader))[0].shape[1]
    model = AutoEncoder(num_features=num_features, k=50)

    train(model, train_loader, valid_loader, num_epoch=10, lr=0.05)

if __name__ == "__main__":
    main()
