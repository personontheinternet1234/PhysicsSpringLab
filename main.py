import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    model_small = Model()

    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.SGD(model_small.parameters(), lr=0.001)

    data = pd.read_csv('data.csv')

    x_data = torch.tensor(data['Distance (cm)'].values, dtype=torch.float32).view(-1, 1)
    y_data = torch.tensor(data['Force (Small) (N)'].values, dtype=torch.float32).view(-1, 1)

    epochs = 1000
    loss = 0
    for epoch in range(epochs):
        model_small.train()

        predictions = model_small(x_data)
        loss = criterion(predictions, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_weight = model_small.linear.weight.item()

    print(f"{loss.item()}")
    print(f"Final weight: {final_weight:.4f}")

    print("##########################################################")

    model_big = Model()

    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.SGD(model_big.parameters(), lr=0.001)

    data = pd.read_csv('data.csv')

    x_data = torch.tensor(data['Distance (cm)'].values, dtype=torch.float32).view(-1, 1)
    y_data = torch.tensor(data['Force (Big) (N)'].values, dtype=torch.float32).view(-1, 1)

    epochs = 1000
    for epoch in range(epochs):
        model_big.train()

        predictions = model_big(x_data)
        loss = criterion(predictions, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_weight = model_big.linear.weight.item()

    print(f"{loss.item()}")
    print(f"Final weight: {final_weight:.4f}")
