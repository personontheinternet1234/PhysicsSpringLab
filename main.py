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


def train(model, x, y, epochs=1000):
    print("\n")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    epochs = epochs
    loss = 0
    for epoch in range(epochs):
        model.train()

        predictions = model(x)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_weight = model.linear.weight.item()

    print(f"Loss: {loss.item()}")
    print(f"Final weight: {final_weight:.4f}")


if __name__ == "__main__":
    model_small_spring = Model()

    data = pd.read_csv('data.csv')

    x_data = torch.tensor(data['Distance (cm)'].values, dtype=torch.float32).view(-1, 1)
    y_data = torch.tensor(data['Force (Small) (N)'].values, dtype=torch.float32).view(-1, 1)

    train(model_small_spring, x_data, y_data)

    x_data = torch.tensor(data['Distance (cm)'].values, dtype=torch.float32).view(-1, 1)
    y_data = torch.tensor(data['Force (Big) (N)'].values, dtype=torch.float32).view(-1, 1)

    model_spring_big = Model()

    train(model_spring_big, x_data, y_data)
