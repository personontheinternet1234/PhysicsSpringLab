import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    model = Model()

    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)
    y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]], requires_grad=False)

    epochs = 100
    for epoch in range(epochs):
        model.train()

        predictions = model(x_train)
        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    final_weight = model.linear.weight.item()
    final_bias = model.linear.bias.item()

    print("\nTraining completed.")
    print(f"Final weight: {final_weight:.4f}")
    print(f"Final bias: {final_bias:.4f}")