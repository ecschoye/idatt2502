import matplotlib.pyplot as plt
import pandas as pd
import torch

# Read the CSV file using pandas
df = pd.read_csv("length_weight.csv")

# Extract the 'length' and 'weight' columns
x_train = df["length"].values.reshape(-1, 1)
y_train = df["weight"].values.reshape(-1, 1)

# Convert the data into PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.matmul(x, self.W) + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


def main():
    model = LinearRegressionModel()

    # Optimize: adjust W and b to minimize loss using stochastic gradient descent
    optimizer = torch.optim.SGD([model.W, model.b], 0.0001)

    num_epochs = 500_000

    for epoch in range(num_epochs):
        loss = model.loss(x_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for every few epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # Print model variables and loss
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    # Visualize result
    plt.plot(x_train, y_train, "o", label="$(x^{(i)},y^{(i)})$")
    plt.title("Predict weight based on length")
    plt.xlabel("x - Length")
    plt.ylabel("y - Weight")
    x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
    plt.plot(x, model.f(x).detach(), label="$\\hat y = f(x) = xW+b$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
