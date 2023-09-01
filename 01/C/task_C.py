import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "day_head_circumference.csv")

# Read the CSV file using pandas
df = pd.read_csv(csv_path)

# Extract the 'day' and 'head circumference' columns
x_train = df["# day"].values.reshape(-1, 1)
y_train = df["head circumference"].values.reshape(-1, 1)

# Convert the data into PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)


class NonlinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20 * torch.sigmoid(torch.matmul(x, self.W) + self.b) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


def main():
    model = NonlinearRegressionModel()

    # Optimize: adjust W and b to minimize loss using stochastic gradient descent
    optimizer = torch.optim.SGD([model.W, model.b], 0.000001)

    num_epochs = 200_000

    for epoch in range(num_epochs):
        loss = model.loss(x_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for every few epochs
        if (epoch + 1) % 10000 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # Print model variables and loss
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    # Visualize result
    plt.figure(figsize=(8, 6))
    plt.title("Predict head circumference based on age (in days)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x_train, y_train, marker="o")
    x = torch.arange(torch.min(x_train), torch.max(x_train), 1.0).reshape(-1, 1)
    y = model.f(x).detach()
    plt.plot(
        x,
        y,
        color="orange",
        linewidth=2,
        label=r"$f(x) = 20\sigma(xW + b) + 31$\n$\sigma(z) = \dfrac{1}{1+e^{-z}}$",
    )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
