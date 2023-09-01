import matplotlib.pyplot as plt
import pandas as pd
import torch

# Read the CSV file using pandas
df = pd.read_csv("day_length_weight.csv")

# Print the column names
print(df.columns)

# Extract the 'length', 'weight', and 'day' columns
x_train = df[["length", "weight"]].values
print(x_train)
y_train = df["day"].values.reshape(-1, 1)
print(y_train)

# Convert the data into PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
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

    num_epochs = 1_250_000

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

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Extract the length, weight, and day from x_train and y_train
    length = x_train[:, 0]
    weight = x_train[:, 1]
    day = y_train.flatten()

    # Scatter plot of length, weight, and day
    threshold = 500  # Adjust the threshold value as needed

    # Scatter plot with orange markers for day > threshold and blue markers for day <= threshold
    ax.scatter(
        length[day > threshold],
        weight[day > threshold],
        day[day > threshold],
        marker="o",
        color="orange",
    )
    ax.scatter(
        length[day <= threshold],
        weight[day <= threshold],
        day[day <= threshold],
        marker="o",
        color="blue",
    )

    ax.set_xlabel("Length")
    ax.set_ylabel("Weight")
    ax.set_zlabel("Day")

    plt.show()


if __name__ == "__main__":
    main()
