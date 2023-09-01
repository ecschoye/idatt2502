import matplotlib.pyplot as plt
import torch


class NANDModel:
    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return torch.matmul(x, self.W) + self.b

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)
        # Similar to:
        # return -torch.mean(y * torch.log(self.f(x)) +
        #                    (1 - y) * torch.log(1 - self.f(x)))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


def main():
    model = NANDModel()

    x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]])

    # Training loop
    optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)
    num_epochs = 100_000
    for epoch in range(num_epochs):
        # Forward pass
        loss = model.loss(x_train, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for every few epochs
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # Test the model using test data
    x_test = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_test = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
    accuracy = model.accuracy(x_test, y_test)
    print(f"Accuracy: {accuracy.item()}")

    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    # Create a 3D scatter plot
    fig = plt.figure("NAND Operator")
    ax = fig.add_subplot(111, projection="3d")

    # Plot input points
    ax.scatter(x_test[:, 0], x_test[:, 1], y_test[:, 0], c="b", marker="o", label="True Output")
    ax.scatter(x_test[:, 0], x_test[:, 1], model.f(x_test).detach(), c="r", marker="x", label="Predicted Output")

    # Plot sigmoid curve
    input_vals = torch.linspace(0, 1, 100)
    input_grid = torch.meshgrid(input_vals, input_vals)
    input_grid_flat = torch.stack([input_grid[0].flatten(), input_grid[1].flatten()], dim=1)
    sigmoid_vals = model.f(input_grid_flat).detach().reshape(input_grid[0].shape)

    ax.plot_surface(input_grid[0], input_grid[1], sigmoid_vals, cmap="viridis", alpha=0.5)

    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_zlabel("Output")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
