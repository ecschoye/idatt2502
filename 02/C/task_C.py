import matplotlib.pyplot as plt
import torch
import numpy as np


class XORModel:
    def __init__(self):
        self.W1 = torch.rand((2, 2), requires_grad=True)
        self.b1 = torch.rand((1, 2), requires_grad=True)
        self.W2 = torch.rand((2, 1), requires_grad=True)
        self.b2 = torch.rand((1, 1), requires_grad=True)

    def f(self, x):
        return self.f2(self.f1(x))

    def f1(self, x):
        return torch.sigmoid(torch.matmul(x, self.W1) + self.b1)

    def f2(self, h):
        return torch.sigmoid(torch.matmul(h, self.W2) + self.b2)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


def main():
    model = XORModel()

    x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32, requires_grad=True)

    # Training loop
    optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], lr=1)

    num_epochs = 10_000
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
    y_test = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    accuracy = model.accuracy(x_test, y_test)
    print(f"Accuracy: {accuracy.item()}")

    print("W1 = %s, b = %s, loss = %s" % (model.W1, model.b1, model.loss(x_train, y_train)))

    print("W2 = %s, b = %s" % (model.W2, model.b2))

    # Create a 3D scatter plot
    fig = plt.figure("XOR Operator")
    ax = fig.add_subplot(111, projection="3d")

    # Set the view angle
    ax.view_init(elev=30, azim=120)

    # Set ticks and tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Add grid
    ax.grid(False)

    # Add labeled points
    ax.scatter(x_train[:, 0].squeeze().detach().numpy(),
                x_train[:, 1].squeeze().detach().numpy(),
                y_train[:, 0].squeeze().detach().numpy(),
                c="blue",
                marker="o",
                s=30)

    # Plot the surface
    x1_grid, x2_grid = np.meshgrid(np.linspace(0, 1, 10),
                                    np.linspace(0, 1, 10))
    y_grid = np.empty([10, 10])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            input_data = torch.tensor([[x1_grid[i, j], x2_grid[i, j]]], dtype=torch.float32)
            y_grid[i, j] = model.f(input_data).detach().numpy()

    ax.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")

    # Show the plot
    plt.show()




if __name__ == "__main__":
    main()
