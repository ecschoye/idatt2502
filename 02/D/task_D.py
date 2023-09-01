import matplotlib.pyplot as plt
import torch
import torchvision


class SoftmaxModel:
    def __init__(self):
        self.W = torch.ones([784, 10], requires_grad=True)
        self.b = torch.ones([1, 10], requires_grad=True)

    def logits(self, x):
        return torch.matmul(x, self.W) + self.b

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


def main():
    model = SoftmaxModel()

    # Load observations from the MNIST dataset. The observations are divided into a training set and a test set.
    mnist_train = torchvision.datasets.MNIST("./data", train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input to a flattened vector of size 784
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor with one-hot encoding
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output with one-hot encoding

    mnist_test = torchvision.datasets.MNIST("./data", train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input to a flattened vector of size 784
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor with one-hot encoding
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output with one-hot encoding

    # Training loop
    optimizer = torch.optim.SGD([model.W, model.b], lr=0.01)  # Stochastic Gradient Descent optimizer with learning rate 0.01
    num_epochs = 1000

    for epoch in range(num_epochs):
        # Forward pass
        loss = model.loss(x_train, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # Print the final trained model's parameters, loss, and accuracy on the training set
    print("Final Parameters:")
    print(f"W: {model.W}")
    print(f"b: {model.b}")
    print(f"Loss: {model.loss(x_train, y_train).item()}")
    print(f"Accuracy: {model.accuracy(x_train, y_train).item()}")
    print()

    # test and show 10 random images from the test set with the model's prediction
    plt.figure(figsize=(10, 5))  # Set the figure size

    for i in range(10):
        index = torch.randint(0, x_test.shape[0], ())
        x = x_test[index]

        plt.subplot(2, 5, i + 1)  # Create a subplot in a 2x5 grid
        plt.imshow(x.reshape(28, 28), cmap="gray")
        plt.title(f"Model: {model.f(x).argmax().item()}\nActual: {y_test[index].argmax().item()}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


if __name__ == "__main__":
    main()
