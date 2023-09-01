import matplotlib.pyplot as plt
import torch


class NotModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
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
    model = NotModel()

    x_train = torch.tensor([[0.0], [1.0]])
    y_train = torch.tensor([[1.0], [0.0]])

    # Training loop
    optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)
    num_epochs = 200_000
    for epoch in range(num_epochs):
        # Forward pass
        loss = model.loss(x_train, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for every few epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # Test the model using test data
    x_test = torch.tensor([[0.0], [1.0]])
    y_test = torch.tensor([[1.0], [0.0]])
    accuracy = model.accuracy(x_test, y_test)
    print(f"Accuracy: {accuracy.item()}")

    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    plt.plot(x_train, y_train, "o", label="$(x^{(i)},y^{(i)})$")
    plt.xlabel("x")
    plt.ylabel("y")
    x = torch.arange(0.0, 1.0, 0.01).reshape(-1, 1)
    y = model.f(x).detach()
    plt.plot(x, y, linewidth=2, label=r"$\hat y = f(x) = \sigma(xW + b)$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
