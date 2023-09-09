import torch
import torch.nn as nn
import torchvision

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float().to(device)
y_train = torch.zeros((mnist_train.targets.shape[0], 10)).to(device)
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(device)
y_test = torch.zeros((mnist_test.targets.shape[0], 10)).to(device)
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        # First convolution
        self.first_conv_layer = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        # First Max-pooling
        self.first_maxpool_layer = nn.MaxPool2d(kernel_size=2)

        # Second convolution
        self.second_conv_layer = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # Second Max-pooling
        self.second_maxpool_layer = nn.MaxPool2d(kernel_size=2)

        # Fully connected dense layer
        self.fc_dense_layer = nn.Linear(64 * 7 * 7, 10)

    def logits(self, x):
        x = self.first_conv_layer(x)
        x = self.first_maxpool_layer(x)

        x = self.second_conv_layer(x)
        x = self.second_maxpool_layer(x)

        return self.fc_dense_layer(x.reshape(-1, 64 * 7 * 7))

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel().to(device)

accuracy_list = []

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch].to(device), y_train_batches[batch].to(device)).backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"accuracy = {model.accuracy(x_test.to(device), y_test.to(device)).item() * 100:.2f}%")
    accuracy_list.append(model.accuracy(x_test.to(device), y_test.to(device)))

average_accuracy = sum(accuracy_list) / len(accuracy_list)
print("average accuracy: {:.2f}%".format(average_accuracy * 100))
print("done")
