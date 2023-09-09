import torch
import torch.nn as nn
import torchvision

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28,
                                   28).float().to(device)  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10)).to(device)  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(device)  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10)).to(device)  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

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
        self.first_bn = nn.BatchNorm2d(32)  # Batch normalization
        self.first_maxpool_layer = nn.MaxPool2d(kernel_size=2)  # First Max-pooling
        self.first_dropout = nn.Dropout(0.25)  # Dropout 1

        # Second convolution
        self.second_conv_layer = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.second_bn = nn.BatchNorm2d(64)  # Batch normalization
        self.second_maxpool_layer = nn.MaxPool2d(kernel_size=2)  # Second Max-pooling
        self.second_dropout = nn.Dropout(0.25)  # Dropout 2

        # Fully connected layer
        self.fully_connected_dense_layer = nn.Linear(64 * 7 * 7, 1024)
        self.third_bn = nn.BatchNorm2d(1024)  # Batch normalization
        self.relu_layer = nn.ReLU()
        self.third_dropout = nn.Dropout(0.5)  # Dropout 3

        # Second fully connected layer
        self.second_fc_dense_layer = nn.Linear(1024, 10)

        # May not need this?
        # self.second_relu_layer = nn.ReLU()

    def logits(self, x):
        x = self.first_conv_layer(x)
        x = self.first_maxpool_layer(x)
        x = self.first_dropout(x)

        x = self.second_conv_layer(x)
        x = self.second_maxpool_layer(x)
        x = self.second_dropout(x)

        x = self.fully_connected_dense_layer(x.reshape(-1, 64 * 7 * 7))
        x = self.relu_layer(x)
        x = self.third_dropout(x)

        x = self.second_fc_dense_layer(x.reshape(-1, 1024))

        return x

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

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)

for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch].to(device), y_train_batches[batch].to(device)).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print(f"accuracy = {model.accuracy(x_test.to(device), y_test.to(device)).item() * 100:.2f}%")

print("done")

# Not impressive results from ReLU alone, 50-80%

# Adding Dropout increased accuracy to 97-98%

# Adding Batch Normalization did not do much, staying in the same range 97-98%
# pushing 99%
