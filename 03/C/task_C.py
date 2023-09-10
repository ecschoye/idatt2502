import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

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

        # First convolution
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        # Second convolution
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        # Fully connected dense layer
        self.layer3 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Second fully connected dense layer
        self.layer4 = nn.Linear(1024, 10)

    def logits(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x.reshape(-1, 64 * 7 * 7))
        x = self.layer4(x.reshape(-1, 1024))
        
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
        model.loss(x_train_batches[batch].to(device),
                   y_train_batches[batch].to(device)).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print(f"accuracy = {model.accuracy(x_test.to(device), y_test.to(device)).item() * 100:.2f}%")

print("done")


# Not impressive results from ReLU alone, 50-80%

# Adding Dropout increased accuracy to 97-98%

# Adding Batch Normalization did not do much, staying in the same range 97-98%
# pushing 99%

def plot_test_images(x_test, y_test, model):
    model.eval()
    plt.figure(figsize=(10, 10))

    random_indices = torch.randint(0, len(x_test), (25,))  # Generate 25 random indices

    for i, idx in enumerate(random_indices):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[idx].cpu().reshape(28, 28), cmap=plt.cm.binary)
        predicted_label = model.f(x_test[idx:idx + 1]).argmax(1).item()
        true_label = y_test[idx].argmax().item()
        plt.xlabel(f"Predicted: {predicted_label}, True: {true_label}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Task C", fontsize=16)
    plt.show()
    model.train()


plot_test_images(x_test[0:25], y_test[0:25], model)

# Final accuracy: 99.13%