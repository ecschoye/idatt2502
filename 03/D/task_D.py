import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28,
                                   28).float().to(
    device)  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10)).to(device)  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(
    device)  # torch.functional.nn.conv2d argument must include channels (1)
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
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def logits(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc_layers(x)

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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch].to(device),
                   y_train_batches[batch].to(device)).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    scheduler.step()

    print(f"accuracy = {model.accuracy(x_test.to(device), y_test.to(device)).item() * 100:.2f}%")

print("done")



clothing_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

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
    plt.suptitle("Task D", fontsize=16)
    plt.show()
    model.train()


plot_test_images(x_test[0:25], y_test[0:25], model)

# Final accuracy: 91.24%

# -------------------------------------------------------------------------------------------
# Alternative way of plotting the test images
# Let's the user input an index to check the prediction
# -------------------------------------------------------------------------------------------

#def plot_single_test_image(x_test, y_test, model, index):
#    plt.figure()
#    plt.imshow(x_test[index].cpu().reshape(28, 28), cmap=plt.cm.binary)
#    predicted_label = model.f(x_test[index:index + 1]).argmax(1).item()
#    true_label = y_test[index].argmax().item()
#    plt.xlabel(f"Predicted: {clothing_names[predicted_label]}, True: {clothing_names[true_label]}")
#    plt.title("Prediction Result")
#    plt.show()
#
#
## Asking the user for an index input
#while True:
#    try:
#        user_input = input("Enter an index in the test dataset range to check the prediction (or 'exit' to quit): ")
#
#        if user_input.lower() == 'exit':
#            break
#
#        user_index = int(user_input)
#        if 0 <= user_index < len(x_test):
#            plot_single_test_image(x_test, y_test, model, user_index)
#        else:
#            print("Index out of range. Please try again.")
#
#    except ValueError:
#        print("Invalid input. Please enter an integer.")