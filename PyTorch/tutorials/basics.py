"""
 * https://pytorch.org/tutorials/beginner/basics/intro.html

We’ll use the FashionMNIST dataset to train a neural network that predicts if an input image belongs to one of the following classes:
  - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot.
"""

# PyTorch has two primitives to work with data:
# - torch.utils.data.DataLoader: wraps an iterable around the Dataset
# -  and torch.utils.data.Dataset: Dataset stores the samples and their corresponding labels

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


####
# Training data and Test data
####

# PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets.
# For this tutorial, we will be using a TorchVision dataset.

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Every TorchVision Dataset includes two arguments: transform and target_transform to modify the samples and labels respectively.

# Here we define a batch size of 64,
# i.e. each element in the dataloader iterable will return a batch of 64 features and labels
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# debug
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
#
# >Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
# >Shape of y: torch.Size([64]) torch.int64

####
# Creating Models
# * https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#creating-models
####

# Get cpu, gpu or mps device for training.

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
#   We define the layers of the network in the __init__ function
#   and specify how data will pass through the network in the forward function.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

####
# Optimizing the Model Parameters
# * https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#optimizing-the-model-parameters
####

# To train a model, we need a loss function and an optimizer.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# In a single training loop, the model makes predictions on the training dataset (fed to it in batches),
# and backpropagates the prediction error to adjust the model’s parameters.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# We also check the model’s performance against the test dataset to ensure it is learning.

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# The training process is conducted over several iterations (epochs).
# During each epoch, the model learns parameters to make better predictions.
# We print the model’s accuracy and loss at each epoch;
# we’d like to see the accuracy increase and the loss decrease with every epoch.

epochs = 5

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

####
# Saving Models
# * https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#saving-models
####

# A common way to save a model is to serialize the internal state dictionary (containing the model parameters).

model_path = "./data/model.pth"

torch.save(model.state_dict(), model_path)
print("Saved PyTorch Model State to {}".format(model_path))

####
# Loading Models
# * https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#loading-models
####

# The process for loading a model includes re-creating the model structure and loading the state dictionary into it.

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(model_path))

# This model can now be used to make predictions.

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]

with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

