import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # First conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second conv layer
        self.fc1 = nn.Linear(64 * 28 * 28, 256)  # Adjust input size accordingly
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class CNN128(nn.Module):
    def __init__(self, num_classes):
        super(CNN128, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Adding pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)  # Adjusted for pooling layers
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # Pooling
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # Pooling
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x