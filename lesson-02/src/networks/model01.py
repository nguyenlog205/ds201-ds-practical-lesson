import torch
import torch.nn as nn
import torch.nn.functional as F

class model01(nn.Module):
    """
    LeNet-5 architecture (1998) adapted for grayscale 28x28 input (e.g., MNIST).
    Layers:
        Conv1: 1 -> 6, kernel 5x5, padding=2
        AvgPool1: 2x2, stride=2
        Conv2: 6 -> 16, kernel 5x5
        AvgPool2: 2x2, stride=2
        FC1: 16*5*5 -> 120
        FC2: 120 -> 84
        FC3: 84 -> 10
    """

    def __init__(self, num_classes=10):
        super(model01, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Convolutional layers with activation + pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
