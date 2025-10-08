import torch
import torch.nn as nn
import torch.nn.functional as F

class T2(nn.Module):
    def __init__(self):
        """
        Initializes the layers for the T2 neural network.
        """
        super(T2, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        """
        Defines the forward pass of the T2 neural network.
        x is the input tensor (batch of images).
        """
        x = x.view(x.size(0), -1)
        

        x = self.fc1(x) 
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x