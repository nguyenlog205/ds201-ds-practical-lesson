import torch
import torch.nn as nn
import torch.nn.functional as F

class T1(nn.Module):
    def __init__(self):
        super(T1, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        
        return x