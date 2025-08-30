import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class AlexNet(nn.Module):
    def __init__(self, class_num = 10):

      super().__init__()
        
      self.features = torch.nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )
        
      self.classifier = torch.nn.Sequential(
            nn.Linear(5*5*256, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, class_num)
        )
        
    def forward(self, X):
            X = self.features(X)
            print(X.shape)
            logits = self.classifier(X)
            return logits 
