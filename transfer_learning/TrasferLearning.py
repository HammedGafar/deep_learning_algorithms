# Simple code illustrating transfer learning using the pretrained ResNet-18

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import Tensor
from torch import optim

num_classes = 10  #assuming 10 classes

model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

model.fc = nn.Linear(512, num_classes, bias=True)

#freeze all layers except the final avgpool layer and fc layers
for param in model.parameters():
    param.requires_grad=False
    
for param_avgpool in model.avgpool.parameters():
    param_avgpool=True
    
for param_fclayer in model.fc.parameters():
    param_fclayer.requires_grad=True