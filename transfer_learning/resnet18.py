##### CSCD485/585: Deep Learning  ########

### Total: 50 points

### Author: Bojian Xu, bojianxu@ewu.edu


#In this homework, you will practice the transfer learning by using the pretrained ResNet-18.


import torch
import torch.nn as nn
import torchvision
from torch import optim
import torch.nn.functional as F



num_classes = 20

model = torchvision.models.resnet18(weights='IMAGENET1K_V1') #weights='DEFAULT'

torch.manual_seed(10)  #fix seed for reproducibility

###### START YOUR CODE HERE #########

# delete the following statement
# raise NotImplementedError


# replace the output layer by a linear layer that recognize `num_classes` classes
model.fc = torch.nn.Linear(512, 20, bias=True)

# Freeze all the layers except the final avgpool and fc layers.
for param in model.parameters():
    param.requires_grad = False

for param in model.avgpool.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

###### END YOUR CODE HERE ###########

    
    
    
########## Do not change the code below #############


def report(result, i=1):
    if result is True:
        print(f"Task {i} passed.")
    else:
        print(f"Task {i} failed.")
        


##########Coding for testing ##############
def run_tests(model, num_classes):
    num_samples = 20
    num_channels = 3
    width = 224
    height = 224

    i = 1
    
    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")  # use the fixed cpu device for grading purpose. 


    model = model.to(device)

    lr = 0.005 #learning rate
    opt = optim.SGD(model.parameters(), lr=lr)
    

    torch.manual_seed(10)  #fix seed for reproducibility

    X = torch.randn(num_samples, num_channels, height, width).to(device)
    y = torch.randint(num_classes, (num_samples,))
    Z = model(X)
    loss = F.cross_entropy(Z, y)
    loss.backward()
    opt.step()

    X = torch.randn(num_samples, num_channels, height, width).to(device)    
    Z = model(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(10.1904), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    X = torch.randn(num_samples, num_channels, height, width).to(device)
    Z = model(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(9.8985), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    X = torch.randn(num_samples, num_channels, height, width).to(device)
    Z = model(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(10.1344), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    X = torch.randn(num_samples, num_channels, height, width).to(device)
    Z = model(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(10.4419), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    X = torch.randn(num_samples, num_channels, height, width).to(device)
    Z = model(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(9.7872), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    
    



# Use your own spare time to play with the training and testing of
# the above model you have coded, using the notebook I have shared. 

if __name__ == "__main__":
    run_tests(model, num_classes)
    
