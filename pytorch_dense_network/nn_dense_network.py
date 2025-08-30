##### CSCD485/585: Deep Learning  ########

### homework 3 
### Total: 100 points

### Author: Bojian Xu, bojianxu@ewu.edu


#In this homework, we will practice to use PyTorch's nn module 
#and its predefined layers to create a general purpose dense 
#network model class that users can use in various settings.
#The class can serve for classfication as well as regression. 

#In your own spare time, you can try to use this model to train on various data
#sets using various architecture/activations, and see how well it works. 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim



class MLP(nn.Module):
    def __init__(self, arch, acts, loss_fn, num_classes=None, bias=True):
        '''
            arch: a list of tuples. 
                  Each tuple is of two integers, representing the dimension sizes of the matrix of a linear layer.
            acts: the list of activation functions to be used by every layer. 
                  Note: If a layer does not need an activation function, use torch.Identity, which does not do anything, as its activation. 
            loss_fn: loss function to be used. 
                  Note: For classification problem, softmax loss is often used. 
                        For regression problem, mean square error is often used. 
            num_classes: the number of classes. If None, the model is used for regression.
            bias: Whether to use a bias vector at every layer.

            Note: You cannot hardcode your model by using the arch given in the testing code. 
                  Your code should be able to handle any arch that is given to the `__init__`. 
                  We will use a diff arch for grading. 
        '''
        
        super().__init__()
        
        self.num_layers = len(arch)
        self.num_classes = num_classes
        self.num_feature = arch[0][0]
        
        assert len(arch) == len(acts)

        if num_classes is not None: # for classification
            assert arch[-1][1] == num_classes
            assert num_classes > 1
        else:                       # for regression
            assert arch[-1][1] == 1
            
        if self.num_layers > 1: 
            for i in range(self.num_layers-1):
                assert arch[i][1] == arch[i+1][0]

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.loss_fn = loss_fn
                
        ###### START YOUR CODE HERE #########
        network = []
        for net, act in zip(arch, acts):
            layer = nn.Linear(*net, bias = bias)
            network.append(layer)
            network.append(act)
            
        self.model = nn.Sequential(*network).to(self.device)
            
        
        #Save your model in `self.model`
        
        
        
        
        ###### END YOUR CODE HERE ############

    
    def forward(self, X):
        '''
            X: n x m matrix. 
               n: sample count
               m: feature count, which equals arch[0][0]
            
            return: 1) for classifiction: the raw logits n x k matrix produced by the model. 
                       n: sample count
                       k: num_classes
                    2) for regression: n x 1 matrix (vector).
                       n: sample count
        '''
        ###### START YOUR CODE HERE #########

        return self.model(X)

        
        ###### END YOUR CODE HERE #########
    
    
    def fit(self, X, y, opt, epochs=1):
        '''
            X: n x m matrix. 
               n: sample count
               m: feature count, which equals arch[0][0]
            y: n x 1 matrix (vector)
               n: sample count
            opt: the optimizer to be used
            epochs: epoch count used to train on this given (X, y) batch
            
            return: the `loss` and `error` of the most recent epoch.
                    For classification, the error is the misclassfied sample percentage. 
                    For regression, the error is the mean absolute difference. 
                    (code for this part is provided already.)

            Note: We use the entire given (X,y) as one mini batch.
            
        '''

        X = X.to(self.device)
        y = y.to(self.device)

        ###### START YOUR CODE HERE #########
        self.model.train()
        
        X = X.reshape(-1, self.num_feature)
        
        for i in range(epochs):
            Z = self.model(X)
            loss = self.loss_fn(Z, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
        loss = loss.to("cpu")

        ###### END YOUR CODE HERE #########

        with torch.no_grad():
            if self.num_classes is not None:
                error = (Z.argmax(dim=1) != y).sum().to(torch.float)
                error = (error / X.shape[0]).to("cpu")
            else:
                #Mean Absolute Error (MAE), reduction=mean
                error = F.l1_loss(Z, y)

        return loss, error

    
    def predict(self, X):
        '''
            X: n x m matrix. 
               n: sample count
               m: feature count, which equals arch[0][0]
               
            return: n x 1 matrix (vector), 
                    where every element is the predicted class id (in classification), 
                    or the predicated function value (in regression) of the sample. 
        '''

        X = X.to(self.device)
        
        with torch.no_grad():        
            ###### START YOUR CODE HERE #########
            self.model.eval()
            
            X = X.reshape(-1, self.num_feature)
            Z = self.model(X)
            
            if self.num_classes is not None:
                y = Z.argmax(dim = 1).to("cpu")
            else:
                y = Z.to("cpu")
                
            return y
        
            ###### END YOUR CODE HERE #########

    def error(self, X, y):
        '''
            X: n x m matrix. 
               n: sample count
               m: feature count, which equals arch[0][0]
            y: n x 1 matrix (vector)
               n: sample count
            
            return: a tensor of one element.
                    In classification: the percentage of misclassified samaples
                    In regression: the average relative error in the prediction of the values of the function of the samples. 
        '''
        
        with torch.no_grad():
            predictions = self.predict(X)
            if self.num_classes is not None:
                error = (predictions != y).sum().to("cpu")
                error = error /  X.shape[0]
            else: 
                error = F.l1_loss(predictions, y)
                
        return error


        
########## Do not change the code below #############


########## code for testing #############

def report(result, i=1):
    if result is True:
        print(f"Task {i} passed.")
    else:
        print(f"Task {i} failed.")
        
i = 0 
        
########## code for testing: classification #############
num_classes = 10
num_samples = 20
num_features = 784

arch = [(num_features, 256), (256, 128), (128, num_classes)]
acts = [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Identity()]
loss_fn = F.cross_entropy

epochs = 1
lr = 0.05

torch.manual_seed(0)

model = MLP(arch, acts, loss_fn, num_classes)
opt = optim.SGD(model.parameters(), lr=lr)

X = torch.randn(num_samples, num_features)
y = torch.randint(0, num_classes, (num_samples,))

loss, error = model.fit(X, y, opt, epochs=epochs)
#model.fit(X, y, opt, lr=lr, epochs=epochs)
#print(f"loss = {loss}, error = {error}")

#print(loss)
i += 1
report(torch.allclose(loss, torch.tensor(2.2829), rtol=1e-04, atol=1e-04), i=i)

#print(error)
i += 1
report(torch.allclose(error, torch.tensor(1.0), rtol=1e-04, atol=1e-04), i=i)


with torch.no_grad():
    a = torch.tensor(0.0)
    for p in model.parameters():
        a += torch.linalg.norm(p)
    #print(a)
    i += 1
    report(torch.allclose(a, torch.tensor(18.5633), rtol=1e-04, atol=1e-04), i=i)


with torch.no_grad():
    logits = model(X)
    #print(f"logits = {logits}")
    #print(torch.linalg.norm(logits))
    i += 1
    report(torch.allclose(torch.linalg.norm(logits), torch.tensor(1.6901), rtol=1e-04, atol=1e-04), i=i)

with torch.no_grad():
    y_hat = model.predict(X).to(torch.float)
    #print(torch.linalg.norm(y_hat))
    #print(f"y_hat = {y_hat}")
    i += 1
    report(torch.allclose(torch.linalg.norm(y_hat), torch.tensor(22.3607), rtol=1e-04, atol=1e-04), i=i)

with torch.no_grad():
    error = model.error(X, y)
    #print(f"error = {error}")

    


########## code for testing: regression #############
num_samples = 20
num_features = 784

arch = [(num_features, 256), (256, 128), (128, 1)]
acts = [torch.nn.Tanh(), torch.nn.ReLU(), torch.nn.Sigmoid()]
loss_fn = F.mse_loss

epochs = 1
lr = 0.05

torch.manual_seed(0)

model = MLP(arch, acts, loss_fn)
opt = optim.SGD(model.parameters(), lr=lr)

X = torch.randn(num_samples, num_features)
y = torch.randn(num_samples, 1)

loss, error = model.fit(X, y, opt, epochs=epochs)
#model.fit(X, y, opt, lr=lr, epochs=epochs)
#print(f"loss = {loss}, error = {error}")

#model.fit(X, y, opt, lr=lr, epochs=epochs)
#print(f"loss = {loss}, error = {error}")

#print(loss)
i += 1
report(torch.allclose(loss, torch.tensor(1.4860), rtol=1e-04, atol=1e-04), i=i)

#print(error)
i += 1
report(torch.allclose(error, torch.tensor(0.9223), rtol=1e-04, atol=1e-04), i=i)



with torch.no_grad():
    a = torch.tensor(0.0)
    for p in model.parameters():
        a += torch.linalg.norm(p)
    #print(a)
    i += 1
    report(torch.allclose(a, torch.tensor(17.2280), rtol=1e-04, atol=1e-04), i=i)


with torch.no_grad():
    y_hat = model.predict(X).to(torch.float)
    #print(torch.linalg.norm(y_hat))
    #print(f"y_hat = {y_hat}")
    i += 1
    report(torch.allclose(torch.linalg.norm(y_hat), torch.tensor(2.0514), rtol=1e-04, atol=1e-04), i=i)

with torch.no_grad():
    error = model.error(X, y)
    #print(f"error = {error}")
 
    

    
    
exit()


# Note: The testing cases provided above, which use a fixed random seed,
# are assuming to run on the CPU, although the code will run on GPU if
# you have one. If your code fails in the testings on GPU but passes on
# CPU, it will be fine. (CPU and GPU may produce different random number
# sequences even being given the same initial random seed.



######### code for playing with the model for MNIST, in your spare time, if you want to ################

# Packages needed for using the torch's MINIST dataset
# This notebooks requirens: `torchvision 0.15.0` or newer version
import torchvision
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


num_classes = 10
num_samples = 20
num_features = 784

arch = [(num_features, 256), (256, 128), (128, num_classes)]
acts = [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Identity()]
loss_fn = F.cross_entropy

epochs = 1
lr = 0.05

torch.manual_seed(0)

model = MLP(arch, acts, loss_fn, num_classes)
opt = optim.SGD(model.parameters(), lr=lr)

train_data = MNIST(root='./MNIST', train=True, download=True, transform=ToTensor()) 
test_data = MNIST(root='./MNIST', train=False, download=True, transform=ToTensor()) 
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10000, shuffle=True) # use the whole test data set of testing

print(train_data.data.shape)
print(train_data.targets.shape)
print(test_data.data.shape)
print(test_data.targets.shape)

for i in range(100):
    for j, batch in enumerate(train_loader):
        X, y = batch
        X = X.reshape(-1, 28*28)
        loss, error = model.fit(X, y, opt, epochs=epochs)
        print(f"loss = {loss}, error = {error}")





        
