import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim

class MyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', dtype=torch.float, device=None):
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size =hidden_size
        self.bias = bias
        self.nonlinearity = torch.tanh if nonlinearity =='tanh' else torch.relu
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device('cpu') 
        
        self.w_ih = torch.empty(input_size, hidden_size)
        self.w_hh = torch.empty(hidden_size, hidden_size)
        
        k=1/hidden_size
        lower_bound = -math.sqrt(k)
        upper_bound = math.sqrt(k)
        
        nn.init.uniform_(self.w_ih, a=lower_bound, b=upper_bound)
        nn.init.uniform_(self.w_hh, a=lower_bound, b=upper_bound)
        
        self.w_ih = self.w_ih.to(self.device)
        self.w_hh = self.w_hh.to(self.device)
        
        if self.bias:
            self.b_ih = torch.empty(hidden_size,)
            self.b_hh = torch.empty(hidden_size,)
            
            nn.init.uniform_(self.b_ih, a=lower_bound, b=upper_bound)
            nn.init.uniform_(self.b_hh, a=lower_bound, b=upper_bound)
            
            self.b_ih = self.b_ih.to(self.device)
            self.b_hh = self.b_hh.to(self.device)
            
            
    def forward(self, X, h=None):
        
        if h is None:
            h = torch.zeros(X.shape[0], self.hidden_size, device=self.device)
            
        if self.bias:
            return self.nonlinearity((X@self.w_ih +self.b_ih) + (h@self.w_hh + self.b_hh))
            
        return self.nonlinearity((X@self.w_ih) + (h@self.w_hh))




########## Do not change the code below #############


def report(result, i=1):
    if result is True:
        print(f"Task {i} passed.")
    else:
        print(f"Task {i} failed.")
        


##########Coding for testing ##############
def run_tests():
    i = 1
    batch_size = 20
    input_size = 30
    hidden_size = 15

    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    torch.manual_seed(10)  #fix seed for reproducibility

    model1 = MyRNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=device)
    model2 = MyRNNCell(input_size, hidden_size, bias=True, nonlinearity='relu', device=device)
    model3 = MyRNNCell(input_size, hidden_size, bias=False, nonlinearity='tanh', device=device)
    model4 = MyRNNCell(input_size, hidden_size, bias=False, nonlinearity='relu', device=device)
 
    X = torch.randn(batch_size, input_size).to(device)

    Z1 = model1(X)
    Z2 = model2(X)
    Z3 = model3(X)
    Z4 = model4(X)


    #print(Z1.to('cpu').shape)
    #print(torch.linalg.norm(Z1))
    report(torch.allclose(torch.linalg.norm(Z1), torch.tensor(10.0638), rtol=1e-04, atol=1e-04), i=i)
    i += 1
    
    #print(Z2.to('cpu').shape)
    #print(torch.linalg.norm(Z2))
    report(torch.allclose(torch.linalg.norm(Z2), torch.tensor(11.0243), rtol=1e-04, atol=1e-04), i=i)
    i += 1
    
    #print(Z3.to('cpu').shape)
    #print(torch.linalg.norm(Z3))
    report(torch.allclose(torch.linalg.norm(Z3), torch.tensor(9.9735), rtol=1e-04, atol=1e-04), i=i)
    i += 1
    
    #print(Z4.to('cpu').shape)
    #print(torch.linalg.norm(Z4))
    report(torch.allclose(torch.linalg.norm(Z4), torch.tensor(10.4574), rtol=1e-04, atol=1e-04), i=i)
    i += 1
    



# Use your own spare time to play with the training and testing of
# the above model you have coded, using the notebook I have shared. 

if __name__ == "__main__":
    run_tests()
    
