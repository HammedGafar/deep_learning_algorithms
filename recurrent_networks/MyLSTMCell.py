##### CSCD485/585: Deep Learning  ########


### Total: 100 points

### Author: Bojian Xu, bojianxu@ewu.edu


# In this homework, you will implement your own LSTM Cell class.
# See the lecture slides for the explanation of the LSTM Cell.

import torch
import torch.nn as nn
import math

from torch import optim
import torch.nn.functional as F

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=torch.float):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        
        ** Note: Weights implemented here have a transposed shape of the Weights implemented in PyTorch's torch.nn.LSTMCell
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        ###### START YOUR CODE HERE #########
        
        self.W_ih = torch.empty(self.input_size, 4*self.hidden_size)
        self.W_hh = torch.empty(self.hidden_size, 4*self.hidden_size)
        
        k=1/hidden_size
        lower_bound = -math.sqrt(k)
        upper_bound = math.sqrt(k)
        
        nn.init.uniform_(self.W_ih, a=lower_bound, b=upper_bound)
        nn.init.uniform_(self.W_hh, a=lower_bound, b=upper_bound)
        
        if self.bias:
            self.bias_ih = torch.empty(4*self.hidden_size,)
            self.bias_hh = torch.empty(4*self.hidden_size,)
            
            nn.init.uniform_(self.bias_ih, a=lower_bound, b=upper_bound)
            nn.init.uniform_(self.bias_hh, a=lower_bound, b=upper_bound)

        # Move weights and biases to the specified device
        self.W_ih = self.W_ih.to(self.device)
        self.W_hh = self.W_hh.to(self.device)
        if self.bias:
            self.bias_ih = self.bias_ih.to(self.device)
            self.bias_hh = self.bias_hh.to(self.device)
        
        

        ###### END YOUR CODE HERE ###########


    def forward(self, X, hc=None):
        """
        Inputs: X, hc
            X:  of shape (batch, input_size): Tensor containing input features
            hc: tuple of (h0, c0), with
                h0 of shape (batch_size, hidden_size): Tensor containing the initial hidden state
                    for each element in the batch. Defaults to zero if not provided.
                c0 of shape (batch_size, hidden_size): Tensor containing the initial cell state
                    for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
            h' of shape (bs, hidden_size): Tensor containing the next hidden state for each element in the batch.
            c' of shape (bs, hidden_size): Tensor containing the next cell state for each element in the batch.
        """

        ###### START YOUR CODE HERE #########
        
        
        if hc is None:
            hc = (torch.zeros(X.shape[0], self.hidden_size).to(self.device), torch.zeros(X.shape[0], self.hidden_size).to(self.device) )
            
        h0, c0 = hc

        if self.bias:
            gates = X@self.W_ih + self.bias_ih + h0@self.W_hh + self.bias_hh
        else:
            gates = X@self.W_ih + h0@self.W_hh
            
        #gates = gates.to(self.device)
        
        i = torch.sigmoid(gates[:, :self.hidden_size])
        f = torch.sigmoid(gates[:, self.hidden_size:2*self.hidden_size])
        g = torch.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])
        o = torch.sigmoid(gates[:, 3*self.hidden_size:4*self.hidden_size])
        
        next_c = f*c0 + i*g 
        next_h = o*torch.tanh(next_c)
        
        return (next_c, next_h)
        

        ###### END YOUR CODE HERE ###########
    
    
    
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

    model1 = MyLSTMCell(input_size, hidden_size, bias=True, device=device)
    model2 = MyLSTMCell(input_size, hidden_size, bias=True, device=device)
    model3 = MyLSTMCell(input_size, hidden_size, bias=False, device=device)
    model4 = MyLSTMCell(input_size, hidden_size, bias=False, device=device)
 
    X = torch.randn(batch_size, input_size).to(device)
    h = torch.randn(batch_size, hidden_size).to(device)
    c = torch.randn(batch_size, hidden_size).to(device)
        
    Z1 = model1(X)
    Z2 = model2(X, (h,c))
    Z3 = model3(X)
    Z4 = model4(X, (h,c))


    #print(Z1[0].to('cpu').shape)
    #print(Z1[1].to('cpu').shape)
    #print(torch.linalg.norm(Z1[0])+torch.linalg.norm(Z1[1]))
    report(torch.allclose(torch.linalg.norm(Z1[0])+torch.linalg.norm(Z1[1]), torch.tensor(7.6052), rtol=1e-04, atol=1e-04), i=i)
    i += 1
    
    #print(Z2[0].to('cpu').shape)
    #print(Z2[1].to('cpu').shape)
    #print(torch.linalg.norm(Z2[0])+torch.linalg.norm(Z2[1]))
    report(torch.allclose(torch.linalg.norm(Z2[0])+torch.linalg.norm(Z2[1]), torch.tensor(15.1879), rtol=1e-04, atol=1e-04), i=i)
    i += 1
    
    #print(Z3[0].to('cpu').shape)
    #print(Z3[1].to('cpu').shape)
    #print(torch.linalg.norm(Z3[0])+torch.linalg.norm(Z3[1]))
    report(torch.allclose(torch.linalg.norm(Z3[0])+torch.linalg.norm(Z3[1]), torch.tensor(7.8024), rtol=1e-04, atol=1e-04), i=i)
    i += 1
    
    #print(Z4[0].to('cpu').shape)
    #print(Z4[1].to('cpu').shape)
    #print(torch.linalg.norm(Z4[0])+torch.linalg.norm(Z4[1]))
    report(torch.allclose(torch.linalg.norm(Z4[0])+torch.linalg.norm(Z4[1]), torch.tensor(14.7379), rtol=1e-04, atol=1e-04), i=i)
    i += 1
    
    


if __name__ == "__main__":
    run_tests()
    
