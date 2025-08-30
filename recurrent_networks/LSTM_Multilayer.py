##### CSCD485/585: Deep Learning  ########

### Total: 100 points

### Author: Bojian Xu, bojianxu@ewu.edu


# In this homework, you will implement your own multi-layer LSTM RNN class using the PyTorch's LSTEM Cell class. 
# See the lecture slides for the explanation of the LSTM RNN.

import torch
import torch.nn as nn
import math

from torch import optim
import torch.nn.functional as F



        
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype=torch.float):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
            input_size - The number of expected features in the input x
            hidden_size - The number of features in the hidden state h
            num_layers - Number of recurrent layers.
            bias - If False, then the layer does not use bias weights.

        Variables:
            lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer.
                                shape: (input_size, 4*hidden_size) for k=0;
                                shape: (hidden_size, 4*hidden_size), otherwise.
            lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer, of shape (hidden_size, 4*hidden_size).
            lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer, of shape (4*hidden_size,).
            lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer, of shape (4*hidden_size,).
        """

        ###### START YOUR CODE HERE #########
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        
        LSTMCell  = []
        self.LSTMCell = LSTMCell

        layer1 = nn.LSTMCell(self.input_size, self.hidden_size, bias=self.bias, device=self.device, dtype=self.dtype)
        LSTMCell.append(layer1)
        for i in range(num_layers - 1):
            layer = nn.LSTMCell(self.hidden_size, self.hidden_size, bias=self.bias, device=self.device, dtype=self.dtype)
            LSTMCell.append(layer)
        

        
        # We use nn.LSTMCell
                                       
        ###### END YOUR CODE HERE ###########

    def forward(self, X, hc=None):
        """
        Inputs: X, hc
            X:  of shape (seq_len, batch_size, input_size) containing the features of the input sequences.
            hc: tuple of (h0, c0) with
                h_0 of shape (num_layers, batch_size, hidden_size) containing the initial
                    hidden state for each element in the batch. Defaults to zeros if not provided.
                c0 of shape (num_layers, batch_size, hidden_size) containing the initial
                    hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
            output: of shape (seq_len, bs, hidden_size) containing the output features
                (h_t) from the last layer of the LSTM, for each t.
                i.e., `output` is the output states leaving the top layer over all time steps. 
            tuple of (h_n, c_n):
                h_n: of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
                c_n: of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
                i.e., `(h_n, c_n)` is the output states and cell states leaving all the layers at the end of processing the batch. 
        """

        ###### START YOUR CODE HERE #########
        #Extract dimensions
        seq_len, bs, input_size = X.shape
        
        if hc is None:
            hc = (torch.zeros(self.num_layers, bs, self.hidden_size).to(self.device), torch.zeros(self.num_layers, bs, self.hidden_size).to(self.device) )
            
        h0, c0 = hc
        
        
        
        #move to device
        X = X.to(self.device)


        #list containing each layer final hidden and cell state
        h_n = []
        c_n = []
        
        last_layer_hidden_states = []
        
        for i in range(self.num_layers):
            #Extract cell
            cell = self.LSTMCell[i]
            #Extract initial state
            layer_state = (h0[i], c0[i])
            #Saves the states of the current layer which is to be fed into the next layer
            curr_layer_hidden_states = []
            


            for j in range(seq_len):
                layer_state = cell(X[j], layer_state)
                curr_layer_hidden_states.append(layer_state[0])

                #Extract the last states of each layer
                if (j == seq_len - 1):
                    h_n.append(layer_state[0])
                    c_n.append(layer_state[1])
                    
            
                #Extract the top layer's hidden states
                if (i == self.num_layers - 1):
                    last_layer_hidden_states.append(layer_state[0])

            #next layer inputs
            X = curr_layer_hidden_states

        h_n = torch.stack(h_n, dim=0)
        c_n = torch.stack(c_n, dim=0)
        last_layer_hidden_states = torch.stack(last_layer_hidden_states, dim=0)
        

        return last_layer_hidden_states, (h_n, c_n)

        
        
        
            
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
    sequence_size = 100
    batch_size = 20
    num_layers = 5
    input_size = 30
    hidden_size = 15
    

    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    torch.manual_seed(10)  #fix seed for reproducibility

    model1 = MyLSTM(input_size, hidden_size, num_layers=num_layers, bias=True, device=device)
    model2 = MyLSTM(input_size, hidden_size, num_layers=num_layers, bias=False, device=device)
    model3 = MyLSTM(input_size, hidden_size, num_layers=num_layers, bias=True, device=device)
    model4 = MyLSTM(input_size, hidden_size, num_layers=num_layers, bias=False, device=device)
 
    X = torch.randn(sequence_size, batch_size, input_size).to(device)
    h = torch.randn(num_layers, batch_size, hidden_size).to(device)
    c = torch.randn(num_layers, batch_size, hidden_size).to(device)
    

    Z1 = model1(X)
    Z2 = model2(X)
    Z3 = model3(X, (h,c))
    Z4 = model4(X, (h,c))

    

    #print(Z1[0].to('cpu').shape)
    #print(Z1[1][0].to('cpu').shape)
    #print(Z1[1][1].to('cpu').shape)
    norm = torch.linalg.norm(Z1[0])+torch.linalg.norm(Z1[1][0])+torch.linalg.norm(Z1[1][1])
    #print(norm)
    report(torch.allclose(norm, torch.tensor(36.1313), rtol=1e-04, atol=1e-04), i=i)
    i+= 1

    #print(Z2[0].to('cpu').shape)
    #print(Z2[1][0].to('cpu').shape)
    #print(Z2[1][1].to('cpu').shape)
    norm = torch.linalg.norm(Z2[0])+torch.linalg.norm(Z2[1][0])+torch.linalg.norm(Z2[1][1])
    #print(norm)
    report(torch.allclose(norm, torch.tensor(9.0864), rtol=1e-04, atol=1e-04), i=i)
    i+= 1

    #print(Z3[0].to('cpu').shape)
    #print(Z3[1][0].to('cpu').shape)
    #print(Z3[1][1].to('cpu').shape)
    norm = torch.linalg.norm(Z3[0])+torch.linalg.norm(Z3[1][0])+torch.linalg.norm(Z3[1][1])
    #print(norm)
    report(torch.allclose(norm, torch.tensor(31.0516), rtol=1e-04, atol=1e-04), i=i)
    i+= 1

    #print(Z4[0].to('cpu').shape)
    #print(Z4[1][0].to('cpu').shape)
    #print(Z4[1][1].to('cpu').shape)
    norm = torch.linalg.norm(Z4[0])+torch.linalg.norm(Z4[1][0])+torch.linalg.norm(Z4[1][1])
    #print(norm)
    report(torch.allclose(norm, torch.tensor(14.9988), rtol=1e-04, atol=1e-04), i=i)
    i+= 1


if __name__ == "__main__":
    run_tests()


# Use your own spare time to play with the above model you have coded, using the notebook I have shared. 

    
