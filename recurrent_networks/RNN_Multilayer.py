import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import Tensor

class MyRNN(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, num_layers=1, bias=False, nonlinearity=None, device=None, dtype=torch.float16):
        
        super().__init__()
        
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype=dtype
        
        self.rnn_cells = []
        
        for _ in range(num_layers):
            layer = nn.RNNCell(self.input_size, self.hidden_size, bias=self.bias, nonlinearity=self.nonlinearity, device=self.device, dtype=self.dtype)
            self.rnn_cells.append(layer)
        
    def forward(self, X, h0=None):
        
        #Extract dimensions
        seq_len, bs, input = X.shape
        
        #Move to device
        X = X.to(self.device)
        
        #List containing the final state from each layer
        layer_final_states = []
        
        #Saves the states of the current layer which is to be fed into the next layer
        curr_layer_states = []
        
        if h0 is None:
            h0 = torch.zeros((self.num_layers, bs, self.hidden_size), device=self.dtype)
            
        h0 = h0.to(self.device)
        
        for i in range(self.num_layers):
            #Extract cell
            cell = self.rnn_cells[i]
            #Extract initial states
            layer_state = h0[i]
            
            #Saves the states of the current layer which is to be fed into the next layer
            curr_layer_states = []
            
            for j in range(seq_len):
                layer_state = cell(X[j], layer_state)
                curr_layer_states.append(layer_state)
                
                #Extract the last state of each layer
                if (j == seq_len-1):
                    layer_final_states.append(layer_state)
                    
            #next layer inputs
            X = curr_layer_states
            
        output = torch.stack(curr_layer_states, dim=0)
        h_n = torch.stack(layer_final_states, dim=-0)
        
        return output, h_n