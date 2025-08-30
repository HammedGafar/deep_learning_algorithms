# First practice to get familar with PyTorch programming via a dense network training. 
# The goal of this hw is to get familar with the forward and backward computation in the PyTorch framework. 
# The actual/practical way to create and train ML/DL models using PyTorch will be very different, which we will discuss in future lectures. 

# Hammed Gafar, EWU_ID = 1008912

import torch
import torch.nn.functional as F



def nn_train(X, Y, W_list, b_list, act_list, loss_fn=F.mse_loss, lr=0.01, epochs = 1):
    
    '''
        X: input batch of samples, n x m
            n: #samples
            m: #features.
        Y: input labels: n x k
            n: #samples
            k: #classes
        W_list: the list of the W matrices of all the layers. 
                The list size the number of layers (excluding the input layer).
        b_list: the list of the bias vectors of all the layers. 
                The list size the number of layers (excluding the input layer).
        act_list: the list of activation functions of all the layers. 
                The list size the number of layers (excluding the input layer).
        loss: the loss function being used
        lr: learning rate being used
        epochs: the number of epochs being used

        Note: 1) In every epoch, we use the entire X and Y. We don't do mini batches in this hw, for simplicity. 
              2) PyTorch creates the CG from scratch for every forward-backward pass.
              3) Updates to the model parameters must happen in-place. 
    '''
    
    num_layers = len(W_list)
    

    
    for i in range(epochs): 
        #YOUR CODE STARTS HERE
        X_Out = X
        for i in range(num_layers):
            X_1 = X_Out@W_list[i] + b_list[i]
            X_Out = act_list[i](X_1)
        
        mse = loss_fn(X_Out, Y)
        mse.backward()
        
    with torch.no_grad():
        for i in range(num_layers):
            W_list[i] -= lr * W_list[i].grad
            b_list[i] -= lr * b_list[i].grad
            W_list[i].grad.zero_()
            b_list[i].grad.zero_()
                
        #Delete the following `raise` statement. Do not change any of the code outside this `for` loop. 

        
        #YOUR CODE ENDS HERE
    
    
    return W_list, b_list



torch.manual_seed(0)


num_classes = 3

X = torch.randn((4,5))
Y = F.one_hot(torch.arange(0, 4) % 3, num_classes=num_classes).to(torch.float)

#print(f"X: {X}")
#print(f"Y: {Y}")

W1 = torch.randn((5,4), requires_grad = True)
b1 = torch.randn(4, requires_grad = True)
W2 = torch.randn((4,4), requires_grad = True)
b2 = torch.randn(4, requires_grad = True)
W3 = torch.randn((4, num_classes), requires_grad = True)
b3 = torch.randn(num_classes, requires_grad = True)

W_list = [W1, W2, W3]
b_list = [b1, b2, b3]
act_list = [F.tanh, F.relu, F.sigmoid]


loss_fn = F.mse_loss
lr = 0.01
epochs = 5

Ws, bs = nn_train(X, Y, W_list, b_list, act_list, loss_fn=loss_fn, lr=lr, epochs = epochs)

#print(f"Ws: {Ws}")
#print(f"bs: {bs}")


def report(result, i=1):
    if result is True:
        print(f"Layer {i} passed.")
    else:
        print(f"Layer {i} failed.")
        
report(torch.allclose(torch.linalg.norm(torch.vstack((b1, W1))), torch.tensor(3.6786), rtol=1e-04, atol=1e-04), i=1)
report(torch.allclose(torch.linalg.norm(torch.vstack((b2, W2))), torch.tensor(4.4156), rtol=1e-04, atol=1e-04), i=2)
report(torch.allclose(torch.linalg.norm(torch.vstack((b3, W3))), torch.tensor(5.5048), rtol=1e-04, atol=1e-04), i=3)







