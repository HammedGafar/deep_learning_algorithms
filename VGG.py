##### CSCD485/585: Deep Learning  ########

### Total: 100 points

### Author: Bojian Xu, bojianxu@ewu.edu


#In this homework, you will implement the VGG block class and the VGG network class,
#which can be used to create a VGG model of various sizes. 
#See the lecture slides for the reference of the VGG network. 



import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, num_convs, in_channel, out_channels):
        '''
            num_convs: the number of conv layers
            in_channel: the channel count of the input volumn of the fist conv layer
            out_channel: the channel count of the volumn going through the entire vgg block,
                         ie, the channel count of the output volumn of the first conv layer,
                         as well as the channel count of input/output volumn of the follow up conv layers.
        '''
        
        super().__init__()

        ###### START YOUR CODE HERE #########
        
        # Every conv layer in/out channel count is given in the passed-in parameters. 
        # See the doc string for details

        # Every conv layer uses 3x3 kernel, stride = 1, and padding = 1
        # Every conv layer is followed by a relu layer
        
        # Following the final conv layer is a max pooling layer using kernel_size=2 and stride=2
        
        # Save the layers in `self.block`
        
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channels, kernel_size=3, padding= 1, stride = 1))
        layers.append(nn.ReLU())
        
        for i in range(num_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding= 1, stride = 1))
            layers.append(nn.ReLU())
            
        layers.append(nn.MaxPool2d(kernel_size=2, stride = 2))
        
        self.block = torch.nn.Sequential(*layers)
        

        
        ###### END YOUR CODE HERE ############
        
        
        
        
    def forward(self, X):
        ###### START YOUR CODE HERE #########

        #delete the following statement
        X = self.block(X)
        
        return X

        
        ###### END YOUR CODE HERE ############
        
    

class VGG(nn.Module):
    def __init__(self, arch, num_classes=10):
        
        # arch: a list of tuples that specifies each VGG block: (num_convs, in_channel, out_channels)
        #       in_channel: the channel count for the first conv of the block
        #       By changing this arch list, you can have different choices of archs. 
        #       For ex., VGG16 has 13 conv layers, plus with the later 3 linear layers, yielding a total of 16 layers. 
        #       arch=[(2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512)]

        super().__init__()

        
        ###### START YOUR CODE HERE #########

        # divide your code into `features` and `classifier`
        # The following describes the model's arch. 
        # See the lecture slides for a picture of a typical VGG model
        
        # the testing code will use:
        #     input image batch: N x 3 x 224 x 224
        #     class count: 10
        
        ## `self.features`: Create the sequence of VGG blocks that are specified in the passed-in `arch`. 
        ##                  Add a flatten layer in the end. 
        
        X = []
        
        for conv, in_channels, out_channels in arch:
            X.append(VGGBlock(conv, in_channels, out_channels))
            
        X.append(nn.Flatten())
        
        self.features = torch.nn.Sequential(*X)
        
        ## `self.classifier`:
        self.classifier = torch.nn.Sequential( 
        ### linear layer 1: 512*7*7 x 4096, followed by a relu and droppout with p=0.5
        nn.Linear(512*7*7, 4096),
        # relu
        nn.ReLU(),
        # dropout with p = 0.5
        nn.Dropout(p=0.5),
        ### linear layer 2: 4096 x 4096, followed by a relu and droppout with p=0.5
        nn.Linear(4096, 4096),
        # relu
        nn.ReLU(),
        # dropout with p = 0.5
        nn.Dropout(p=0.5),
        ### linear layer 3: 4096 x num_classes
        nn.Linear(4096, num_classes)
        )
        

        

        ###### END YOUR CODE HERE ############

        
        # I have tried that the following init is VERY important:
        # it becomes very hard for the model to pick up the training/learning. 
        # This for loop code is copied (with minor changes) from: https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L14/1.1-vgg16.ipynb
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

        
    def forward(self, X):

        X = self.features(X)
        logits = self.classifier(X)
        return logits


########## Do not change the code below #############


def report(result, i=1):
    if result is True:
        print(f"Task {i} passed.")
    else:
        print(f"Task {i} failed.")
        
i = 0 


##########Coding for testing ##############
def run_tests():
    i = 1 

    num_classes = 10
    num_samples = 20
    num_channels = 3
    width = 224
    height = 224


    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    ####****** Testing the vgg block class

    torch.manual_seed(10)  #fix seed for reproducibility
    X = torch.randn(num_samples, num_channels, height, width).to(device)
    model_vggblock = VGGBlock(5, num_channels, 20).to(device)
    Z = model_vggblock(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))

    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(82.3339,), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    ####****** testing the vgg class

    torch.manual_seed(10)  #fix seed for reproducibility
    arch=[(2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512)]  #VGG-16
    X = torch.randn(num_samples, num_channels, height, width).to(device)
    model_vgg = VGG(arch, num_classes=num_classes).to(device)
    Z = model_vgg(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))

    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(113.4693), rtol=1e-04, atol=1e-04)
           or torch.allclose(torch.linalg.norm(Z), torch.tensor(127.8114), rtol=1e-04, atol=1e-04), i=i) #handling apple's M chips and other chips.
    i += 1



    # In the above testing code, it does not check the result from the backward computation,
    # which will be correct if your code can pass the forward computation test, as long as
    # your code follows the code skeleton that is offered. 


    # Use your own spare time to play with the training and testing of the model 
    # that you have successfully created.
    # You can also try the notebook I shared for this purpose 
    # if you are too lazy to create your own but want to have some fun. 

if __name__ == "__main__":
    run_tests()
    
