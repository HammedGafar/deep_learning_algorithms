##### CSCD485/585: Deep Learning  ########

### Total: 100 points

### Author: Bojian Xu, bojianxu@ewu.edu


#In this homework, you will implement the class for the ResNet model,
#which requires the construction of the residual block and the residual block group classes.
#See the lecture slides for the reference of the ResNet network. 



import torch
import torch.nn as nn
import torch.nn.functional as F



# Within a block: 
#       image H W are not to change, except for the first block of the 2nd, 3rd, ..., groups.
#       the 2nd conv layer does not change channel size
#       Channel count must match for the final + operation. The side 1x1 conv layers is used to make that happen when needed. 

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channels, use_1x1conv=False, stride=1):
        '''
            in_channel: the channel count of the input tensor, 
                        i.e., the input channel count of the fist conv layer of the block
            out_channels: the output channel count of all conv layers, 
                          as well as the input channel size of the 2nd conv layer.  
            use_1x1conv: if true, use a 1x1 conv layer, using the passed-in 
                         `in_channel`, `out_channels`, and `stride`
        '''        
        super().__init__()
        
        
        ###### START YOUR CODE HERE #########
        
        # conv 2d layer: in_channel, out_channels, 3x3 kernal, padding 1, stride=stride
        # batch norma 2d layer
        # relu layer
        # conv 2d layer: out_channels, out_channels, 3x3 kernal, padding 1, stride 1
        # batch norm 2d layer
        # save the above sequentially in `self.block`
        
        # if use_1x1conv is true: add a conv 2d layer: in_channel, out_channels, 1x1 kernal, stride=stride
        #                         save it as `self.conv_1x1`
        # otherwise, set `self.conv_1x1` as None. 
        

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        
        
        if use_1x1conv == True:
            self.conv_1x1 = nn.Conv2d(in_channel, out_channels, kernel_size=1,stride=stride)
        
        else:
            self.conv_1x1 = None
            
            
        
            
            
            
            

        ###### END YOUR CODE HERE ############
        
        
        

        
    def forward(self, X):
        if self.conv_1x1 is None:
            return F.relu(self.block(X) + X)
        else:
            return F.relu(self.block(X) + self.conv_1x1(X))

        
        
# within a group, over its blocks
#      the output channel count does not change
#      the image H W do not change except for the first block of the 2nd, 3rd, etc. group.

class ResBlockGroup(nn.Module):
    def __init__(self, num_resblocks, in_channel, out_channels, first_group=False):
        '''
            num_resblocks: number of res blocks
            in_channel: the in_channel count of the first block
            out_channels: the out_channel count of the 1st, 2nd, 3rd, ..., block, 
                          as well as the in_channel count of the 2nd, 3rd, ..., block
            first_group: true if this is the first res block group in the entire model.
        '''
        super().__init__()
        
        ###### START YOUR CODE HERE #########

        
        
        # create a sequence of res blocks, save them in `self.group`
        ## block 0: use in_channel, out_channels, and the side 1x1 conv layer.
        ##          use stride 1 if it is in the first group; use stride 2 in the 2nd, 3rd, ... group. 
        ## block 1,2,...,num_resblocks-1: use out_channels, out_channels, stride 1.
        ##                                do not use 1x1 side conv layer.
        
        
    
        #delete the following statement

        blocks = []
        if first_group == True:
            blocks.append(ResBlock(in_channel, out_channels, use_1x1conv=True, stride=1))
            
            
            for i in range(num_resblocks - 1):
                    blocks.append(ResBlock(out_channels, out_channels))
                    
                    
        else:
            blocks.append(ResBlock(in_channel, out_channels, use_1x1conv=True, stride=2))
            
            
            for i in range(num_resblocks - 1):
                    blocks.append(ResBlock(out_channels, out_channels))
            
        
        
        self.group = nn.Sequential(*blocks)
        



        
        ###### END YOUR CODE HERE ############

        
    def forward(self, X):
        return self.group(X)
    
    
    
# Over the groups: 
#    the image size is shrinked by a half by each block group except for the first group, 
#    because a maxpool of stride 2 was just applied before it. 

class ResNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        # arch: a list of tuples that specifies each ResBlockGroup: (num_resblocks, in_channel, out_channels)
        #       in_channel: the channel count for the first conv of the first block of the group
        #       By changing this arch list, you can have different choices of architectures. 
        #       For ex., the following is the arch of ResNet-18: 
        #       [(2, 64, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512)]
        #       i.e.: 4 group; each group has 2 blocks; each block has 2 conv layers (excluding the 1x1 conv layer)
        #             That is: 4x2x2 = 16 conv layers
        #             plus the first 7x7 conv layer, and the final fully connected layer.
        #             A total of 18 layers.
        super().__init__()
        
        ######## initial layers ########
        intro = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        
        ####### residual block groups and the final pooling and flatten #######
        
        
        ###### START YOUR CODE HERE #########

        # use the passed-in `arch` to create the sequence of res block groups,
        # save the result in `rbgs`. Note that the `first_group` needs to be properly set. 
        
        res_groups = []
        for index, res_group in enumerate(arch):
                num_resblocks, in_channel, out_channel = res_group
                if index == 0:
                    res_groups.append(ResBlockGroup(num_resblocks, in_channel, out_channel, first_group=True))
                    
                else:
                    res_groups.append(ResBlockGroup(num_resblocks, in_channel, out_channel))
                    

        rbgs = nn.Sequential(*res_groups)
        
        

        ###### END YOUR CODE HERE ############
        
        
        
        
        ######## put initial layers and res block groups and final pooling and flatten together ########
        self.features = nn.Sequential(
                            intro, 
                            rbgs, 
                            nn.AdaptiveAvgPool2d((1, 1)), 
                            nn.Flatten()
                        )
        
        
        
        ######## dense layer #########
        self.classifier = nn.Linear(512, num_classes)
        
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
        


##########Coding for testing ##############
def run_tests():
    i = 1 
    

    num_classes = 10
    num_samples = 20
    num_channels = 3
    width = 224
    height = 224


    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")  # use the fixed cpu device for grading purpose. 


    ####****** Testing the res block 

    torch.manual_seed(10)  #fix seed for reproducibility
    X = torch.randn(num_samples, num_channels, height, width).to(device)

    print("\ntesting the res block class\n")
    
    #model_resblock = ResBlock(in_channel, out_channels, use_1x1conv=False, stride=1)
    
    model_resblock = ResBlock(3, 6, use_1x1conv=True, stride=1)
    Z = model_resblock(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(2072.6157), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    model_resblock = ResBlock(3, 6, use_1x1conv=True, stride=2)
    Z = model_resblock(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(1070.407), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    model_resblock = ResBlock(3, 3, use_1x1conv=False, stride=1)
    Z = model_resblock(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(1809.9084), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    model_resblock = ResBlock(3, 3, use_1x1conv=True, stride=2)
    Z = model_resblock(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(848.9090), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    
    
    
    
    ####****** testing the res block group
    
    print("\ntesting the res block group class\n")

    model_rbg = ResBlockGroup(1, 3, 3, first_group=False)
    Z = model_rbg(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(750.5177), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    model_rbg = ResBlockGroup(1, 3, 6, first_group=True)
    Z = model_rbg(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(1865.8810), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    model_rbg = ResBlockGroup(2, 3, 3, first_group=False)
    Z = model_rbg(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(1049.9005), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    model_rbg = ResBlockGroup(2, 3, 6, first_group=True)
    Z = model_rbg(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(2888.4202), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    model_rbg = ResBlockGroup(3, 3, 3, first_group=False)
    Z = model_rbg(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(1298.6190), rtol=1e-04, atol=1e-04), i=i)
    i += 1

    model_rbg = ResBlockGroup(3, 3, 6, first_group=True)
    Z = model_rbg(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(4030.6443), rtol=1e-04, atol=1e-04), i=i)
    i += 1
    
    
    
    
    ####****** testing the resnet-18
    print("\ntesting the resnet-18\n")
    arch = [(2, 64, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512)] #ResNet-18
    model_resnet = ResNet(arch, num_classes=10)
    Z = model_resnet(X)
    #print(Z.to('cpu').shape)
    #print(torch.linalg.norm(Z))
    report(torch.allclose(torch.linalg.norm(Z), torch.tensor(4.8063), rtol=1e-04, atol=1e-04), i=i)
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
    
