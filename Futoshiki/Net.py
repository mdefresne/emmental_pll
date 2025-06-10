import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):

    """
    For initializing weights of linear layers (bias are put to 0).
    """
    
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)
    

class ResBlock(nn.Module):
    
    """ 
    Residual block of 2 hidden layer for resMLP 
    Init: size of the input (the output layer as the same dimension for the sum)
          size of the 1st hidden layer
    """

    def __init__(self, input_size, hidden_size):
        super(ResBlock, self).__init__()   
        

        self.MLP = nn.Sequential(
            nn.BatchNorm1d(num_features=input_size),
            nn.ReLU(), 
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, input_size)
            )

    def forward(self, x):
        x_out = self.MLP(x)
        x = x_out + x 
        
        return(x)
        
        
class ResMLP(nn.Module):
    
    """
    ResMLP with nblocks residual blocks of 2 hidden layers.
    Init: size of the output output size (int)
          size of the input (int)
          size of the hidden layers 
    """
    
    def __init__(self, output_size, input_size, hidden_size, nblocks = 2):
        super(ResMLP, self).__init__()   

        self.ResNet = torch.nn.Sequential()
        self.ResNet.add_module("In_layer", nn.Linear(input_size, hidden_size))
        self.ResNet.add_module("relu_1", torch.nn.ReLU())
        for k in range(nblocks):
            self.ResNet.add_module("ResBlock" + str(k), ResBlock(hidden_size, hidden_size))
        self.ResNet.add_module("Final_BN", nn.BatchNorm1d(num_features=hidden_size))
        self.ResNet.add_module("relu_n", torch.nn.ReLU())
        self.ResNet.add_module("Out_layer", nn.Linear(hidden_size, output_size))


    def forward(self, x):
        
        x = self.ResNet(x)
        
        return(x)
    

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        #logits = torch.rand(logits.shape, device = logits.device)
        return logits #probs




class Net(nn.Module):
    
    """
    Network composed of embedding + MLP
    Init: grid_size (int)
          hiddensize (int): number of neurons in hidden layer (suggestion: 128 or 256)
          resNet (bool). If False (default), use a regular MLP. Else, use a ResMLP
          nblocks (int): number of residual blocks. Default is 2
    """
    
    def __init__(self, nb_var, nb_val, nb_feat, hidden_size = 128, nblocks = 2):
        super(Net, self).__init__()
    
        self.feature_size = nb_feat # ex 5 for futoshi : column, line, column, line, inequality
        input_size = self.feature_size
        self.nb_var= nb_var
        self.nb_val= nb_val
        
        self.MLP = ResMLP(self.nb_val**2, input_size, hidden_size, nblocks)      
        self.MLPu = ResMLP(self.nb_val, input_size, hidden_size,1) #unary net      
        self.MLP.apply(weights_init)  
        self.MLPu.apply(weights_init)  
        

    def forward(self, x, device, unary = False):
        bs = x.shape[0]
        """
        pred = self.MLP(x.reshape((-1,self.feature_size)))

        pred= pred.reshape(bs, self.nb_var, self.nb_var, self.nb_val, self.nb_val)
        pred=(pred+pred.transpose(1,2).transpose(3,4))/2
        #pred = pred**2
        if unary:
            unary = pred[:,torch.arange(self.nb_var),torch.arange(self.nb_var)][:,:,torch.arange(self.nb_val), torch.arange(self.nb_val)]
        else:
            unary = torch.zeros(bs,self.nb_var,self.nb_val).to(x.device)
        pred[:,torch.arange(self.nb_var),torch.arange(self.nb_var)]=0
        
        return pred, unary
        """
        bs = x.shape[0]
        t = torch.triu_indices(self.nb_var,self.nb_var,1)
        rr = x[:,t[0],t[1]].reshape(-1,self.feature_size)
        pred = self.MLP(rr)
        if unary:
            un = x[:,torch.arange(self.nb_var, device = device),torch.arange(self.nb_var, device = device)].reshape(-1,self.feature_size)
            predu = self.MLPu(un) # prediction termes unitaires
            predu = predu.reshape(bs,self.nb_var,self.nb_val)
        else:
            predu = torch.zeros(bs,self.nb_var,self.nb_val, device = device)

        pred = pred.reshape(bs,-1,self.nb_val,self.nb_val)
        out = torch.zeros(bs,self.nb_var,self.nb_var,self.nb_val,self.nb_val, device = device)
        out[:,t[0],t[1]] = pred
        pred = torch.swapaxes(pred,2,3)
        out[:,t[1],t[0]] = pred
        predu = predu-predu.min(axis=-1)[0].detach()[:,:,None]
        out = out-out.min(axis=-1)[0].min(axis=-1)[0].detach()[:,:,:,None,None]
        return(out, predu) 


class VerySimpleNet(nn.Module):
    
    
    def __init__(self, nb_var, nb_val, nb_feat, device = "cpu"):
        super().__init__()
        self.nb_var= nb_var
        self.nb_val= nb_val
        self.W = torch.nn.Parameter(torch.rand((nb_var,nb_var,nb_val,nb_val), device = device, requires_grad=True))
        self.unary = torch.nn.Parameter(torch.rand((nb_var,nb_val), device = device, requires_grad=True))

    def forward(self, x, device, unary = False):
        bs = x.shape[0]
        pred = self.W[None,...].expand(bs,*self.W.shape)
        unary = self.unary[None,...].expand(bs,*self.unary.shape)
        return pred, unary


