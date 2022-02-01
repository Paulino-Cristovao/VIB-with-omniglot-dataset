#
import argparse
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from loss import cuda

# Hyper-parameters
#num_features = 512
num_classes = 964

#fc1 = 512 #num_features
h4 = 256
h3 = 256
h2 = 128
h1 = 64

kernel_pool = 2
#stride = 1
k_s = 3
p = 1


class Net(nn.Module):
    def __init__(self, num_features, num_classes=num_classes, norm=True, scale=True):
        super(Net,self).__init__()
        self.extractor = Extractor()
        self.embedding = Embedding(num_features)
        self.classifier = Classifier(num_classes, num_features)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        #self.s = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        self.norm = norm
        self.scale = scale
        
      
   
    def norm_emb(self, x):
        if self.norm:
            x = self.l2_norm(x)
          
        if self.scale:
            x = self.s * x
        return x

    def _forward(self, x):
        x0 = self.extractor(x)
        x1_,x1 = self.embedding(x0)

        x1_ = self.norm_emb(x1_)
        x1 = self.norm_emb(x1)
      
        return x1_,x1
    
    def forward(self, x):
        x1_,x1 = self._forward(x)
        
        logit = self.classifier(x1_)
        
        return logit, x1_


    def helper_extract(self, x):
        x = self.extractor(x)

        x1_,x1 = self.embedding(x)

        x1_ = self.norm_emb(x1_)
        x1 = self.norm_emb(x1)
        

        return x1_,x1
    
    def forward_wi_fc1(self, x):
        x1_,x1 = self.helper_extract(x)
        logit = self.classifier(x1)
        
        return logit, x1
    
    def forward_wi_fc1_(self, x):
        x1_,x1 = self.helper_extract(x)
        logit = self.classifier(x1_)
        
        return logit, x1
    

    def extract(self, x):
        x = self.helper_extract(x)
        return x

    def l2_norm(self, input):
        return F.normalize(input, p=2, dim=1)

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))
        #self.classifier.fc.weight.data = F.normalize(w.view(w.size(0), -1), dim=1, p=2).view(w.size())
    
    
class Layer(nn.Module):
    def __init__(self,in_feature, output_feature, kernel_conv, kernel_pool=kernel_pool, p=p):
      super(Layer, self).__init__()
      self.in_feature = in_feature
      self.output_feature = output_feature
      self.kernel_conv = kernel_conv
      self.kernel_pool = kernel_pool
      self.p = p

    
      self.layer = nn.Sequential(
                  nn.Conv2d(self.in_feature, self.output_feature, self.kernel_conv, self.p),
                  nn.ReLU(),
                  nn.BatchNorm2d(self.output_feature),
                  nn.MaxPool2d(self.kernel_pool),
                  nn.ReLU(),
                  )
     

    def forward(self, x):
      return self.layer(x) 

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor,self).__init__()
        #basenet = models.resnet18(pretrained=True)
        #basenet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        #self.extractor = nn.Sequential(*list(basenet.children())[:-1])

        self.fc1 = Layer(1, h1, k_s, p)
        self.fc2 = Layer(h1, h2, k_s, p)
        self.fc3 = Layer(h2, h3, k_s, p)
        self.fc4 = Layer(h3, h4, k_s, p)

        #self.dropout = nn.Dropout(p=0.80)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(x.size(0), -1)

        return x

class Embedding(nn.Module):
    def __init__(self,fc1):
        super(Embedding,self).__init__()
        self.fc1 = nn.Linear(20*20*h4, fc1)
        
        
    def forward(self, x):
        x = self.fc1(x)
        fc1 = F.relu(x)
        
        
        return fc1, x


class Classifier(nn.Module):
    def __init__(self, num_classes,fc1):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(fc1, num_classes, bias=False)
        
    def forward(self, x):
        x = self.fc(x)  
       
        return x
