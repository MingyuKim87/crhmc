from sklearn.datasets import load_iris
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, layer_sizes, loss = "multi_class", bias=True):
        super().__init__() 
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.loss = loss
        self.bias = bias
        self.layers = self._make_moduleList()

    def _make_moduleList(self):
        # MLP embedding architectures
        layers = nn.ModuleList([])
        for i in range(1, len(self.layer_sizes)):
            if i < len(self.layer_sizes)-1:
                layers.append(nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], bias=self.bias))
                layers.append(torch.nn.ReLU())
            else:
                layers.append(nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], bias=self.bias))

        # make sequential
        sequential = nn.Sequential(*layers)

        return sequential


    def forward(self, x):
        output = self.layers(x)

        return output

class Conv_Net(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self, loss="multi_class"):
        super().__init__() 
        
        self.loss = loss
        
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=-1)
        return x