import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

import os
import pandas as pd
from torchvision.io import read_image




class Layer:
    ''' Базовый слой. От него будут наследоваться все проклятые Богом слои'''
    def __init__(self):
        self.param = None
    #def __repr__(self):
        # Функция, которая вызовется при print()
        
    def __call__(self, data):
        return data

class Linear(Layer):
    def __init__(self, input_len, output_len):
        self.W = torch.rand(input_len+1, output_len, requires_grad=True)
        self.param = [self.W]
    
    def __call__(self, X):
        X_1 = torch.ones(X.size()[1]).unsqueeze(0)
        X_con = torch.cat((X_1, X),0)
        
        return self.W.T @ X_con
        
class Model:
    def __init__(self):
        ''' Набиваем модель слоями. Исполнение идет с лева на право: X------> '''
        self.layers = [Layer(), Layer(), Layer(), Layer()]

    #@property
    def __repr__(self):
        return "Bobr Kurva"
    
    def __call__(self, data):
        return self.forward(data)

    def parameters(self):
        parameters_arr = []

        for i in range(len(self.layers)):
            parameters_arr.append(self.layers[i].param)
        
        return parameters_arr
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class Optimizator:
    def __init__(self, model_parameters, lr=0.001):
        self.model_parameters = model_parameters[::-1]
        self.lr = lr
    
    def zero_grad(self):
        ''' Тут мы сбрасывыем  градиенты в нуль '''
        for i in range(len(self.model_parameters)):
            for j in range(len(self.model_parameters[i])):
                self.model_parameters[i][j].retain_grad()

    
    def step(self):
        ''' Тут мы коррентируем веса в соотвествии с SGD '''
        for i in range(len(self.model_parameters)):
            for j in range(len(self.model_parameters[i])):
                #print(self.model_parameters[i][j])
                self.model_parameters[i][j] = self.model_parameters[i][j] - self.lr * self.model_parameters[i][j].grad.data

class MyModel(Model):
    def __init__(self):
        ''' Самая простая линейная модель '''
        self.layers = [Linear(1, 1)]

if __name__ == "__main__":
    model = MyModel()
    loss_fn = nn.MSELoss()
    optimizer = Optimizator(model.parameters(), lr=1e-3)
    
    
    x1 = torch.arange(-15, 15, 0.1)
    x2 = torch.arange(-15, 15, 0.1) / 5
    x = torch.stack([x1, x2], dim=1)
    y = x[:,0] * 2. + 0.2 * x[:, 1]**2 - 3 + torch.normal(0., 0.2, (1, 300))
    #plt.scatter(x[:, 0], y)
    
    for i in range(1):
        preds = model(x[:, 0].unsqueeze(0))
        loss = loss_fn(preds, y)
        #loss = torch.mean((preds - y) * (preds - y))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(model.layers[0].W)
        print(optimizer.model_parameters[0][0])
        
    #    optimizer.step()
    
    plt.scatter(x[:, 0], y)
    plt.scatter(x[:, 0], preds.detach().numpy())
