import torch
import torch.nn as nn
import torch.nn.functional as F


def get_optimizer(params, model):
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params['lr']))
    elif params['optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=float(params['lr']), rho=float(params['momentum']))
    elif params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = float(params['lr']))
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=float(params['lr']))

    return optimizer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=True, pooling='max', batch_norm=True, dropout=0):
        super(ConvBlock, self).__init__()
        if padding=='same':
            padding = int((in_channels*stride - in_channels - stride + kernel_size)/2)
        else:
            padding = 0

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))  # Batch normalization
        
        layers.append(nn.ReLU())  # ReLU activation function
        
        if dropout!=0:
            layers.append(nn.Dropout(dropout)) #Dropout
            
        self.conv_block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv_block(x)



class CustomModel(torch.nn.Model):
    def __init__(self, params):
        super(CustomModel, self).__init__()

        if params['depth'] > 3:
            output_size = int(params['n_filters_4'])
        elif params['depth'] > 2:
            output_size = int(params['n_filters_3'])
        elif params['depth'] > 1:
            output_size = int(params['n_filters_2'])
        else: 
            output_size = int(params['n_filters_1'])

        self.conv_block1 = ConvBlock(in_channels=3, out_channels=int(params['n_filters_1']), kernel_size=int(params['kernel_size_1']), stride= params['stride'], padding=params['padding'], batch_norm=params['bn'], dropout=float(params['dropout']))
        self.conv_block2 = ConvBlock(in_channels=int(params['n_filters_1']), out_channels=int(params['n_filters_2']), kernel_size=int(params['kernel_size_2']), stride= params['stride'], padding=params['padding'], batch_norm=params['bn'], dropout=float(params['dropout']))
        self.conv_block3 = ConvBlock(in_channels=int(params['n_filters_2']), out_channels=int(params['n_filters_3']), kernel_size=int(params['kernel_size_3']), stride= params['stride'], padding=params['padding'], batch_norm=params['bn'], dropout=float(params['dropout']))        
        self.conv_block4 = ConvBlock(in_channels=int(params['n_filters_3']), out_channels=int(params['n_filters_4']), kernel_size=int(params['kernel_size_4']), stride= params['stride'], padding=params['padding'], batch_norm=params['bn'], dropout=float(params['dropout']))

        if params['pool'] == 'max':
            self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        elif params['pool'] == 'avg':
            self.pool = nn.AvgPool2d(2, 2)  #Avg pooling    

        self.globavgpool = nn.AdaptiveAvgPool2d((1, 1)) #Global avg pooling
        self.fc = nn.Linear(output_size, int(params['neurons']))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(params['neurons']), params['output'])

    def forwards(self, x, params):
        if params['depth'] > 1:
            x = self.pool(x)
            x = self.conv_block2(x)

        if params['depth'] > 2:
            x = self.pool(x)
            x = self.conv_block3(x)

        if params['depth'] > 3:
            x = self.pool(x)
            x = self.conv_block4(x)

        x = self.pool(x)    
        x = self.globavgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.softmax(x,dim=1)
        return x
