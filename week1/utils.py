#from __future__ import print_function
import os
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MiTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.images = []
        self.labels = []
        for i, cls in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def train(model, dataloader_train, criterion, optimizer, params, device):
  model.train()
  train_loss = 0.0
  correct = 0
  total = 0
  for imgs, labels in dataloader_train:

      imgs, labels = imgs.to(device), labels.to(device)
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(imgs, params)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # Compute training accuracy and loss
      train_loss += loss.item() * imgs.size(0)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  train_loss /= len(dataloader_train.dataset)
  train_accuracy = correct/total
  #print(f'Training loss in epoch {epoch}: {train_loss}')
  return train_loss, train_accuracy

def validation(model, dataloader_val, criterion, params, device):
  model.eval()
  val_loss = 0.0
  correct = 0
  total = 0
  with torch.no_grad():
      for imgs, labels in dataloader_val:
          imgs, labels = imgs.to(device), labels.to(device)

          outputs =  model(imgs, params)
          loss = criterion(outputs, labels)
          
          val_loss += loss.item()
          _, predicted = torch.max(outputs, 1)
          
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  val_loss /= len(dataloader_val)
  val_accuracy = correct / total
  #print(f'Validation loss and accuracy in epoch {epoch}: {val_loss} / {val_accuracy}')
  return val_loss, val_accuracy



def data_augmentation(augment: bool, params, width, height):
  """
  Generates augmented data.
  :param augment: Boolean, when true the data augmentation is done.
  :return: ImageDataGenerator object instance
  """
  rotation_range=0
  width_shift_range=0.
  height_shift_range=0.
  shear_range=0.
  zoom_range=0.
  horizontal_flip=0


  if 'rotation' in params['data_aug']:
      rotation_range = 20
  if 'wsr' in params['data_aug']:
      width_shift_range=0.2
  if 'hsr' in params['data_aug']:
      height_shift_range=0.2
  if 'sr' in params['data_aug']:
      shear_range=0.2
  if 'zr' in params['data_aug']:
      zoom_range=0.2
  if 'hf' in params['data_aug']:
      horizontal_flip=0.2

  if augment:
    transform = transforms.Compose([
      transforms.Resize((width, height)),
      transforms.RandomHorizontalFlip(horizontal_flip),
      transforms.RandomRotation(rotation_range),
      transforms.RandomAffine(degrees=0, shear=(-shear_range, shear_range)), #Shear
      transforms.RandomResizedCrop(size=(width, height), scale=(1.0-zoom_range, 1.0)), #Zoom
      transforms.ToTensor(),
      # transforms.Lambda(lambda x: x / 255.0),  # Normalize by dividing by 255
      transforms.Normalize(
        mean=[0.43239, 0.45149, 0.44213],
        std=[0.255, 0.244, 0.270]
      )])
        
    return transform

  else:

    transform = transforms.Compose([
                transforms.Resize((width, height)),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: x / 255.0),  # Normalize by dividing by 255
                transforms.Normalize(
                  mean=[0.43239, 0.45149, 0.44213],
                  std=[0.255, 0.244, 0.270]
                )])
    
    return transform
    

def get_optimizer(params, model):
    if params['optimizer'] == 'adam':
        #optimizer = Adam(learning_rate=float(params['lr']), beta_1=float(params['momentum']))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params['lr']))
    elif params['optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=float(params['lr']), rho=float(params['momentum']))
    elif params['optimizer'] == 'sgd':
        #optimizer = SGD(learning_rate=float(params['lr']), momentum=float(params['momentum']))
        optimizer = torch.optim.SGD(model.parameters(), lr = float(params['lr']))
    elif params['optimizer'] == 'rmsprop':
        #optimizer = RMSprop(learning_rate=float(params['lr']), rho=float(params['momentum']))
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


class Model(nn.Module):
  def __init__(self, params):
    super(Model, self).__init__()
    
    output_size = int(params['n_filters_1'])
    if params['depth'] > 1:
      output_size = int(params['n_filters_2'])
      if params['depth'] > 2:
        output_size = int(params['n_filters_3'])
        if params['depth'] > 3:
          output_size = int(params['n_filters_4'])
    
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
        
  def forward(self, x, params):
    x = self.conv_block1(x)
    
    if params['depth'] > 1:
      x = self.pool(x)
      x = self.conv_block2(x)

      if params['depth'] > 2:
        x = self.pool(x)
        x = self.conv_block3(x)

        if params['depth'] > 3:
          x = self.pool(x)
          x = self.conv_block4(x)

    # x = self.pool(x)    
    x = self.globavgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    x = self.relu(x)
    x = self.fc2(x)
    # x = F.softmax(x,dim=1)
    return x
      
