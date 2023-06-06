# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""ResnetT2.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1GnGCgunOxD7-J7k0ngzxGU13zVOLyvD5
"""

import numpy as np
import nibabel as nib
import glob
import os
import random
import tensorflow as tf
#import splitfolders  # or import split_folders
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from tifffile import imsave
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report,confusion_matrix
from PIL import Image
from keras.models import load_model
from skimage.color import rgb2gray
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F


# Define the device to be used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_from_dir(path):
      file_paths = glob.glob(os.path.join(path, '*.npy'))
   
      slices_list=[]
      for img in range(len(file_paths)):
          new_img = np.load(file_paths[img])
          slices_list.append(new_img)
      return slices_list



HGG_train = load_from_dir('/home/viktoriia.trokhova/Stacked_4/train/HGG_stack')
LGG_train = load_from_dir('/home/viktoriia.trokhova/Stacked_4/train/LGG_stack')


HGG_val = load_from_dir('/home/viktoriia.trokhova/Stacked_4/val/HGG_stack')
LGG_val = load_from_dir('/home/viktoriia.trokhova/LGG_stack')

# Put X and y to device
#X = torch.tensor(X, dtype=torch.float32).to(device)
#y = torch.tensor(y, dtype=torch.long).to(device)

# Combine the HGG and LGG lists
X_train = np.array(HGG_train + LGG_train)
y_train = np.array([0] * len(HGG_train) + [1] * len(LGG_train))

X_val = np.array(HGG_val + LGG_val)
y_val = np.array([0] * len(HGG_val) + [1] * len(LGG_val))

# X_test = np.array(HGG_test + LGG_test)
# y_test = np.array([0] * len(HGG_test) + [1] * len(LGG_test))


# Print the shapes of the train and test sets
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_train shape:', X_val.shape)
print('y_train shape:', y_val.shape)
# print('X_test shape:', X_test.shape)
# print('y_test shape:', y_test.shape)

class Effnet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Load the pretrained ResNet50 model
        efficientnet_b1 = EfficientNet.from_pretrained('efficientnet-b1')

        # Replace the first convolutional layer to handle images with shape (240, 240, 4)
        efficientnet_b1.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Reuse the other layers from the pretrained ResNet50 model
        self.features = nn.Sequential(*list(efficientnet_b1.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        in_features = efficientnet_b1._fc.in_features
        self.fc1 = nn.Linear(in_features, out_features=128, bias=True)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, dropout = nn.Dropout(p=0.4)
):
        x = self.features(x)
        x = self.avgpool(x)
        x = dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(self.fc2(x))

        return x

# class MyCustomEfficientNetB1(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
        
#         efficientnet_b1 = EfficientNet.from_pretrained('efficientnet-b1')
#         self.features = efficientnet_b1.extract_features
#         in_features = efficientnet_b1._fc.in_features
#         self.attention = SelfAttention(in_features)
#         self.last_pooling_operation = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc1 = nn.Linear(in_features, 128)
#         self.fc2 = nn.Linear(128, 2)


#     def forward(self, input_imgs):
#         images_feats = self.features(input_imgs.cpu())
#         images_att = self.attention(images_feats.cuda())
#         output = self.last_pooling_operation(images_att)
#         output = output.view(input_imgs.size(0), -1)
#         images_outputs = self.fc1(output)
#         #output = dropout(images_outputs)
#         images_outputs = F.relu(self.fc2(output))
#         #images_outputs = nn.ReLU(self.fc2(output))
    
    

# Define the transformation to be applied to the images
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224,224)),                                  
                                transforms.Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225],),
])

# Convert the train, val and test data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).long()
# X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).long()

# Define the test dataset
#test_dataset = TensorDataset(X_test, y_test)

# Define the dataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
#test_dataset = TensorDataset(X_test, y_test)

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
model = Effnet().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.004)


# Define the training loop
def train(model, device, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.permute(0, 3, 1, 2).to(device), target.to(device) # Permute dimensions
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    return train_loss, train_accuracy

# from hyperopt import fmin, tpe, hp
# from hyperopt.pyll.base import scope

# # Define the hyperparameter search space
# space = {
#     'lr': hp.loguniform('lr', -6, -3),
#     'momentum': hp.uniform('momentum', 0.1, 0.9)
# }

# from hyperopt import Trials

# # Define the objective function to minimize
# def objective(params):
#     optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
#     train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)
#     return {'loss': 1 - train_accuracy / 100, 'status': 'ok'}

# trials = Trials()  # Define the 'trials' variable

# def hyperband_stopping(trials, trial, result, early_stopping_rounds):
#     if len(trials.trials) < early_stopping_rounds:
#         return False
#     best_trial = max(trials.trials, key=lambda t: t['result']['loss'])
#     if trial.number >= best_trial.number + early_stopping_rounds:
#         return True
#     else:
#         return False


# # Run the Hyperband algorithm to find the best hyperparameters
# best = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=81,
#             rstate=np.random.seed(42),
#             #early_stop_fn=hyperband_stopping,
#             verbose=1)

# # Update the optimizer with the best hyperparameters
# optimizer = optim.SGD(model.parameters(), lr=best['lr'], momentum=best['momentum'])

# Train the model with the best hyperparameters
train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)


# Define the validating loop
def validation(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.permute(0, 3, 1, 2).to(device), target.to(device) # Permute dimensions
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * val_correct / len(val_loader.dataset)
    return val_loss, val_accuracy

# # Define the testing loop
# def test(model, device, test_loader, criterion):
#     model.eval()
#     test_loss = 0
#     test_correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.permute(0, 3, 1, 2).to(device), target.to(device) # Permute dimensions
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             test_correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     test_accuracy = 100. * test_correct / len(test_loader.dataset)
#     return test_loss, test_accuracy

# Train and val the model
for epoch in range(30):
    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)
    val_loss, val_accuracy = validation(model, device, val_loader, criterion)
    print('Epoch: {} \tTrain Loss: {:.6f} \tTrain Accuracy: {:.2f}% \tVal Loss: {:.6f} \tVal Accuracy: {:.2f}%'.format(
        epoch+1, train_loss, train_accuracy, val_loss, val_accuracy))

# Evaluate the model on the test set
#test_loss, test_accuracy = test(model, device, test_loader, criterion)
#print('Test Loss: {:.6f} \tTest Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
