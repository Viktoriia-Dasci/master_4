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

HGG_list_t2 = sorted(glob.glob('/home/viktoriia.trokhova/TrainT2/HGG_t2/*.nii'))
LGG_list_t2 = sorted(glob.glob('/home/viktoriia.trokhova/TrainT2/LGG_t2/*.nii'))

#train HGG
HGG_train_list_t2 = sorted(glob.glob('/home/viktoriia.trokhova/T2_split/train/HGG_t2/*.nii'))
#HGG_mask_train_list_t2 = sorted(glob.glob('/home/viktoriia.trokhova/T2_split/train/HGG_masks/*.nii'))
#val HGG
HGG_val_list_t2 = sorted(glob.glob('/home/viktoriia.trokhova/T2_split/val/HGG_t2/*.nii'))
#HGG_mask_val_list_t2 = sorted(glob.glob('/content/drive/MyDrive/T2_split/val/HGG_masks/*.nii'))
#test HGG
HGG_test_list_t2 = sorted(glob.glob('/home/viktoriia.trokhova/T2_split/test/HGG_t2/*.nii'))
#HGG_mask_test_list_t2 = sorted(glob.glob('/content/drive/MyDrive/T2_split/test/HGG_masks/*.nii'))

#train LGG
LGG_train_list_t2 = sorted(glob.glob('/home/viktoriia.trokhova/T2_split/train/LGG_t2/*.nii'))
#LGG_mask_train_list_t2 = sorted(glob.glob('/content/drive/MyDrive/T2_split/train/LGG_masks/*.nii'))
#val LGG
LGG_val_list_t2 = sorted(glob.glob('/home/viktoriia.trokhova/T2_split/val/LGG_t2/*.nii'))
#LGG_mask_val_list_t2 = sorted(glob.glob('/content/drive/MyDrive/T2_split/val/LGG_masks/*.nii'))
#test LGG
LGG_test_list_t2 = sorted(glob.glob('/home/viktoriia.trokhova/T2_split/test/LGG_t2/*.nii'))
#LGG_mask_test_list = sorted(glob.glob('/content/drive/MyDrive/T2_split/test/LGG_masks/*.nii'))

HGG_train = []
LGG_train = []
HGG_val = []
LGG_val = []
HGG_test = []
LGG_test = []
for img in range(len(HGG_train_list_t2)):   #Using t1_list as all lists are of same size   
    train_image_t2_HGG = nib.load(HGG_train_list_t2[img]).get_fdata()
    HGG_train.append(train_image_t2_HGG)

for img in range(len(LGG_train_list_t2)):
    train_image_t2_LGG = nib.load(LGG_train_list_t2[img]).get_fdata()
    LGG_train.append(train_image_t2_LGG)

for img in range(len(HGG_val_list_t2)):   #Using t1_list as all lists are of same size   
    val_image_t2_HGG = nib.load(HGG_val_list_t2[img]).get_fdata()
    HGG_val.append(val_image_t2_HGG)

for img in range(len(LGG_val_list_t2)):
    val_image_t2_LGG = nib.load(LGG_val_list_t2[img]).get_fdata()
    LGG_val.append(val_image_t2_LGG)

for img in range(len(HGG_list_t2)):   #Using t1_list as all lists are of same size   
    test_image_t2_HGG = nib.load(HGG_test_list_t2[img]).get_fdata()
    HGG_test.append(test_image_t2_HGG)

for img in range(len(LGG_list_t2)):
    test_image_t2_LGG = nib.load(LGG_test_list_t2[img]).get_fdata()
    HGG_test.append(test_image_t2_LGG)

# Put X and y to device
#X = torch.tensor(X, dtype=torch.float32).to(device)
#y = torch.tensor(y, dtype=torch.long).to(device)

# Combine the HGG and LGG lists
X_train = np.array(HGG_train + LGG_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
y_train = np.array([0] * len(HGG_train) + [1] * len(LGG_train))

X_val = np.array(HGG_val + LGG_val)
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], X_val.shape[3], 1))
y_val = np.array([0] * len(HGG_val) + [1] * len(LGG_val))

X_test = np.array(HGG_test + LGG_test)
y_test = np.array([0] * len(HGG_test) + [1] * len(LGG_test))


# Print the shapes of the train and test sets
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_train shape:', X_val.shape)
print('y_train shape:', y_val.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Flatten, Dense, Softmax


import torch.nn as nn

def custom_3d_cnn(input_shape=(240, 240, 155, 1)):
    inputs = nn.Input(shape=input_shape)

    # Layer 1
    conv1 = nn.Conv3d(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(inputs)
    bn1 = nn.BatchNorm3d()(conv1)

    # Layer 2
    conv2 = nn.Conv3d(64, kernel_size=(3, 3, 3), padding='same')(bn1)
    mp2 = nn.MaxPool3d(pool_size=(2, 2, 2))(conv2)
    bn2 = nn.BatchNorm3d()(mp2)

    # Layer 3
    conv3 = nn.Conv3d(128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(bn2)
    mp3 = nn.MaxPool3d(pool_size=(2, 2, 2))(conv3)
    bn3 = nn.BatchNorm3d()(mp3)

    # Layer 4
    conv4 = nn.Conv3d(256, kernel_size=(3, 3, 3), padding='same')(bn3)
    mp4 = nn.MaxPool3d(pool_size=(2, 2, 2))(conv4)
    bn4 = nn.BatchNorm3d()(mp4)

    # Layer 5
    conv5 = nn.Conv3d(256, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(bn4)
    bn5 = nn.BatchNorm3d()(conv5)

    # Layer 6
    conv6 = nn.Conv3d(128, kernel_size=(3, 3, 3), padding='same')(nn.Upsample(size=(2, 2, 2))(bn5))
    bn6 = nn.BatchNorm3d()(conv6)

    # Layer 7
    conv7 = nn.Conv3d(64, kernel_size=(3, 3, 3), padding='same')(nn.Upsample(size=(2, 2, 2))(bn6))
    bn7 = nn.BatchNorm3d()(conv7)

    # Layer 8
    conv8 = nn.Conv3d(32, kernel_size=(3, 3, 3), padding='same')(bn7)

    # Layer 9
    flatten = nn.Flatten()(conv8)
    fc1 = nn.Linear(256)(flatten)

    # Layer 10
    fc2 = nn.Linear(256)(fc1)

    # Layer 11
    softmax = nn.Softmax()(fc2)

    # Create model
    model = nn.Model(inputs=inputs, outputs=softmax)

    return model





# Define the transformation to be applied to the images
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((240,240)),                                  
                                transforms.Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225],),
])

# Convert the train, val and test data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

# Define the test dataset
test_dataset = TensorDataset(X_test, y_test)

# Define the dataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model
model = custom_3d_cnn()

# Move the model to the device
model = model.to(device)

# Define the model
#model = custom_3d_cnn().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.004)


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

from hyperopt import fmin, tpe, hp
from hyperopt.pyll.base import scope

# Define the hyperparameter search space
space = {
    'lr': hp.loguniform('lr', -6, -3),
    'momentum': hp.uniform('momentum', 0.1, 0.9)
}

from hyperopt import Trials

# Define the objective function to minimize
def objective(params):
    optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)
    return {'loss': 1 - train_accuracy / 100, 'status': 'ok'}

trials = Trials()  # Define the 'trials' variable

def hyperband_stopping(trials, trial, result, early_stopping_rounds):
    if len(trials.trials) < early_stopping_rounds:
        return False
    best_trial = max(trials.trials, key=lambda t: t['result']['loss'])
    if trial.number >= best_trial.number + early_stopping_rounds:
        return True
    else:
        return False


# Run the Hyperband algorithm to find the best hyperparameters
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=81,
            rstate=np.random.seed(42),
            #early_stop_fn=hyperband_stopping,
            verbose=1)

# Update the optimizer with the best hyperparameters
optimizer = optim.SGD(model.parameters(), lr=best['lr'], momentum=best['momentum'])

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

# Define the testing loop
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.permute(0, 3, 1, 2).to(device), target.to(device) # Permute dimensions
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * test_correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# Train and val the model
for epoch in range(30):
    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)
    val_loss, val_accuracy = validation(model, device, val_loader, criterion)
    print('Epoch: {} \tTrain Loss: {:.6f} \tTrain Accuracy: {:.2f}% \tVal Loss: {:.6f} \tVal Accuracy: {:.2f}%'.format(
        epoch+1, train_loss, train_accuracy, val_loss, val_accuracy))

# Evaluate the model on the test set
test_loss, test_accuracy = test(model, device, test_loader, criterion)
print('Test Loss: {:.6f} \tTest Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
