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
from efficientnet_pytorch import EfficientNet

from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

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


def plot_acc_loss_f1(history, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    epochs = range(1, len(history['loss']) + 1)
    
    plt.plot(epochs, history['loss'], 'y', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'loss_stacked.png'))
    plt.close()
    
    plt.plot(epochs, history['accuracy'], 'y', label='Training accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'accuracy_stacked.png'))
    plt.close()

    plt.plot(epochs, history['f1_score'], 'y', label='Training F1 Score')
    plt.plot(epochs, history['val_f1_score'], 'r', label='Validation F1 Score')
    plt.title('Training and validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'f1_score_stacked.png'))
    plt.close()


HGG_train = load_from_dir('/home/viktoriia.trokhova/Stacked_4/train/HGG_stack')
LGG_train = load_from_dir('/home/viktoriia.trokhova/Stacked_4/train/LGG_stack')


HGG_val = load_from_dir('/home/viktoriia.trokhova/Stacked_4/val/HGG_stack')
LGG_val = load_from_dir('/home/viktoriia.trokhova/LGG_stack')

from torch.utils.data import TensorDataset

X_train = np.array(HGG_train + LGG_train)
X_val = np.array(HGG_val + LGG_val)

# Create the labels array
y_train = np.array([0] * len(HGG_train) + [1] * len(LGG_train))
y_val = np.array([0] * len(HGG_val) + [1] * len(LGG_val))

class_counts = Counter(y_train)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
print(class_weights)

class_weights_np = np.array(class_weights, dtype=np.float32)
class_weights_tensor = torch.from_numpy(class_weights_np)
if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor

# Convert labels to categorical tensor
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Convert arrays to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

# Convert labels to one-hot encoded format
num_classes = len(np.unique(y_train))
y_train_categorical = torch.nn.functional.one_hot(y_train_tensor, num_classes=num_classes)
y_val_categorical = torch.nn.functional.one_hot(y_val_tensor, num_classes=num_classes)

# Create train and validation datasets
train_dataset = TensorDataset(X_train_tensor, y_train_categorical)
val_dataset = TensorDataset(X_val_tensor, y_val_categorical)

    
# Print the shapes of the train and test sets
# print('X_train shape:', X_train.shape)
# print('y_train shape:', y_train.shape)
# print('X_train shape:', X_val.shape)
# print('y_train shape:', y_val.shape)
# # print('X_test shape:', X_test.shape)
# # print('y_test shape:', y_test.shape)
# from efficientnet_pytorch import EfficientNet

# X_train = torch.from_numpy(X_train).float()
# y_train = torch.from_numpy(y_train).long()
# X_val = torch.from_numpy(X_val).float()
# y_val = torch.from_numpy(y_val).long()
# # X_test = torch.from_numpy(X_test).float()
# # y_test = torch.from_numpy(y_test).long()
# # Define the test dataset
# #test_dataset = TensorDataset(X_test, y_test)
# # Define the dataset
# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)


# def add_labels(X, y, images_list, label):
#     for img in images_list:
#         X.append(img)
#         y.append(label)
#     return X, y

# # Assuming HGG_list_new_train, LGG_list_new_train, HGG_list_new_val, LGG_list_new_val are already defined

# X_train, y_train = add_labels([], [], HGG_train, label='HGG')
# X_train, y_train = add_labels(X_train, y_train, LGG_train, label='LGG')
# X_val, y_val = add_labels([], [], HGG_val, label='HGG')
# X_val, y_val = add_labels(X_val, y_val, LGG_val, label='LGG')

# # Convert labels to numerical values
# labels = {'HGG': 0, 'LGG': 1}
# y_train = [labels[y] for y in y_train]
# y_val = [labels[y] for y in y_val]

# X_train_array = np.array(X_train)
# y_train_array = np.array(y_train)

# X_val_array = np.array(X_val)
# y_val_array = np.array(y_val)

# # Convert data to tensors
# X_train_tensor = torch.tensor(X_train_array)
# y_train_tensor = torch.tensor(y_train_array)

# X_val_tensor = torch.tensor(X_val_array)
# y_val_tensor = torch.tensor(y_val_array)

# # Convert labels to one-hot encoding
# num_classes = len(set(y_train))
# y_train_one_hot = torch.nn.functional.one_hot(y_train_tensor, num_classes=num_classes).float()
# y_val_one_hot = torch.nn.functional.one_hot(y_val_tensor, num_classes=num_classes).float()

# # Shuffle the data
# X_val_tensor, y_val_one_hot = shuffle(X_val_tensor, y_val_one_hot, random_state=101)
# X_train_tensor, y_train_one_hot = shuffle(X_train_tensor, y_train_one_hot, random_state=101)

# print(X_train_tensor.shape)
# print(y_train_one_hot.shape)
# print(X_val_tensor.shape)
# print(y_val_one_hot.shape)


# # Create datasets
# train_dataset = TensorDataset(X_train_tensor, y_train_one_hot)
# val_dataset = TensorDataset(X_val_tensor, y_val_one_hot)

# class MyCustomResnet50(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         # Load the pretrained ResNet50 model
#         resnet50 = models.resnet50(pretrained=True)
#         # Replace the first convolutional layer to handle images with shape (240, 240, 155)
#         resnet50.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # Reuse the other layers from the pretrained ResNet50 model
#         self.features = nn.Sequential(*list(resnet50.children())[:-2])
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.fc1 = nn.Linear(in_features=2048, out_features=128, bias=True)
#         self.fc2 = nn.Linear(128, 2)
#     def forward(self, x, dropout = nn.Dropout(p=0.79)
# ):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = dropout(x)
#         x = F.relu(self.fc2(x))
#         return x



class Effnet(nn.Module):
    def __init__(self, pretrained=True, dense_0_units=None, dense_1_units=None, dropout=None):
        super().__init__()

        # Load the pretrained EfficientNet-B1 model
        efficientnet_b1 = EfficientNet.from_pretrained('efficientnet-b1')

        # Replace the first convolutional layer to handle images with shape (240, 240, 4)
        efficientnet_b1._conv_stem = nn.Conv2d(4, 32, kernel_size=3, stride=2, bias=False)
        
        # Reuse the other layers from the pretrained EfficientNet-B1 model
        self.features = efficientnet_b1.extract_features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(dropout)
        if dense_0_units is not None:
            dense_0_units = int(dense_0_units)
            self.fc1 = nn.Linear(in_features=1280, out_features=dense_0_units, bias=True)
        else:
            self.fc1 = None
        if dense_1_units is not None:
            dense_1_units = int(dense_1_units)
            self.fc2 = nn.Linear(in_features=dense_0_units, out_features=dense_1_units, bias=True)
            self.fc3 = nn.Linear(dense_1_units, 2)
        else:
            self.fc2 = None
            self.fc3 = nn.Linear(dense_0_units, 2)
        
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.fc2 is not None:
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.fc3(x)   
        return x
        

#         # Reuse the other layers from the pretrained ResNet50 model
#         self.features = nn.Sequential(*list(efficientnet_b1.children())[:-2])
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         in_features = efficientnet_b1._fc.in_features
#         self.fc1 = nn.Linear(in_features, out_features=128, bias=True)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x, dropout = nn.Dropout(p=0.4)
# ):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = dropout(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = F.relu(self.fc2(x))

#         return x

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
from torchvision import transforms
from torchvision import transforms
from torchvision.transforms.functional import resize, to_tensor


# transform = transforms.Compose([
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
# ])

# aug_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation([-90, 90])
# ])

# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.CenterCrop((224,224)),                                  
#                                 transforms.Normalize(
#                                    mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225],),
# ])

# aug_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(), 
#     transforms.RandomVerticalFlip(), 
#     transforms.RandomRotation([-90, 90])
# ])

# Convert the train, val and test data to PyTorch tensors
# X_train = torch.from_numpy(X_train).float()
# y_train = torch.from_numpy(y_train).long()
# X_val = torch.from_numpy(X_val).float()
# y_val = torch.from_numpy(y_val).long()
# X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).long()

# Define the test dataset
#test_dataset = TensorDataset(X_test, y_test)


# test_dataset = TensorDataset(X_test, y_test)



#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
#model = Effnet().to(device)

# Define the loss function and optimizer
#criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
#optimizer = optim.SGD(model.parameters(), lr=0.004)


# Define the training loop
# def train(model, device, train_loader, criterion, optimizer):
#     model.train()
#     train_loss = 0
#     train_correct = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.permute(0, 3, 1, 2).to(device), target.to(device) # Permute dimensions
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         train_loss += loss.item()
#         pred = output.argmax(dim=1, keepdim=True)
#         train_correct += pred.eq(target.view_as(pred)).sum().item()
#         loss.backward()
#         optimizer.step()

#     train_loss /= len(train_loader.dataset)
#     train_accuracy = 100. * train_correct / len(train_loader.dataset)
#     return train_loss, train_accuracy


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        return focal_loss

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import optuna


# def train_and_evaluate(param, model, trial):
#     f1_scores = []
#     accuracies = []
#     EPOCHS = 5
    
#     criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0, alpha=0.25)
#     optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr=param['learning_rate'])
#     train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=param['batch_size'], shuffle=False)

#     for epoch_num in range(EPOCHS):
#         torch.cuda.empty_cache()
#         model.train()
#         total_acc_train = 0
#         total_loss_train = 0
#         train_correct = 0
#         train_loss = 0

#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.permute(0, 3, 1, 2), target.float() # Permute dimensions
#             #print(target.float)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             #print('loss:', loss)
#             train_loss += loss.item()
#             #print('output:', output)

#             predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
#             #print('predictions:', predictions)

#             target_numpy = target.detach().cpu().numpy()
#             correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))
           
#             #print('correct_predictions:', correct_predictions)

#             batch_accuracy = correct_predictions / target_numpy.shape[0]
#             #print("Number of correct predictions:", correct_predictions)
#             #print("Accuracy of the batch:", batch_accuracy)
#             train_correct += batch_accuracy
#             loss.backward()
#             optimizer.step()

#         # Calculate epoch-level loss and accuracy
#         epoch_loss = train_loss / len(train_loader)
#         epoch_accuracy = train_correct / len(train_loader)

#         print("Epoch Loss:", epoch_num, ': ', epoch_loss)
#         print("Epoch Accuracy:", epoch_num, ': ', epoch_accuracy)
            
       
#         model.eval()
#         val_loss = 0
#         val_correct = 0
#         val_f1_score = 0  # Initialize val_f1_score
#         val_labels = []
#         y_preds = []
        
#         with torch.no_grad():
#             for data, target in val_loader:
#                 data, target = data.permute(0, 3, 1, 2), target.float() # Permute dimensions
#                 #data = data.float()
#                 output = model(data)
#                 val_loss += criterion(output, target)
#                 softmax = nn.Softmax(dim=1)
#                 output = softmax(output)
#                 predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
#                 #print('predictions:', predictions)

#                 target_numpy = target.detach().cpu().numpy()
#                 correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))

#                 #print('correct_predictions:', correct_predictions)

#                 batch_accuracy = correct_predictions / target_numpy.shape[0]
#                 #print("Number of correct predictions:", correct_predictions)
#                 #print("Accuracy of the batch:", batch_accuracy)
#                 val_correct += batch_accuracy
                
#                 # Calculate F1 score
#                 f1 = f1_score(target_numpy.argmax(axis=1), predictions, average='macro')
#                 val_f1_score += f1

#             # Calculate epoch-level loss, accuracy, and F1 score
#             epoch_val_loss = val_loss / len(val_loader)
#             epoch_val_accuracy = val_correct / len(val_loader)
#             epoch_val_f1_score = val_f1_score / len(val_loader)
#             print('val f1-score:',  epoch_num, ': ', epoch_val_f1_score)
#             print('val accuracy:',  epoch_num, ': ', epoch_val_accuracy)
        
#         f1_scores.append(epoch_val_f1_score)        
#         print(f1_scores)
#         trial.report(epoch_val_f1_score, epoch_num)
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()
    
#     final_f1 = max(f1_scores)
#     PATH = '/home/viktoriia.trokhova/model_weights/stack_4.pt'
#     torch.save(model.state_dict(), PATH)

#     return final_f1

# # # Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
# def objective(trial):

#     params = {
#         'learning_rate': trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.01, 0.1]),
#         'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
#         'dense_0_units': trial.suggest_categorical("dense_0_units", [16, 32, 48, 64, 80, 96, 112, 128]),
#         'dense_1_units': trial.suggest_categorical("dense_1_units", [None, 16, 32, 48, 64, 80, 96, 112, 128]),
#         'batch_size': trial.suggest_categorical("batch_size", [16, 32, 64]),
#         'drop_out': trial.suggest_float("dropout", 0.2, 0.8, step=0.1)
#     }

#     model = Effnet(pretrained=True, dense_0_units=params['dense_0_units'],  dense_1_units=params['dense_1_units'], dropout=params['drop_out'])

#     max_f1 = train_and_evaluate(params, model, trial)

#     return max_f1


# EPOCHS = 50
    
# study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=6, reduction_factor=5))
# study.optimize(objective, n_trials=40)
# pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
# complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

# print("Study statistics: ")
# print("  Number of finished trials: ", len(study.trials))
# print("  Number of pruned trials: ", len(pruned_trials))
# print("  Number of complete trials: ", len(complete_trials))

# print("Best trial:")
# trial = study.best_trial
# print("  Value: ", trial.value)

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))



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









learning_rate_best = 0.0001
optimizer_best = 'SGD'
dense_0_units_best = 32
dense_1_units_best = 128
batch_size_best = 32
dropout_best = 0.8
    

model = Effnet(pretrained=True, dense_0_units=dense_0_units_best, dense_1_units=dense_1_units_best, dropout=dropout_best)
                                                             
                                                              

#EPOCHS = 50

import torch
from sklearn.metrics import f1_score
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

def train_and_evaluate(model, device, learning_rate_best, optimizer_best, dense_0_units_best, dense_1_units_best, 
                       batch_size_best):    

    model = model.to(device)
    dataloaders = load_data(batch_size=batch_size_best)
    EPOCHS = 50
    
    # Create optimizer
    optimizer = getattr(optim, optimizer_best)(model.parameters(), lr=learning_rate_best)
    criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0, alpha=0.25)

    # For tracking metrics over epochs
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'f1_score': [], 'val_f1_score': []}
    
    # For early stopping
    best_val_f1 = 0
    best_epoch = 0
    patience = 5
    no_improve = 0


    for epoch_num in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        total_loss_train = 0
        train_correct = 0
        train_f1_score = 0
        train_loss = 0

        # Training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.permute(0, 3, 1, 2), target.float() # Permute dimensions
            #print(target.float)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            #print('loss:', loss)
            train_loss += loss.item()
            #print('output:', output)

            predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
            #print('predictions:', predictions)

            target_numpy = target.detach().cpu().numpy()
            correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))
           
            #print('correct_predictions:', correct_predictions)

            batch_accuracy = correct_predictions / target_numpy.shape[0]
            #print("Number of correct predictions:", correct_predictions)
            #print("Accuracy of the batch:", batch_accuracy)
            train_correct += batch_accuracy
            loss.backward()
            optimizer.step()

        epoch_loss = total_loss_train / len(dataloaders['Train'])
        epoch_accuracy = train_correct / len(dataloaders['Train'])
        epoch_f1_score = train_f1_score / len(dataloaders['Train'])

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        history['f1_score'].append(epoch_f1_score)

        model.eval()
        total_loss_val = 0
        val_correct = 0
        val_f1_score = 0
        val_loss = 0
        val_labels = []
        y_preds = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.permute(0, 3, 1, 2), target.float() # Permute dimensions
                #data = data.float()
                output = model(data)
                val_loss += criterion(output, target)
                softmax = nn.Softmax(dim=1)
                output = softmax(output)
                predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
                #print('predictions:', predictions)

                target_numpy = target.detach().cpu().numpy()
                correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))

                #print('correct_predictions:', correct_predictions)

                batch_accuracy = correct_predictions / target_numpy.shape[0]
                #print("Number of correct predictions:", correct_predictions)
                #print("Accuracy of the batch:", batch_accuracy)
                val_correct += batch_accuracy
                
                # Calculate F1 score
                f1 = f1_score(target_numpy.argmax(axis=1), predictions, average='macro')
                val_f1_score += f1

            
        epoch_val_loss = total_loss_val / len(dataloaders['Val'])
        epoch_val_accuracy = val_correct / len(dataloaders['Val'])
        epoch_val_f1_score = val_f1_score / len(dataloaders['Val'])

        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['val_f1_score'].append(epoch_val_f1_score)
            
        if epoch_val_f1_score > best_val_f1:
            best_val_f1 = epoch_val_f1_score
            best_epoch = epoch_num
            no_improve = 0

            # Save best model
            PATH = '/home/viktoriia.trokhova/model_weights/model_best.pt'
            torch.save(model.state_dict(), PATH)

        else:
            no_improve += 1

        if no_improve > patience:
            print("Early stopping at epoch: ", epoch_num)
            break

    return history, best_val_f1


  
  
history, best_val_f1 = train_and_evaluate(model, learning_rate_best, optimizer_best, dense_0_units_best, dense_1_units_best, batch_size_best)

plot_acc_loss_f1(history, '/home/viktoriia.trokhova/plots/resnet')


