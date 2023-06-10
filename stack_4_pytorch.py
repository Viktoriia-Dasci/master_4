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

class_counts = Counter(y_train)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
print(class_weights)

class_weights_np = np.array(class_weights, dtype=np.float32)
class_weights_tensor = torch.from_numpy(class_weights_np)
if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()

# X_test = np.array(HGG_test + LGG_test)
# y_test = np.array([0] * len(HGG_test) + [1] * len(LGG_test))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Convert y_train to one-hot encoded format
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

# Convert y_val to one-hot encoded format
integer_encoded_val = label_encoder.transform(y_val)
y_val = onehot_encoder.transform(integer_encoded_val.reshape(-1, 1))


# Print the shapes of the train and test sets
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_train shape:', X_val.shape)
print('y_train shape:', y_val.shape)
# print('X_test shape:', X_test.shape)
# print('y_test shape:', y_test.shape)
from efficientnet_pytorch import EfficientNet


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
    def __init__(self, pretrained=True, dense_0_units=None, dense_1_units=None):
        super().__init__()

        # Load the pretrained EfficientNet-B1 model
        efficientnet_b1 = EfficientNet.from_pretrained('efficientnet-b0')

        # Replace the first convolutional layer to handle images with shape (240, 240, 4)
        efficientnet_b1._conv_stem = nn.Conv2d(4, 32, kernel_size=3, stride=2, bias=False)
        
        # Reuse the other layers from the pretrained EfficientNet-B1 model
        self.features = efficientnet_b1.extract_features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
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
        
        
    def forward(self, x, dropout=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
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
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).long()
# X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).long()

# Define the test dataset
#test_dataset = TensorDataset(X_test, y_test)

# Define the dataset
from torch.utils.data import TensorDataset
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)


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




import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import optuna

def train_and_evaluate(param, model, trial):
    f1_scores = []
    accuracies = []
    EPOCHS = 5
    
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr=param['learning_rate']).to(device)
    train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=param['batch_size'], shuffle=False)

    for epoch_num in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        total_acc_train = 0
        total_loss_train = 0
        train_correct = 0
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.permute(0, 3, 1, 2).to(device), target.to(device) # Permute dimensions
            optimizer.zero_grad()
            output = model(data, dropout=param['drop_out'])
            output = output.to(device)  # Convert output to the GPU device
            loss = criterion(output, target)
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(train_loader.dataset)
        train_accuracy = 100. * train_correct / len(train_loader.dataset)
        print('train accuracy:', train_accuracy)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_labels = []
        y_preds = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.permute(0, 3, 1, 2).to(device), target.to(device) # Permute dimensions
                output = model(data, dropout=param['drop_out'])
                output = output.to(device)  # Convert output to the GPU device
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_labels.extend(target.cpu().numpy())
                y_preds.extend(output.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100. * val_correct / len(val_loader.dataset)
        print('val accuracy:', val_accuracy)
        
        f1 = f1_score(val_labels, np.round(y_preds))
        f1_scores.append(f1)
        print('val f1-score:', f1)

        trial.report(f1, epoch_num)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    final_f1 = max(f1_scores)
    PATH = '/home/viktoriia.trokhova/model_weights/stack_4.pt'
    torch.save(model.state_dict(), PATH)

    return final_f1

# # Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
def objective(trial):

    params = {
        'learning_rate': trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.01, 0.1]),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
        'dense_0_units': trial.suggest_categorical("dense_0_units", [16, 32, 48, 64, 80, 96, 112, 128]),
        'dense_1_units': trial.suggest_categorical("dense_1_units", [None, 16, 32, 48, 64, 80, 96, 112, 128]),
        'batch_size': trial.suggest_categorical("batch_size", [16, 32, 64]),
        'drop_out': trial.suggest_float("dropout", 0.2, 0.8, step=0.1)
    }

    model = Effnet(pretrained=True, dense_0_units=params['dense_0_units'],  dense_1_units=params['dense_1_units']).to(device)

    max_f1 = train_and_evaluate(params, model, trial)

    return max_f1

  
EPOCHS = 50
    
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=6, reduction_factor=5))
study.optimize(objective, n_trials=40)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))



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
