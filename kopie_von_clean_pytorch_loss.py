# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.


# Commented out IPython magic to ensure Python compatibility.
#%matplotlib inline
import torch
#from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch import optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import tensor

import torchvision.models as models
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import seaborn as sns


from sklearn.utils import shuffle
from sklearn import preprocessing as p
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix, roc_curve, auc


from torchmetrics.classification import ROC
from torchmetrics import AUROC

import optuna
from optuna.trial import TrialState

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

#from google.colab import drive
#drive.mount('/content/drive')

# !pip install segmentation_models_pytorch
from segmentation_models_pytorch import losses
dice_loss = losses.DiceLoss('binary')
foc_loss = losses.FocalLoss('binary')


"""### 2. Create PyTorch data generators"""

#transformations
import random

random.seed(1)
torch.manual_seed(1)


import random
import torchvision.transforms as T

# class RandomChoice(torch.nn.Module):
#     def __init__(self, transforms):
#        super().__init__()
#        self.transforms = transforms

#     def __call__(self, imgs):
#         t = random.choice(self.transforms)
#         return [t(img) for img in imgs]

train_transforms = transforms.Compose([torchvision.transforms.ToTensor(),
                                       transforms.Resize((224,224)),                                  
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],),
                                       ])

aug_transform = transforms.Compose([
     transforms.RandomHorizontalFlip(), 
     transforms.RandomVerticalFlip(),
     transforms.RandomRotation(45),
])



val_transforms = transforms.Compose([torchvision.transforms.ToTensor(),
                                      transforms.Resize((224,224)),
                                      torchvision.transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
    ),
                                      ])

class myDataset_train(Dataset):

    def __init__(self, transform=False): 
        #folder containing class folders with images
        self.imgs_path = "/home/viktoriia.trokhova/__MACOSX/data-pytorch/Train/"  
        self.masks_path = "/home/viktoriia.trokhova/__MACOSX/Slices/train_masks/" 
        file_list = glob.glob(self.imgs_path + "*")
        msk_list = glob.glob(self.masks_path + "*")
        print(file_list)
        print(msk_list)
        #msk_list[0], msk_list[1] = msk_list[1], msk_list[0]
        #print(file_list)
        self.images = []
        self.targets = []
        self.masks = []       
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            print(class_path)
            print(class_name)
            #print(glob.glob(class_path + "/.*"))
            #print(sorted(glob.glob(class_path + "/.*")))
            for img_path in sorted(glob.glob(class_path + "/*")):
                #print(img_path)
                self.images.append(img_path)
            for img_path in sorted(glob.glob(class_path + "/*")):
                self.targets.append(class_name)
        for msk_path in msk_list:
            for masks_path in sorted(glob.glob(msk_path + "/*")):
                  self.masks.append(masks_path)
        print(len(self.images))
        print(len(self.targets))
        print(len(self.masks))
        self.images, self.targets, self.masks = shuffle(self.images, self.targets, self.masks, random_state=101)
        print(self.images[-100])
        print(self.targets[-100])
        print(self.masks[-100])
        # print(len(self.images))
        # print(len(self.targets))
        # print(len(self.masks))
        self.class_map = {"HGG" : 0, "LGG": 1}
        self.img_dim = (224, 224)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        class_name = self.targets[idx]
        masks_path = self.masks[idx]
        masks_ID = self.masks[idx]
        masks_path = self.masks[idx]
        print(img_path)
        img = np.load(img_path, allow_pickle=True,fix_imports=True,encoding='latin1')
        msk = np.load(masks_path, allow_pickle=True,fix_imports=True,encoding='latin1')
        reshap_img = img.reshape(-1, 3)
        min_max_scaler = p.MinMaxScaler()
        img_t = min_max_scaler.fit_transform(reshap_img)
        img = img_t.reshape(img.shape)
        reshap_msk = msk.reshape(-1, 3)
        msk_t = min_max_scaler.fit_transform(msk)
        msk = msk_t.reshape(msk.shape)
        img_float32 = np.float32(img)
        img_color = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
        img_tensor = train_transforms(img_color)
        state = torch.get_rng_state()
        img_tensor = aug_transform(img_tensor)
        msk_float32 = np.float32(msk)
        msk_color = cv2.cvtColor(msk_float32, cv2.COLOR_GRAY2RGB)
        msk_tensor = train_transforms(msk_color)
        torch.set_rng_state(state)
        msk_tensor = aug_transform(msk_tensor)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
    
        return img_tensor, class_id, msk_tensor

class myDataset_val(Dataset):

    def __init__(self, transform=None): 
        #folder containing class folders with images
        self.imgs_path = "/home/viktoriia.trokhova/__MACOSX/Val/"
        self.masks_path = "/home/viktoriia.trokhova/__MACOSX/Slices/val_masks/"
        file_list = glob.glob(self.imgs_path + "*")
        msk_list = glob.glob(self.masks_path + "*")
        print(file_list)
        print(msk_list)
        #msk_list[0], msk_list[1] = msk_list[1], msk_list[0]
        self.images = []
        self.targets = []
        self.masks = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in sorted(glob.glob(class_path + "/*")):
                self.images.append(img_path)
            for img_path in sorted(glob.glob(class_path + "/*")):
                self.targets.append(class_name)
        for msk_path in msk_list:
            for masks_path in sorted(glob.glob(msk_path + "/*")):
                  self.masks.append(masks_path)
        self.images, self.targets, self.masks = shuffle(self.images, self.targets, self.masks, random_state=101)
        print(len(self.images))
        print(len(self.targets))
        print(len(self.masks))
        print(self.images[-100])
        print(self.targets[-100])
        print(self.masks[-100])
        # print(len(self.images))
        # print(len(self.targets))
        # print(len(self.masks))
        self.class_map = {"HGG" : 0, "LGG": 1}
        self.img_dim = (224, 224)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        class_name = self.targets[idx]
        masks_path = self.masks[idx]
        print(img_path)
        img = np.load(img_path)
        msk = np.load(masks_path)
        reshap_img = img.reshape(-1, 3)
        min_max_scaler = p.MinMaxScaler()
        img_t = min_max_scaler.fit_transform(reshap_img)
        img = img_t.reshape(img.shape)
        reshap_msk = msk.reshape(-1, 3)
        msk_t = min_max_scaler.fit_transform(msk)
        msk = msk_t.reshape(msk.shape)
        img_float32 = np.float32(img)
        img_color = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
        img_tensor = val_transforms(img_color)
        msk_float32 = np.float32(msk)
        msk_color = cv2.cvtColor(msk_float32, cv2.COLOR_GRAY2RGB)
        msk_tensor = val_transforms(msk_color)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
    
        return img_tensor, class_id, msk_tensor

class myDataset_test(Dataset):

    def __init__(self, transform=None): 
        #folder containing class folders with images
        self.imgs_path = "/home/viktoriia.trokhova/__MACOSX/data-pytorch/Test/"
        self.masks_path = "/home/viktoriia.trokhova/__MACOSX/Slices/test_masks/"
        file_list = glob.glob(self.imgs_path + "*")
        msk_list = glob.glob(self.masks_path + "*")
        #msk_list[0], msk_list[1] = msk_list[1], msk_list[0]
        self.images = []
        self.targets = []
        self.masks = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in sorted(glob.glob(class_path + "/.*")):
                self.images.append(img_path)
            for img_path in sorted(glob.glob(class_path + "/.*")):
                self.targets.append(class_name)
        for msk_path in msk_list:
            for masks_path in sorted(glob.glob(msk_path + "/.*")):
                  self.masks.append(masks_path)
        self.images, self.targets, self.masks = shuffle(self.images, self.targets, self.masks, random_state=101)
        print(self.images[-100])
        print(self.targets[-100])
        print(self.masks[-100])
        print(len(self.images))
        print(len(self.targets))
        print(len(self.masks))
        self.class_map = {"HGG" : 0, "LGG": 1}
        self.img_dim = (224, 224)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        class_name = self.targets[idx]
        masks_path = self.masks[idx]
        print(img_path)
        img = np.load(img_path, allow_pickle=True, fix_imports=True, encoding='latin1')
        msk = np.load(masks_path, allow_pickle=True, fix_imports=True, encoding='latin1')
        min_max_scaler = p.MinMaxScaler()
        img = min_max_scaler.fit_transform(img)
        msk = min_max_scaler.fit_transform(msk)
        img_float32 = np.float32(img)
        img_color = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
        img_tensor = val_transforms(img_color)
        msk_float32 = np.float32(msk)
        msk_color = cv2.cvtColor(msk_float32, cv2.COLOR_GRAY2RGB)
        msk_tensor = val_transforms(msk_color)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
    
        return img_tensor, class_id, msk_tensor

image_datasets = {
    'Train': 
    myDataset_train(),
    'Val': 
    myDataset_val(transform = val_transforms)
}

def load_data(batch_size):

    dataloaders = {
        'Train':
        torch.utils.data.DataLoader(myDataset_train(),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0), 
        'Val':
        torch.utils.data.DataLoader(myDataset_val(transform = val_transforms),
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=0)  
    }

    return dataloaders


#unnormalize images
'''def imshow(image):
    npimg = image.numpy()
    npimg = np.transpose(npimg, (1,2,0))
    #npimg = np.clip(npimg, 0, 1)
    npimg = ((npimg * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])
    plt.rcParams["figure.figsize"] = [10.00, 5.0]
    plt.rcParams["figure.autolayout"] = True
    plt.imshow(npimg)
    return npimg

dataloaders = load_data(batch_size=16)

# get images 
images, labels, masks = next(iter(dataloaders['Train']))
# create grid of images
img_grid = torchvision.utils.make_grid(images)
#msk_grid = torchvision.utils.make_grid(masks)
# get and show the unnormalized images
img_grid = imshow(img_grid)
#msk_grid = show_img(msk_grid)

# dataloaders = load_data(batch_size=8)

# # get images 
# images, labels, masks = next(iter(dataloaders['Train'])) 
# create grid of images
#img_grid = torchvision.utils.make_grid(images)
msk_grid = torchvision.utils.make_grid(masks)
# get and show the unnormalized images
#img_grid = show_img(img_grid)
msk_grid = imshow(msk_grid)
# write to tensorboard
#writer.add_image('training images', img_grid)

dataloaders = load_data(batch_size=8)

# get images 
images, labels, masks = next(iter(dataloaders['Val']))
# create grid of images
img_grid = torchvision.utils.make_grid(images)
#msk_grid = torchvision.utils.make_grid(masks)
# get and show the unnormalized images
img_grid = imshow(img_grid)
#msk_grid = show_img(msk_grid)

# dataloaders = load_data(batch_size=8)

# # get images 
# images, labels, masks = next(iter(dataloaders['Train'])) 
# create grid of images
#img_grid = torchvision.utils.make_grid(images)
msk_grid = torchvision.utils.make_grid(masks)
# get and show the unnormalized images
#img_grid = show_img(img_grid)
msk_grid = imshow(msk_grid)
# write to tensorboard
#writer.add_image('training images', img_grid)'''

"""### 3. Create the network"""

class MyCustomResnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.ModuleList(resnet50.children())[:-2]
        self.features = nn.Sequential(*self.features)
        in_features = resnet50.fc.in_features
        self.last_pooling_operation = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(2048, 2)




    def forward(self, input_imgs, targets=None, masks=None, batch_size = None, xe_criterion=nn.CrossEntropyLoss(), l1_criterion=nn.L1Loss(), dropout=None):
        images_feats = self.features(input_imgs)
        output = self.last_pooling_operation(images_feats)
        output = output.view(input_imgs.size(0), -1)
        #images_outputs = self.fc1(output)
        output = dropout(output)
        images_outputs = F.relu(self.fc2(output))


        # # compute gcam for images
        orig_gradcam_mask = compute_gradcam(images_outputs, images_feats, targets)

        # #upsample gradcam to (224, 224, 3)
        gcam_losses = 0.0

        for i in range(batch_size):
            #print(orig_gradcam_mask[i].shape)
            img_grad = orig_gradcam_mask[i].unsqueeze(0).permute(1, 2, 0)
            img_grad_1 = img_grad.cpu()
            img_grad_2 = img_grad_1.detach().numpy()
            img_grad_3 = cv2.resize(img_grad_2, (224,224), cv2.INTER_LINEAR)
            img_grad_4 = cv2.cvtColor(img_grad_3, cv2.COLOR_GRAY2RGB)
            img_grad_5 = torch.from_numpy(img_grad_4)
            img_grad_6 = img_grad_5.to(device)
            #img_grad_6 = torch.nn.ReLU(inplace=True)(img_grad_6)


            #masks to same dimension
            masks_per = masks[i].permute(1, 2, 0)
            masks_per = cv2.normalize(masks_per.cpu().numpy(), None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            img_grad_6 = cv2.normalize(img_grad_6.cpu().numpy(), None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            masks_per[np.mean(masks_per, axis=-1)<0.2] = 0
            masks_per[np.mean(masks_per, axis=-1)>=0.2] = 1

            gcam_loss = foc_loss(torch.from_numpy(img_grad_6), torch.from_numpy(masks_per))
            #print(gcam_loss)
            #gcam_loss = l1_criterion(img_grad_6, masks_per)
            gcam_losses += gcam_loss

            # gcam_loss = l1_criterion(img_grad_6, masks_per)
            # gcam_losses += gcam_loss

            #gcam_losses += gcam_loss.item() * input_imgs.size(0)
        #gcam_losses = gcam_losses/batch_size
        xe_loss = xe_criterion(images_outputs, targets)
        

        return images_outputs, targets, xe_loss, gcam_losses      #return images_outputs

def compute_gradcam(output, feats, target):
    """
    Compute normalized Grad-CAM for the given target using the model output and features
    :param output:
    :param feats:
    :param target:
    :return:
    """
    eps = 1e-8

    target = target.cpu().detach().numpy()
    one_hot = np.zeros((output.shape[0], output.shape[-1]), dtype=np.float32)
    indices_range = np.arange(output.shape[0])
    one_hot[indices_range, target[indices_range]] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot = Variable(output, requires_grad=True)

    # Compute the Grad-CAM for the original image
    one_hot_cuda = torch.sum(one_hot.to(device) * output)
    dy_dz1, = torch.autograd.grad(one_hot_cuda, feats, grad_outputs=torch.ones(one_hot_cuda.size()).to(device),
                                  retain_graph=True, create_graph=True)

    # We compute the dot product of grad and features (Element-wise Grad-CAM) to preserve grad spatial locations
    gcam512_1 = dy_dz1 * feats
    gradcam = gcam512_1.sum(dim=1)
    gradcam = torch.nn.ReLU(inplace=True)(gradcam)
    spatial_sum1 = gradcam.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam = (gradcam / (spatial_sum1 + eps)) + eps


    return gradcam

model = MyCustomResnet50(pretrained=True).to(device)

# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model.parameters()

"""### 4. Train the model"""

train_loss = []
val_loss = []
train_acc = []
val_acc = []
FPR = []
TPR = []
Thresholds = []
best_model_wts = {}

# Train and evaluate the accuracy of neural network with the addition of pruning mechanism
def train_and_evaluate(param, model, trial):

    dataloaders = load_data(batch_size=param['batch_size'])

    #criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

    for epoch_num in range(EPOCHS):
            model.train()
            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label, train_mask in dataloaders['Train']:

                train_label = train_label.long().to(device)
                train_input = train_input.float().to(device)
                train_mask = train_mask.to(device)

                output, targets_, xe_loss_, gcam_losses_ = model(train_input, train_label, train_mask, batch_size = train_input.size(0), dropout=nn.Dropout(param['drop_out']))
                
                batch_loss = xe_loss_.mean() + param['lambda_val'] * gcam_losses_
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            model.eval()
            # with torch.no_grad():
            

            for val_input, val_label, val_mask in dataloaders['Val']:

                val_label = val_label.long().to(device)
                val_input = val_input.float().to(device)
                val_mask = val_mask.to(device)
                

                output, targets_, xe_loss_, gcam_losses_ = model(val_input, val_label, val_mask, batch_size = val_input.size(0), dropout=nn.Dropout(param['drop_out']))

                batch_loss = xe_loss_.mean() + param['lambda_val'] * gcam_losses_
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
            accuracy = total_acc_val/len(image_datasets['Val'])
            print(accuracy)
            # Add prune mechanism
            trial.report(accuracy, epoch_num)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return accuracy
  
# Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
def objective(trial):

     params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              'batch_size': trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
              'lambda_val': trial.suggest_float("lambda_val", 1e-5, 1e-1),
               'drop_out' : trial.suggest_float("droupout", 0.2, 0.8)
              }
    
     model = MyCustomResnet50(pretrained=True).to(device)

     accuracy = train_and_evaluate(params, model, trial)

     return accuracy
  
  
EPOCHS = 3
    
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), 
        pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=6, reduction_factor=5))
study.optimize(objective, n_trials=3)
