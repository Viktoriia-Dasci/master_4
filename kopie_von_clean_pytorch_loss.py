# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.


# Commented out IPython magic to ensure Python compatibility.
#%matplotlib inline
import os
os.environ["CUDA_MAX_MEM_ALLOC_PERCENT"] = "95"
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
#from PIL import Image
import cv2
import seaborn as sns


from sklearn.utils import shuffle
from sklearn import preprocessing as p
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix, roc_curve, auc

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score


from torchmetrics.classification import ROC
from torchmetrics import AUROC

import optuna
from optuna.trial import TrialState
from torchvision.transforms import RandomApply, Lambda


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
    RandomApply([transforms.RandomHorizontalFlip()], p=0.5), 
    RandomApply([transforms.RandomVerticalFlip()], p=0.5), 
    RandomApply([transforms.RandomRotation([-90, 90])], p=0.5),
    Lambda(lambda x: x)
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
        self.imgs_path = "/home/viktoriia.trokhova/Mri_slices_new/train/"  
        self.masks_path = "/home/viktoriia.trokhova/Mask_slices/train/" 
        file_list = glob.glob(self.imgs_path + "*")
        msk_list = glob.glob(self.masks_path + "*")
        print(file_list)
        print(msk_list)
        msk_list[0], msk_list[1] = msk_list[1], msk_list[0]
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
        self.class_map = {"HGG_t2" : 0, "LGG_t2": 1}
        self.img_dim = (224, 224)
        
        #Oversampling to make the number of samples in both classes the same
        # Count number of samples in each class
#         class_count = [0] * len(self.class_map)
#         for target in self.targets:
#             class_count[self.class_map[target]] += 1

        # Determine maximum number of samples in any class
#         max_count = max(class_count)

#         # Oversample each class to match max_count
#         for class_id in range(len(class_count)):
#             for i in range(len(self.targets)):
#                 if self.class_map[self.targets[i]] == class_id:
#                     while class_count[class_id] < max_count:
#                         self.images.append(self.images[i])
#                         self.targets.append(self.targets[i])
#                         self.masks.append(self.masks[i])
#                         class_count[class_id] += 1
        
        
        # Oversampling
#         class_count = [0, 0]
#         for target in self.targets:
#             class_count[self.class_map[target]] += 1

#         max_count = max(class_count)
#         for i in range(len(self.targets)):
#             class_id = self.class_map[self.targets[i]]
#             if class_count[class_id] < max_count:
#                 self.images.append(self.images[i])
#                 self.targets.append(self.targets[i])
#                 self.masks.append(self.masks[i])
#                 class_count[class_id] += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        class_name = self.targets[idx]
        masks_path = self.masks[idx]
        masks_ID = self.masks[idx]
        masks_path = self.masks[idx]
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
        self.imgs_path = "/home/viktoriia.trokhova/Mri_slices_new/val/"
        self.masks_path = "/home/viktoriia.trokhova/Mask_slices/val/"
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
        self.class_map = {"HGG_t2" : 0, "LGG_t2": 1}
        self.img_dim = (224, 224)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        class_name = self.targets[idx]
        masks_path = self.masks[idx]
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
        self.imgs_path = "/home/viktoriia.trokhova/Mri_slices_new/test/"
        self.masks_path = "/home/viktoriia.trokhova/Mask_slices/test/"
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
        self.class_map = {"HGG_t2" : 0, "LGG_t2": 1}
        self.img_dim = (224, 224)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        class_name = self.targets[idx]
        masks_path = self.masks[idx]
        img = np.load(img_path)
        msk = np.load(masks_path)
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

from collections import Counter

# # create dataset object
# dataset = myDataset_train()

# # count occurrences of each class
# class_counts = Counter(dataset.targets)

# # print number of images in each class
# for class_name, count in class_counts.items():
#     print(f"{class_name}: {count}")
    
from sklearn.utils.class_weight import compute_class_weight

# create dataset object
dataset = myDataset_train()

# count occurrences of each class
class_counts = Counter(dataset.targets)

def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):
  """
  Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
  Some examples of different formats of class_series and their outputs are:
    - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
    - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
    {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
    - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
    - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
    {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
  The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
  of appareance of the label when the dataset was processed. 
  In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
  Author: Angel Igareta (angel@igareta.com)
  """
  if multi_class:
    # If class is one hot encoded, transform to categorical labels to use compute_class_weight   
    if one_hot_encoded:
      class_series = np.argmax(class_series, axis=1)
  
    # Compute class weights with sklearn method
    class_labels = np.unique(class_series)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
    return dict(zip(class_labels, class_weights))
  else:
    # It is neccessary that the multi-label values are one-hot encoded
    mlb = None
    if not one_hot_encoded:
      mlb = MultiLabelBinarizer()
      class_series = mlb.fit_transform(class_series)

    n_samples = len(class_series)
    n_classes = len(class_series[0])

    # Count each class frequency
    class_count = [0] * n_classes
    for classes in class_series:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1
    
    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    return dict(zip(class_labels, class_weights))

#class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

class_weights = generate_class_weights(dataset.targets, multi_class=False, one_hot_encoded=True)
print(class_weights)



# # calculate class weights
# class_weights = compute_class_weight('balanced', np.unique(dataset.targets), dataset.targets)

# # print number of images in each class and their corresponding class weight
# for class_name, count, weight in zip(class_counts.keys(), class_counts.values(), class_weights):
#     print(f"{class_name}: {count}, class weight: {weight}")
    
    
    
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

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out


class MyCustomResnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.ModuleList(resnet50.children())[:-2]
        self.features = nn.Sequential(*self.features)
        in_features = resnet50.fc.in_features
        self.attention = SelfAttention(2048)
        self.last_pooling_operation = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 2)




    def forward(self, input_imgs, targets=None, masks=None, batch_size = None, xe_criterion=nn.CrossEntropyLoss(weight=torch.Tensor(class_weights)), l1_criterion=nn.L1Loss(), dropout=None):
        images_feats = self.features(input_imgs)
        images_att = self.attention(images_feats)
        output = self.last_pooling_operation(images_att)
        output = output.view(input_imgs.size(0), -1)
        images_outputs = self.fc1(output)
        output = dropout(images_outputs)
        images_outputs = F.relu(self.fc2(output))
        #images_outputs = nn.ReLU(self.fc2(output))


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

# # Freeze all layers
# for param in model.parameters():
#     param.requires_grad = False

    
# # Unfreeze the layers for fine-tuning
# for name, child in model.named_children():
#     if name == 'fc2':
#         for param in child.parameters():
#             param.requires_grad = True

"""### 4. Train the model"""

train_loss = []
val_loss = []
train_acc = []
val_acc = []
FPR = []
TPR = []
Thresholds = []
best_model_wts = {}

#Train and evaluate the accuracy of neural network with the addition of pruning mechanism

# def train_and_evaluate(param, model, trial):
#     accuracies = []
#     dataloaders = load_data(batch_size=param['batch_size'])
#     # Freeze all layers
#     EPOCHS = 3
    
#     #criterion = nn.CrossEntropyLoss()
#     optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

#     for epoch_num in range(EPOCHS):
#             torch.cuda.empty_cache()
#             model.train()
#             total_acc_train = 0
#             total_loss_train = 0

#             for train_input, train_label, train_mask in dataloaders['Train']:

#                 train_label = train_label.long().to(device)
#                 train_input = train_input.float().to(device)
#                 train_mask = train_mask.to(device)

#                 output, targets_, xe_loss_, gcam_losses_ = model(train_input, train_label, train_mask, batch_size = train_input.size(0), dropout=nn.Dropout(param['drop_out']))
                
#                 batch_loss = xe_loss_.mean() + param['lambda_val'] * gcam_losses_
#                 total_loss_train += batch_loss.item()
                
#                 acc = (output.argmax(dim=1) == train_label).sum().item()
#                 total_acc_train += acc

#                 model.zero_grad()
#                 batch_loss.backward()
#                 optimizer.step()
            
#             total_acc_val = 0
#             total_loss_val = 0

#             model.eval()
#             # with torch.no_grad():
            

#             for val_input, val_label, val_mask in dataloaders['Val']:

#                 val_label = val_label.long().to(device)
#                 val_input = val_input.float().to(device)
#                 val_mask = val_mask.to(device)
                

#                 output, targets_, xe_loss_, gcam_losses_ = model(val_input, val_label, val_mask, batch_size = val_input.size(0), dropout=nn.Dropout(param['drop_out']))

#                 batch_loss = xe_loss_.mean() + param['lambda_val'] * gcam_losses_
#                 total_loss_val += batch_loss.item()
                
#                 acc = (output.argmax(dim=1) == val_label).sum().item()
#                 total_acc_val += acc
        
#             accuracy = total_acc_val/len(image_datasets['Val'])
#             accuracies.append(accuracy)
#             print(accuracy)
#             if len(accuracies) >= 3 and accuracy <= 0.5729:
#                 break

#             trial.report(accuracy, epoch_num)
#             if trial.should_prune():
#                 raise optuna.exceptions.TrialPruned()
#     final_accuracy = max(accuracies)
#     PATH = '/home/viktoriia.trokhova/model_weights/model_best.pt'
#     torch.save(model.state_dict(), PATH)
  
#     return final_accuracy

from sklearn.metrics import roc_auc_score

# def train_and_evaluate(param, model, trial):
#     aucs = []
#     accuracies = []
#     dataloaders = load_data(batch_size=param['batch_size'])
#     EPOCHS = 3
    
#     #criterion = nn.CrossEntropyLoss()
#     optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

#     for epoch_num in range(EPOCHS):
#             torch.cuda.empty_cache()
#             model.train()
#             total_acc_train = 0
#             total_loss_train = 0

#             for train_input, train_label, train_mask in dataloaders['Train']:

#                 train_label = train_label.long().to(device)
#                 train_input = train_input.float().to(device)
#                 train_mask = train_mask.to(device)

#                 output, targets_, xe_loss_, gcam_losses_ = model(train_input, train_label, train_mask, batch_size = train_input.size(0), dropout=nn.Dropout(param['drop_out']))
                
#                 batch_loss = xe_loss_.mean() + param['lambda_val'] * gcam_losses_
#                 total_loss_train += batch_loss.item()
                
#                 acc = (output.argmax(dim=1) == train_label).sum().item()
#                 total_acc_train += acc

#                 model.zero_grad()
#                 batch_loss.backward()
#                 optimizer.step()
            
#             total_acc_val = 0
#             total_loss_val = 0
#             y_preds = []
#             val_labels = []
#             model.eval()
#             # with torch.no_grad():
            
#             for val_input, val_label, val_mask in dataloaders['Val']:

#                 val_label = val_label.long().to(device)
#                 val_input = val_input.float().to(device)
#                 val_mask = val_mask.to(device)
                

#                 output, targets_, xe_loss_, gcam_losses_ = model(val_input, val_label, val_mask, batch_size = val_input.size(0), dropout=nn.Dropout(param['drop_out']))

#                 batch_loss = xe_loss_.mean() + param['lambda_val'] * gcam_losses_
#                 total_loss_val += batch_loss.item()
                
#                 y_pred = nn.functional.softmax(output, dim=1)[:, 1].cpu().detach().numpy()
#                 y_preds.extend(y_pred)
                
#                 val_labels.extend(val_label.cpu().detach().numpy())
                
#                 acc = (output.argmax(dim=1) == val_label).sum().item()
#                 total_acc_val += acc
                
        
#             accuracy = total_acc_val/len(image_datasets['Val'])
#             accuracies.append(accuracy)
#             print(accuracy)
#             if len(accuracies) >= 3 and accuracy <= 0.5729:
#                 break
        
#             auc = roc_auc_score(val_labels, y_preds)
#             aucs.append(auc)

#             trial.report(auc, epoch_num)
#             if trial.should_prune():
#                 raise optuna.exceptions.TrialPruned()
                
#     final_auc = max(aucs)
#     PATH = '/home/viktoriia.trokhova/model_weights/model_best.pt'
#     torch.save(model.state_dict(), PATH)
  
#     return final_auc


  
# # Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
# def objective(trial):

#      params = {
#               'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.006),
#               'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
#               'batch_size': trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
#               'lambda_val': trial.suggest_float("lambda_val", 0.2, 1.0),
#                'drop_out' : trial.suggest_float("droupout", 0.2, 0.8)
#               }
    
#      model = MyCustomResnet50(pretrained=True).to(device)

#      max_accuracy = train_and_evaluate(params, model, trial)

#      return max_accuracy
  
  
EPOCHS = 50
    
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

#EPOCHS = 50

# def train_and_evaluate(model):
#     accuracies = []
#     dataloaders = load_data(batch_size=8)
#     # Freeze all layers

#     #criterion = nn.CrossEntropyLoss()


#     optimizer = optim.SGD(model.parameters(), lr=0.004)
#     #optimizer = getattr(optim, "SGD")(model.parameters(), lr=0.004)

#     for epoch_num in range(EPOCHS):
#             torch.cuda.empty_cache()
#             model.train()
#             total_acc_train = 0
#             total_loss_train = 0

#             for train_input, train_label, train_mask in dataloaders['Train']:

#                 train_label = train_label.long().to(device)
#                 train_input = train_input.float().to(device)
#                 train_mask = train_mask.to(device)

#                 output, targets_, xe_loss_, gcam_losses_ = model(train_input, train_label, train_mask, batch_size = train_input.size(0), dropout=nn.Dropout(0.79))
                
#                 batch_loss = xe_loss_.mean() + 0.575 * gcam_losses_
#                 #batch_loss = xe_loss_.mean()
#                 total_loss_train += batch_loss.item()
                
#                 acc = (output.argmax(dim=1) == train_label).sum().item()
#                 total_acc_train += acc

#                 model.zero_grad()
#                 batch_loss.backward()
#                 optimizer.step()
            
#             total_acc_val = 0
#             total_loss_val = 0

#             model.eval()
#             # with torch.no_grad():
            

#             for val_input, val_label, val_mask in dataloaders['Val']:

#                 val_label = val_label.long().to(device)
#                 val_input = val_input.float().to(device)
#                 val_mask = val_mask.to(device)
                

#                 output, targets_, xe_loss_, gcam_losses_ = model(val_input, val_label, val_mask, batch_size = val_input.size(0), dropout=nn.Dropout(0.79))

#                 batch_loss = xe_loss_.mean() + 0.575 * gcam_losses_
#                 #batch_loss = xe_loss_.mean()
#                 total_loss_val += batch_loss.item()
                
#                 acc = (output.argmax(dim=1) == val_label).sum().item()
#                 total_acc_val += acc
        
#             accuracy = total_acc_val/len(image_datasets['Val'])
#             accuracies.append(accuracy)
#             print(accuracy)
#             if len(accuracies) >= 3 and accuracy <= 0.5729:
#                 break

# #             trial.report(accuracy, epoch_num)
# #             if trial.should_prune():
# #                 raise optuna.exceptions.TrialPruned()
#     final_accuracy = max(accuracies)
#     PATH = '/home/viktoriia.trokhova/model_weights/model_best.pt'
#     torch.save(model.state_dict(), PATH)
  
#     return final_accuracy
  
  
# train_and_evaluate(model)


optimizer = optim.SGD(model.parameters(), lr=0.0051)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_auc_values = []
val_auc_values = []

def train_with_early_stopping(model, optimizer, patience, PATH):
    dataloaders = load_data(batch_size=8)
    # define early stopping and lr scheduler
    best_val_auc = 0.0
    early_stopping_counter = 0
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=patience//2, verbose=True)

    for epoch_num in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        total_loss_train = 0
        total_preds_train = []
        total_targets_train = []
        correct_train = 0
        total_train = 0
        train_probs = []

        for train_input, train_label, train_mask in dataloaders['Train']:
            train_label = train_label.long().to(device)
            train_input = train_input.float().to(device)
            train_mask = train_mask.to(device)

            optimizer.zero_grad()
            output, targets_, xe_loss_, gcam_losses_ = model(train_input, train_label, train_mask, batch_size=train_input.size(0), dropout=nn.Dropout(0.38))

            batch_loss = xe_loss_.mean() + 0.202 * gcam_losses_
            total_loss_train += batch_loss.item()

            # calculate accuracy
            _, predicted = torch.max(output.data, 1)
            correct_train += (predicted == train_label).sum().item()
            total_train += train_label.size(0)

            # calculate probabilities
            probs = nn.functional.softmax(output, dim=1)
            train_probs.append(probs.detach().cpu().numpy())
            
            total_targets_train.append(train_label.detach().cpu().numpy())

            batch_loss.backward()
            optimizer.step()

        # calculate train auc
        total_targets_train = np.concatenate(total_targets_train, axis=0)
        train_probs = np.concatenate(train_probs, axis=0)
        train_auc = roc_auc_score(total_targets_train, train_probs[:, 1])
        train_auc_values.append(train_auc)

        train_loss = total_loss_train / len(dataloaders['Train'])
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        print('train_acc:', train_accuracy)
        train_accuracies.append(train_accuracy)

        total_loss_val = 0
        total_targets_val = []
        total_preds_val = []
        correct = 0
        total = 0

        model.eval()

        for val_input, val_label, val_mask in dataloaders['Val']:
            val_label = val_label.long().to(device)
            val_input = val_input.float().to(device)
            val_mask = val_mask.to(device)

            output, targets_, xe_loss_, gcam_losses_ = model(val_input, val_label, val_mask, batch_size=val_input.size(0), dropout=nn.Dropout(0.38))

            batch_loss = xe_loss_.mean() + 0.202 * gcam_losses_
            total_loss_val += batch_loss.item()

            # calculate accuracy
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == val_label).sum().item()
            total += val_label.size(0)

            targets_ = targets_.detach().cpu().numpy()
            preds_ = output[:, 1].detach().cpu().numpy()  # use class 1 probabilities for AUC calculation
            total_targets_val.extend(targets_)
            total_preds_val.extend(preds_)

        val_loss = total_loss_val / len(dataloaders['Val'])
        val_losses.append(val_loss)
        val_accuracy = correct / total
        print(val_accuracy)
        val_accuracies.append(val_accuracy)

        val_auc = roc_auc_score(total_targets_val, total_preds_val)
        print("Epoch: {} - Validation AUC: {:.4f}".format(epoch_num+1, val_auc))
        val_auc_values.append(val_auc)

        # update lr_scheduler
        lr_scheduler.step(val_loss)

        # check if early stopping criteria is met
        if val_auc > best_val_auc:
            early_stopping_counter = 0
            best_val_auc = val_auc
            # save model if val_auc is the best so far
            torch.save(model.state_dict(), PATH)
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping.")
            break
        
        
    

    return best_val_auc

  
# #patience = 15
# #PATH = '/home/viktoriia.trokhova/model_weights/model_effnet.pt'
best_val_auc = train_with_early_stopping(model, optimizer = optim.SGD(model.parameters(), lr=0.0051), patience=20, PATH= '/home/viktoriia.trokhova/model_weights/resnet_pytorch.pt')

        
# plot loss and accuracy for each epoch
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.savefig("/home/viktoriia.trokhova/plots/resnet_torch/loss.png")  # save plot to given path

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train')
plt.plot(val_accuracies, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig("/home/viktoriia.trokhova/plots/resnet_torch/accuracy.png")  # save plot to given path

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_auc_values, label='Train')
plt.plot(val_auc_values, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('AUC')
plt.legend()
plt.savefig("/home/viktoriia.trokhova/plots/resnet_torch/AUC.png")  # save plot to given path
