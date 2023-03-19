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
from efficientnet_pytorch import EfficientNet

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


from torchmetrics.classification import ROC
from torchmetrics import AUROC

import optuna
from optuna.trial import TrialState

from segmentation_models_pytorch import losses
dice_loss = losses.DiceLoss('binary')
foc_loss = losses.FocalLoss('binary')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([torchvision.transforms.ToTensor(),
                                      transforms.Resize((224,224)),
                                      torchvision.transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
    ),
                                      ])

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
            for img_path in sorted(glob.glob(class_path + "/*")):
                self.images.append(img_path)
            for img_path in sorted(glob.glob(class_path + "/*")):
                self.targets.append(class_name)
        for msk_path in msk_list:
            for masks_path in sorted(glob.glob(msk_path + "/*")):
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


class MyCustomEfficientNetB1(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        efficientnet_b1 = EfficientNet.from_pretrained('efficientnet-b1').to(device)
        self.features = efficientnet_b1.extract_features
        in_features = efficientnet_b1._fc.in_features
        self.attention = SelfAttention(in_features)
        self.last_pooling_operation = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 2)


    def forward(self, input_imgs, targets=None, masks=None, batch_size = None, xe_criterion=nn.CrossEntropyLoss(), l1_criterion=nn.L1Loss(), dropout=None):
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
            img_grad_3 = cv2.resize(img_grad_2, (240,240), cv2.INTER_LINEAR)
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

model = MyCustomEfficientNetB1(pretrained=True).to(device)


model.load_state_dict(torch.load('/home/viktoriia.trokhova/model_weights/model_best.pt'), strict=False)

test_dataset = myDataset_test(transform = None)
test_dataloader = torch.utils.data.DataLoader(myDataset_test(transform = None),
                                    batch_size=32,
                                    shuffle=False,
                                    num_workers=0)

model.eval()
running_loss = 0.0
running_corrects = 0.0
for inputs, labels, masks in test_dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    masks = masks.to(device)
  
    outputs, targets_, xe_loss_, gcam_losses_ = model(inputs, labels, masks, batch_size = inputs.size(0), dropout=nn.Dropout(0.79))

    loss = xe_loss_.mean() + 0.575 * gcam_losses_.mean()
    #loss = xe_loss_.mean()
   
    _, preds = torch.max(outputs, 1)  

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

epoch_loss = running_loss / 1611
epoch_acc = running_corrects.double() / 1611
print('Test loss: {:.4f}, acc: {:.4f}'.format(epoch_loss,
                                            epoch_acc))
