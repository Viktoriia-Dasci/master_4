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


from torchmetrics.classification import ROC
from torchmetrics import AUROC

import optuna
from optuna.trial import TrialState


class MyCustomResnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.ModuleList(resnet50.children())[:-2]
        self.features = nn.Sequential(*self.features)
        in_features = resnet50.fc.in_features
        self.last_pooling_operation = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 2)




    def forward(self, input_imgs, targets=None, masks=None, batch_size = None, xe_criterion=nn.CrossEntropyLoss(), l1_criterion=nn.L1Loss(), dropout=None):
        images_feats = self.features(input_imgs)
        output = self.last_pooling_operation(images_feats)
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


model = MyCustomResnet50(pretrained=True).to(device)

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

    loss = xe_loss_.mean() + 0.575 * gcam_losses_
    
    _, preds = torch.max(outputs, 1)  

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

epoch_loss = running_loss / 1871
epoch_acc = running_corrects.double() / 1871
print('Test loss: {:.4f}, acc: {:.4f}'.format(epoch_loss,
                                            epoch_acc))
