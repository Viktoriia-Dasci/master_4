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
                                       transforms.CenterCrop((224,224)),                                  
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],),
                                       ])
# aug_transform = transforms.Compose([
#     RandomApply([transforms.RandomHorizontalFlip()], p=0.5), 
#     RandomApply([transforms.RandomVerticalFlip()], p=0.5), 
#     RandomApply([transforms.RandomRotation([-90, 90])], p=0.5),
#     Lambda(lambda x: x)
# ])

aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(), 
    transforms.RandomRotation([-90, 90]),
])

val_transforms = transforms.Compose([torchvision.transforms.ToTensor(),
                                      transforms.CenterCrop((224,224)),
                                      torchvision.transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
    ),
                                      ])
class myDataset_train(Dataset):
    def __init__(self, transform=False): 
        #folder containing class folders with images
        self.imgs_path = "/home/viktoriia.trokhova/T2_new_MRI_slices/val/"  
        self.masks_path = "/home/viktoriia.trokhova/T2_new_Msk_slices/val/" 
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
#         reshap_img = img.reshape(-1, 3)
#         min_max_scaler = p.MinMaxScaler()
#         img_t = min_max_scaler.fit_transform(reshap_img)
#         img = img_t.reshape(img.shape)
#         reshap_msk = msk.reshape(-1, 3)
#         msk_t = min_max_scaler.fit_transform(msk)
#         msk = msk_t.reshape(msk.shape)
        img_color = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
        img_tensor = train_transforms(img_color)
        state = torch.get_rng_state()
        img_tensor = aug_transform(img_tensor)
        msk_color = cv2.cvtColor(np.float32(msk), cv2.COLOR_GRAY2RGB)
        msk_tensor = train_transforms(msk_color)
        torch.set_rng_state(state)
        msk_tensor = aug_transform(msk_tensor)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        class_id_one_hot = F.one_hot(class_id, num_classes=2).float()
    
        return img_tensor, class_id_one_hot, msk_tensor
    
class myDataset_val(Dataset):
    def __init__(self, transform=None): 
        #folder containing class folders with images
        self.imgs_path = "/home/viktoriia.trokhova/T2_new_MRI_slices/test/"
        self.masks_path = "/home/viktoriia.trokhova/T2_new_Msk_slices/test/"
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
#         reshap_img = img.reshape(-1, 3)
#         min_max_scaler = p.MinMaxScaler()
#         img_t = min_max_scaler.fit_transform(reshap_img)
#         img = img_t.reshape(img.shape)
#         reshap_msk = msk.reshape(-1, 3)
#         msk_t = min_max_scaler.fit_transform(msk)
#         msk = msk_t.reshape(msk.shape)
        img_float32 = np.float32(img)
        img_color = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
        img_tensor = val_transforms(img_color)
        msk_float32 = np.float32(msk)
        msk_color = cv2.cvtColor(msk_float32, cv2.COLOR_GRAY2RGB)
        msk_tensor = val_transforms(msk_color)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        class_id_one_hot = F.one_hot(class_id, num_classes=2).float()
    
        return img_tensor, class_id_one_hot, msk_tensor
    
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
  
def f1_score(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=-1)
    y_true = torch.argmax(y_true, dim=-1)
    tp = torch.sum((y_true == 1) & (y_pred == 1)).float()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).float()
    fn = torch.sum((y_true == 1) & (y_pred == 0)).float()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1  
  
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
# create dataset object
dataset = image_datasets['Train']
# count occurrences of each class
class_counts = Counter(dataset.targets)
# # create dataset object
# dataset = myDataset_train()
# # count occurrences of each class
# class_counts = Counter(dataset.targets)
# # print number of images in each class
# for class_name, count in class_counts.items():
#     print(f"{class_name}: {count}")
# create dataset object
dataset = myDataset_train()
# count occurrences of each class
class_counts = Counter(dataset.targets)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataset.targets), y=dataset.targets)
print(class_weights)
class_weights_np = np.array(class_weights, dtype=np.float32)
class_weights_tensor = torch.from_numpy(class_weights_np)
if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()
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
class MyCustomEfficientNetB0(nn.Module):
    def __init__(self, pretrained=True, dense_0_units=None, dense_1_units=None):
        super().__init__()
        
        efficientnet_b0 = EfficientNet.from_pretrained('efficientnet-b0')
        self.features = efficientnet_b0.extract_features
        in_features = efficientnet_b0._fc.in_features
        self.last_pooling_operation = nn.AdaptiveAvgPool2d((1, 1))

        if dense_0_units is not None:
            dense_0_units = int(dense_0_units)
            self.fc1 = nn.Linear(in_features=1280, out_features=dense_0_units, bias=True)
        
        if dense_1_units is not None:
            dense_1_units = int(dense_1_units)
            self.fc2 = nn.Linear(in_features=dense_0_units, out_features=dense_1_units, bias=True)
            self.fc_final = nn.Linear(dense_1_units, 2)
        else:
            self.fc2 = None
            self.fc_final = nn.Linear(dense_0_units, 2)
            
    def forward(self, input_imgs, targets=None, masks=None, batch_size=None, xe_criterion=nn.CrossEntropyLoss(weight=class_weights_tensor), dropout=None):
        images_feats = self.features(input_imgs)
        output = self.last_pooling_operation(images_feats)
        output = dropout(output)
        output = output.view(input_imgs.size(0), -1)
        
        output = F.relu(self.fc1(output))
        
        if self.fc2 is not None:
            output = F.relu(self.fc2(output))
        
        images_outputs = self.fc_final(output)
        

        
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
            img_grad_6[np.mean(img_grad_6, axis=-1)<0.5] = 0
            img_grad_6[np.mean(img_grad_6, axis=-1)>=0.5] = 1
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
#model = MyCustomEfficientNetB1(pretrained=True).to(device)
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
import numpy as np
from sklearn.metrics import f1_score


def train_and_evaluate(param, model, trial):
    f1_scores = []
    accuracies = []
    dataloaders = load_data(batch_size=param['batch_size'])
    EPOCHS = 5
    
    #criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])
    for epoch_num in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        total_acc_train = 0
        total_loss_train = 0
        train_correct = 0
        train_loss = 0
        train_f1_score = 0
        for train_input, train_label, train_mask in dataloaders['Train']:
            optimizer.zero_grad()
            train_label = train_label.float().to(device)
            #print(train_label)
            train_input = train_input.to(device)
            train_mask = train_mask.to(device)
            targets = torch.argmax(train_label, dim=1)
            targets = targets.to(device)
            output, targets_, xe_loss_, gcam_losses_ = model(train_input, targets, train_mask, batch_size=train_input.size(0), dropout=nn.Dropout(param['dropout']))
           
            
            batch_loss = xe_loss_.mean() + param['lambda_val'] * gcam_losses_
            total_loss_train += batch_loss.item()
        
            
            #print('output:', output)
            output=F.softmax(output, dim=1)
            #print('softmax output:', output)
            
            predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
            #print('predictions:', predictions)
            target_numpy = train_label.detach().cpu().numpy()
            correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))
           
            #print('correct_predictions:', correct_predictions)
            batch_accuracy = correct_predictions / target_numpy.shape[0]
            #print("Number of correct predictions:", correct_predictions)
            #print("Accuracy of the batch:", batch_accuracy)
            train_correct += batch_accuracy
            
            f1 = f1_score(target_numpy.argmax(axis=1), predictions, average='macro')
            train_f1_score += f1
            
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
      
        epoch_loss = total_loss_train / len(dataloaders['Train'])
        epoch_accuracy = train_correct / len(dataloaders['Train'])
        epoch_f1score = train_f1_score / len(dataloaders['Train'])
        print("Epoch Loss:", epoch_num, ': ', epoch_loss)
        print("Epoch Accuracy:", epoch_num, ': ', epoch_accuracy)
        print("Epoch F1-Score:", epoch_num,  ': ', epoch_f1score)    
        
        
        total_acc_val = 0
        total_loss_val = 0
        val_correct = 0
        val_f1_score = 0
        y_preds = []
        val_labels = []
        model.eval()
        
        for val_input, val_label, val_mask in dataloaders['Val']:
            val_label = val_label.float().to(device)
            #print(val_label)
            val_input = val_input.to(device)
            val_mask = val_mask.to(device)
            val_targets = torch.argmax(val_label, dim=1)
            val_targets = val_targets.to(device)
            output, targets_, xe_loss_, gcam_losses_ = model(val_input, val_targets, val_mask, batch_size=val_input.size(0), dropout=nn.Dropout(param['dropout']))
            
            batch_loss = xe_loss_.mean() + param['lambda_val'] * gcam_losses_
            total_loss_val += batch_loss.item()
            output=F.softmax(output, dim=1)
            #print('softmax output:', output)
            
            predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
            #print('predictions:', predictions)
            target_numpy = val_label.detach().cpu().numpy()
            correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))
           
            #print('correct_predictions:', correct_predictions)
            batch_accuracy = correct_predictions / target_numpy.shape[0]
            #print("Number of correct predictions:", correct_predictions)
            #print("Accuracy of the batch:", batch_accuracy)
            val_correct += batch_accuracy
            
            f1 = f1_score(target_numpy.argmax(axis=1), predictions, average='macro')
            val_f1_score += f1

        #epoch_val_loss = val_loss / len(dataloaders['Val'])
        epoch_val_loss = total_loss_val / len(dataloaders['Val'])
        epoch_val_accuracy = val_correct / len(dataloaders['Val'])
        epoch_val_f1_score = val_f1_score / len(dataloaders['Val'])
        print('val f1-score:',  epoch_num, ': ', epoch_val_f1_score)
        print('val accuracy:',  epoch_num, ': ', epoch_val_accuracy)
        
        
        f1_scores.append(epoch_val_f1_score)
        print('val f1-score:', epoch_val_f1_score)
        trial.report(epoch_val_f1_score, epoch_num)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    final_f1 = max(f1_scores)
    PATH = '/home/viktoriia.trokhova/model_weights/model_best.pt'
    torch.save(model.state_dict(), PATH)
    return final_f1
  
# # Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
def objective(trial):
    params = {
        'learning_rate': trial.suggest_categorical("learning_rate", [0.00001,0.0001, 0.001, 0.01, 0.1]),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
        'dense_0_units': trial.suggest_categorical("dense_0_units", [16, 32, 48, 64, 80, 96, 112, 128]),
        'dense_1_units': trial.suggest_categorical("dense_1_units", [None, 16, 32, 48, 64, 80, 96, 112, 128]),
        'batch_size': trial.suggest_categorical("batch_size", [16, 32, 64]),
        'lambda_val': trial.suggest_float("lambda_val", 0.01, 1.0),
        'dropout': trial.suggest_float("dropout", 0.2, 0.8, step=0.1)
    }
    model = MyCustomEfficientNetB0(pretrained=True, dense_0_units=params['dense_0_units'], dense_1_units=params['dense_1_units']).to(device)
    max_f1 = train_and_evaluate(params, model, trial)
    return max_f1
  
EPOCHS = 50
    
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=6, reduction_factor=5))
study.optimize(objective, n_trials=50)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))
print("Best trial:")
trial = study.best_trial

def print_best_trial(study, trial):
    print("Finished trial: ", trial.number)
    print("Current best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=6, reduction_factor=5))
study.optimize(objective, n_trials=25, callbacks=[print_best_trial])
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

print("Best trial:")
trial = study.best_trial

best_params = trial.params

learning_rate_best = best_params["learning_rate"]
optimizer_best = best_params["optimizer"]
dense_0_units_best = best_params["dense_0_units"]
#dense_1_units_best = best_params["dense_1_units"]
batch_size_best = best_params["batch_size"]
lambda_val_best = best_params["lambda_val"]
dropout_best = best_params["dropout"]

print(f"Best Params: \n learning_rate: {learning_rate_best}, \n optimizer: {optimizer_best}, \n dense_0_units: {dense_0_units_best}, \n batch_size: {batch_size_best}, \n lambda_val: {lambda_val_best}, \n dropout: {dropout_best}")
    
learning_rate_best = 0.0001
optimizer_best = 'Adam'
dense_0_units_best = 64
#dense_1_units_best = best_params["dense_1_units"]
batch_size_best = 64
lambda_val_best = 0.04
dropout_best = 0.4
print(f"Best Params: \n learning_rate: {learning_rate_best}, \n optimizer: {optimizer_best}, \n dense_0_units: {dense_0_units_best}, \n batch_size: {batch_size_best}, \n lambda_val: {lambda_val_best}, \n dropout: {dropout_best}")

                                                             
                                                              

#EPOCHS = 50

import torch
from sklearn.metrics import f1_score
from torch import nn, optim
import torch.nn.functional as F
import numpy as np




def train_and_evaluate(model, device, learning_rate_best, optimizer_best, dense_0_units_best,
                       batch_size_best, lambda_val_best, dropout_best):    

                        
    dataloaders = load_data(batch_size=batch_size_best)
    EPOCHS = 50
    
    # Create optimizer
    optimizer = getattr(optim, optimizer_best)(model.parameters(), lr=learning_rate_best)

    # For tracking metrics over epochs
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'f1_score': [], 'val_f1_score': []}
    
    # For early stopping
    best_val_f1 = 0
    best_epoch = 0
    patience = 10
    no_improve = 0

    for epoch_num in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        total_loss_train = 0
        train_correct = 0
        train_f1_score = 0

        # Training loop
        for train_input, train_label, train_mask in dataloaders['Train']:
            optimizer.zero_grad()
            train_label = train_label.float().to(device)
            train_input = train_input.to(device)
            train_mask = train_mask.to(device)
            targets = torch.argmax(train_label, dim=1)

            output, targets_, xe_loss_, gcam_losses_ = model(train_input, targets, train_mask, 
                                                             batch_size=train_input.size(0), dropout=nn.Dropout(dropout_best))
            
            batch_loss = xe_loss_.mean() + lambda_val_best * gcam_losses_
            total_loss_train += batch_loss.item()

            output = F.softmax(output, dim=1)
            
            predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
            target_numpy = train_label.detach().cpu().numpy()
            correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))
           
            batch_accuracy = correct_predictions / target_numpy.shape[0]
            train_correct += batch_accuracy
            
            f1 = f1_score(target_numpy.argmax(axis=1), predictions, average='macro')
            train_f1_score += f1
            
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
        epoch_loss = total_loss_train / len(dataloaders['Train'])
        epoch_accuracy = train_correct / len(dataloaders['Train'])
        epoch_f1_score = train_f1_score / len(dataloaders['Train'])
        print("Epoch Loss:", epoch_num, ': ', epoch_loss)
        print("Epoch Accuracy:", epoch_num, ': ', epoch_accuracy)
        print("Epoch F1-Score:", epoch_num,  ': ', epoch_f1_score)    

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        history['f1_score'].append(epoch_f1_score)
            
        total_loss_val = 0
        val_correct = 0
        val_f1_score = 0
        model.eval()
        
        for val_input, val_label, val_mask in dataloaders['Val']:
            val_label = val_label.float().to(device) 
            val_input = val_input.to(device)
            val_mask = val_mask.to(device)
            val_targets = torch.argmax(val_label, dim=1)

            output, targets_, xe_loss_, gcam_losses_ = model(val_input, val_targets, val_mask, 
                                                             batch_size=val_input.size(0), dropout=nn.Dropout(dropout_best))
            
            batch_loss = xe_loss_.mean() + lambda_val_best * gcam_losses_
            total_loss_val += batch_loss.item()

            output = F.softmax(output, dim=1)
            
            predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
            target_numpy = val_label.detach().cpu().numpy()
            correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))
           
            batch_accuracy = correct_predictions / target_numpy.shape[0]
            val_correct += batch_accuracy
            
            f1 = f1_score(target_numpy.argmax(axis=1), predictions, average='macro')
            val_f1_score += f1
            
        epoch_val_loss = total_loss_val / len(dataloaders['Val'])
        epoch_val_accuracy = val_correct / len(dataloaders['Val'])
        epoch_val_f1_score = val_f1_score / len(dataloaders['Val'])
        print('val f1-score:',  epoch_num, ': ', epoch_val_f1_score)
        print('val accuracy:',  epoch_num, ': ', epoch_val_accuracy)

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


model = MyCustomEfficientNetB0(pretrained=True, dense_0_units=128).to(device)  
  
history, best_val_f1 = train_and_evaluate(model, device, learning_rate_best, optimizer_best, dense_0_units_best, batch_size_best, lambda_val_best, dropout_best)



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
#optimizer = optim.SGD(model.parameters(), lr=0.0051)
# train_losses = []
# val_losses = []
# train_accuracies = []
# val_accuracies = []
# train_auc_values = []
# val_auc_values = []
# def train_with_early_stopping(model, optimizer, patience, PATH):
#     dataloaders = load_data(batch_size=8)
#     # define early stopping and lr scheduler
#     best_val_auc = 0.0
#     early_stopping_counter = 0
#     lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=patience//2, verbose=True)
#     for epoch_num in range(EPOCHS):
#         torch.cuda.empty_cache()
#         model.train()
#         total_loss_train = 0
#         total_preds_train = []
#         total_targets_train = []
#         correct_train = 0
#         total_train = 0
#         train_probs = []
#         for train_input, train_label, train_mask in dataloaders['Train']:
#             train_label = train_label.long().to(device)
#             train_input = train_input.float().to(device)
#             train_mask = train_mask.to(device)
#             optimizer.zero_grad()
#             output, targets_, xe_loss_, gcam_losses_ = model(train_input, train_label, train_mask, batch_size=train_input.size(0), dropout=nn.Dropout(0.38))
#             batch_loss = xe_loss_.mean() + 0.202 * gcam_losses_
#             total_loss_train += batch_loss.item()
#             # calculate accuracy
#             _, predicted = torch.max(output.data, 1)
#             correct_train += (predicted == train_label).sum().item()
#             total_train += train_label.size(0)
#             # calculate probabilities
#             probs = nn.functional.softmax(output, dim=1)
#             train_probs.append(probs.detach().cpu().numpy())
            
#             total_targets_train.append(train_label.detach().cpu().numpy())
#             batch_loss.backward()
#             optimizer.step()
#         # calculate train auc
#         total_targets_train = np.concatenate(total_targets_train, axis=0)
#         train_probs = np.concatenate(train_probs, axis=0)
#         train_auc = roc_auc_score(total_targets_train, train_probs[:, 1])
#         train_auc_values.append(train_auc)
#         train_loss = total_loss_train / len(dataloaders['Train'])
#         train_losses.append(train_loss)
#         train_accuracy = correct_train / total_train
#         print('train_acc:', train_accuracy)
#         train_accuracies.append(train_accuracy)
#         total_loss_val = 0
#         total_targets_val = []
#         total_preds_val = []
#         correct = 0
#         total = 0
#         model.eval()
#         for val_input, val_label, val_mask in dataloaders['Val']:
#             val_label = val_label.long().to(device)
#             val_input = val_input.float().to(device)
#             val_mask = val_mask.to(device)
#             output, targets_, xe_loss_, gcam_losses_ = model(val_input, val_label, val_mask, batch_size=val_input.size(0), dropout=nn.Dropout(0.38))
#             batch_loss = xe_loss_.mean() + 0.202 * gcam_losses_
#             total_loss_val += batch_loss.item()
#             # calculate accuracy
#             _, predicted = torch.max(output.data, 1)
#             correct += (predicted == val_label).sum().item()
#             total += val_label.size(0)
#             targets_ = targets_.detach().cpu().numpy()
#             preds_ = output[:, 1].detach().cpu().numpy()  # use class 1 probabilities for AUC calculation
#             total_targets_val.extend(targets_)
#             total_preds_val.extend(preds_)
#         val_loss = total_loss_val / len(dataloaders['Val'])
#         val_losses.append(val_loss)
#         val_accuracy = correct / total
#         print(val_accuracy)
#         val_accuracies.append(val_accuracy)
#         val_auc = roc_auc_score(total_targets_val, total_preds_val)
#         print("Epoch: {} - Validation AUC: {:.4f}".format(epoch_num+1, val_auc))
#         val_auc_values.append(val_auc)
#         # update lr_scheduler
#         lr_scheduler.step(val_loss)
#         # check if early stopping criteria is met
#         if val_auc > best_val_auc:
#             early_stopping_counter = 0
#             best_val_auc = val_auc
#             # save model if val_auc is the best so far
#             torch.save(model.state_dict(), PATH)
#         else:
#             early_stopping_counter += 1
#         if early_stopping_counter >= patience:
#             print("Early stopping.")
#             break
        
        
    
#     return best_val_auc
  
# # #patience = 15
# # #PATH = '/home/viktoriia.trokhova/model_weights/model_effnet.pt'
# best_val_auc = train_with_early_stopping(model, optimizer = optim.SGD(model.parameters(), lr=0.0051), patience=20, PATH= '/home/viktoriia.trokhova/model_weights/resnet_noscale_pytorch.pt')


# plot loss and accuracy for each epoch
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.savefig("/home/viktoriia.trokhova/plots/resnet_torch/loss.png")  # save plot to given path
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train')
plt.plot(history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig("/home/viktoriia.trokhova/plots/resnet_torch/accuracy.png")  # save plot to given path
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['f1_score'], label='Train')
plt.plot(history['val_f1_score'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score')
plt.legend()
plt.savefig("/home/viktoriia.trokhova/plots/resnet_torch/F1_Score.png")  # save plot to given path
