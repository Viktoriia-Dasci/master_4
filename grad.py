import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet
import glob
import os
os.environ["CUDA_MAX_MEM_ALLOC_PERCENT"] = "95"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn import preprocessing as p
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import ROC
from torchmetrics import AUROC
import optuna
from optuna.trial import TrialState
from torchvision.transforms import RandomApply, Lambda
from segmentation_models_pytorch import losses
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import random
from skimage.color import rgb2gray
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score


class myDataset_train(Dataset):

    def __init__(self, transform=False):
        #folder containing class folders with images
        self.imgs_path = "/home/viktoriia.trokhova/T2_new_MRI_slices/train/"  
        self.masks_path = "/home/viktoriia.trokhova/T2_new_Msk_slices/train/" 
        file_list = glob.glob(self.imgs_path + "*")
        msk_list = glob.glob(self.masks_path + "*")
        #msk_list[0], msk_list[1] = msk_list[1], msk_list[0]
        #print(file_list)
        self.images = []
        self.targets = []
        self.masks = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            # print(class_path)
            # print(class_name)
            for img_path in sorted(glob.glob(class_path + "/*.npy")):
                self.images.append(img_path)
            for img_path in sorted(glob.glob(class_path + "/*.npy")):
                self.targets.append(class_name)
        for msk_path in msk_list:
            for masks_path in sorted(glob.glob(msk_path + "/*.npy")):
                  self.masks.append(masks_path)
        self.images, self.targets, self.masks = shuffle(self.images, self.targets, self.masks, random_state=101)
        # print(self.images[-100])
        # print(self.targets[-100])
        # print(self.masks[-100])
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
        reshap_img = img.reshape(-1, 3)
        min_max_scaler = p.MinMaxScaler()
        img_t = min_max_scaler.fit_transform(reshap_img)
        img = img_t.reshape(img.shape)
        reshap_msk = msk.reshape(-1, 3)
        msk_t = min_max_scaler.fit_transform(reshap_msk)
        msk = msk_t.reshape(msk.shape)
        img_float32 = np.float32(img)
        #img_float32 = np.float32(deblurred_img)
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
        self.imgs_path = "/home/viktoriia.trokhova/T2_new_MRI_slices/val/"
        self.masks_path = "/home/viktoriia.trokhova/T2_new_Msk_slices/val/"
        file_list = glob.glob(self.imgs_path + "*")
        msk_list = glob.glob(self.masks_path + "*")
        #msk_list[0], msk_list[1] = msk_list[1], msk_list[0]
        self.images = []
        self.targets = []
        self.masks = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in sorted(glob.glob(class_path + "/*.npy")):
                self.images.append(img_path)
            for img_path in sorted(glob.glob(class_path + "/*.npy")):
                self.targets.append(class_name)
        for msk_path in msk_list:
            for masks_path in sorted(glob.glob(msk_path + "/*.npy")):
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
        reshap_img = img.reshape(-1, 3)
        min_max_scaler = p.MinMaxScaler()
        img_t = min_max_scaler.fit_transform(reshap_img)
        img = img_t.reshape(img.shape)
        reshap_msk = msk.reshape(-1, 3)
        msk_t = min_max_scaler.fit_transform(reshap_msk)
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
        self.imgs_path = "/home/viktoriia.trokhova/T2_new_MRI_slices/test/"
        self.masks_path = "/home/viktoriia.trokhova/T2_new_Msk_slices/test/"
        file_list = glob.glob(self.imgs_path + "*")
        msk_list = glob.glob(self.masks_path + "*")
        #msk_list[0], msk_list[1] = msk_list[1], msk_list[0]
        self.images = []
        self.targets = []
        self.masks = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in sorted(glob.glob(class_path + "/*.npy")):
                self.images.append(img_path)
            for img_path in sorted(glob.glob(class_path + "/*.npy")):
                self.targets.append(class_name)
        for msk_path in msk_list:
            for masks_path in sorted(glob.glob(msk_path + "/*.npy")):
                  self.masks.append(masks_path)
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
        img = np.load(img_path)
        msk = np.load(masks_path)
        reshap_img = img.reshape(-1, 3)
        min_max_scaler = p.MinMaxScaler()
        img_t = min_max_scaler.fit_transform(reshap_img)
        img = img_t.reshape(img.shape)
        reshap_msk = msk.reshape(-1, 3)
        msk_t = min_max_scaler.fit_transform(reshap_msk)
        msk = msk_t.reshape(msk.shape)
        img_float32 = np.float32(img)
        img_color = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
        img_tensor = val_transforms(img_color)
        msk_float32 = np.float32(msk)
        msk_color = cv2.cvtColor(msk_float32, cv2.COLOR_GRAY2RGB)
        msk_tensor = val_transforms(msk_color)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        #class_id_one_hot = F.one_hot(class_id, num_classes=2).float()

        return img_tensor, class_id, msk_tensor

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

def imshow(image):
    npimg = image.numpy()
    npimg = np.transpose(npimg, (1,2,0))
    #npimg = np.clip(npimg, 0, 1)
    npimg = ((npimg * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])
    #npimg = ((npimg * train_std) +  train_mean)
    plt.rcParams["figure.figsize"] = [10.00, 5.0]
    plt.rcParams["figure.autolayout"] = True
    plt.imshow(npimg)
    return npimg


#defining the model
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

    def forward(self, input_imgs, targets=None, masks=None, batch_size=None, xe_criterion=nn.CrossEntropyLoss(weight=class_weights_tensor.to(device)), dropout=None):
        images_feats = self.features(input_imgs)
        output = self.last_pooling_operation(images_feats)
        output = dropout(output)
        output = output.view(input_imgs.size(0), -1)

        output = F.relu(self.fc1(output))

        if self.fc2 is not None:
            output = F.relu(self.fc2(output))

        images_outputs = self.fc_final(output)


        orig_gradcam_mask = compute_gradcam(images_outputs, images_feats, targets)
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

        return images_outputs, targets, xe_loss, gcam_losses


def compute_gradcam(output, feats, target):

    eps = 1e-8
    target = target.cpu().detach().numpy()
    one_hot = np.zeros((output.shape[0], output.shape[-1]), dtype=np.float32)
    indices_range = np.arange(output.shape[0])
    one_hot[indices_range, target[indices_range]] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot = Variable(output, requires_grad=True)
  
    # Computing the Grad-CAM for the original image
    one_hot_cuda = torch.sum(one_hot.to(device) * output)
    dy_dz1, = torch.autograd.grad(one_hot_cuda, feats, grad_outputs=torch.ones(one_hot_cuda.size()).to(device),
                                  retain_graph=True, create_graph=True)
    gcam512_1 = dy_dz1 * feats
    gradcam = gcam512_1.sum(dim=1)
    gradcam = torch.nn.ReLU(inplace=True)(gradcam)
    spatial_sum1 = gradcam.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam = (gradcam / (spatial_sum1 + eps)) + eps
    return gradcam


#Hyperparameter tuning
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

            f1 = f1_score(target, output)
            train_f1_score += f1
            
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
      
        epoch_loss = total_loss_train / len(dataloaders['Train'])
        epoch_accuracy = train_correct / len(dataloaders['Train'])
        epoch_f1_score = train_f1_score / len(dataloaders['Train'])

        print("Epoch Loss:", epoch_num, ': ', epoch_loss)
        print("Epoch Accuracy:", epoch_num, ': ', epoch_accuracy)
        print("Epoch Accuracy:", epoch_num, ': ', epoch_f1_score)    
        
        
        total_acc_val = 0
        total_loss_val = 0
        val_correct = 0
        val_f1_score = 0
        y_preds = []
        val_labels = []
        model.eval()
        
        for val_input, val_label, val_mask in dataloaders['Val']:
            val_label = val_label.float().to(device)
            print(val_label)
            val_input = val_input.to(device)
            val_mask = val_mask.to(device)
            val_targets = torch.argmax(val_label, dim=1)


            output, targets_, xe_loss_, gcam_losses_ = model(val_input, val_targets, val_mask, batch_size=val_input.size(0), dropout=nn.Dropout(param['dropout']))
            
            batch_loss = xe_loss_.mean() + param['lambda_val'] * gcam_losses_
            total_loss_val += batch_loss.item()

            output=F.softmax(output, dim=1)
            print('softmax output:', output)
            
            predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
            print('predictions:', predictions)

            target_numpy = val_label.detach().cpu().numpy()
            correct_predictions = np.sum(predictions == target_numpy.argmax(axis=1))
           
            print('correct_predictions:', correct_predictions)

            batch_accuracy = correct_predictions / target_numpy.shape[0]
            print("Number of correct predictions:", correct_predictions)
            print("Accuracy of the batch:", batch_accuracy)
            val_correct += batch_accuracy
            
            f1 = f1_score(target, output)
            val_f1_score += f1
        
                    
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
        'learning_rate': trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.01, 0.1]),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
        'dense_0_units': trial.suggest_categorical("dense_0_units", [16, 32, 48, 64, 80, 96, 112, 128]),
        'dense_1_units': trial.suggest_categorical("dense_1_units", [None, 16, 32, 48, 64, 80, 96, 112, 128]),
        'batch_size': trial.suggest_categorical("batch_size", [16, 32, 64]),
        'lambda_val': trial.suggest_float("lambda_val", 0.2, 1.0, step=0.1),
        'dropout': trial.suggest_float("dropout", 0.2, 0.8, step=0.1)
    }

    model = MyCustomEfficientNetB0(pretrained=True, dense_0_units=params['dense_0_units'], dense_1_units=params['dense_1_units']).to(device)

    max_f1 = train_and_evaluate(params, model, trial)

    return max_f1

    
def print_best_trial(study, trial):
    print("Finished trial: ", trial.number)
    print("Current best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

def overlay_gradCAM(img, cam3):
    #cam3 = 255 * cam3
    cam3 = cv2.cvtColor(cam3, cv2.COLOR_GRAY2RGB)

    new_img = 0.9 * cam3 + 0.1 * img

    return (new_img * 255.0 / new_img.max()).astype("float32")


#Main code
#Loading data
dataloaders = load_data(batch_size=16)

# get images
images, labels, masks = next(iter(dataloaders['Train']))
# create grid of images
img_grid = torchvision.utils.make_grid(images)
img_grid = imshow(img_grid)
# create grid of masks
msk_grid = torchvision.utils.make_grid(masks)
# get and show the unnormalized masks
msk_grid = imshow(msk_grid)

#Calculating class weights
dataset = image_datasets['Train']
class_counts = Counter(dataset.targets)

dataset = myDataset_train()
class_counts = Counter(dataset.targets)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataset.targets), y=dataset.targets)
print(class_weights)
class_weights_np = np.array(class_weights, dtype=np.float32)
class_weights_tensor = torch.from_numpy(class_weights_np)
if torch.cuda.is_available():
    class_weights_tensor = class_weights_tensor.cuda()

#Focal Loss
foc_loss = losses.FocalLoss('binary')

EPOCHS = 10

#Hyperparameter tuning
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=6, reduction_factor=5))
study.optimize(objective, n_trials=25, callbacks=[print_best_trial])
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

best_params = trial.params

learning_rate_best = best_params["learning_rate"]
optimizer_best = best_params["optimizer"]
dense_0_units_best = best_params["dense_0_units"]
dense_1_units_best = best_params["dense_1_units"]
batch_size_best = best_params["batch_size"]
lambda_val_best = best_params["lambda_val"]
dropout_best = best_params["dropout"]

print(f"Best Params: \n learning_rate: {learning_rate_best}, \n optimizer: {optimizer_best}, \n dense_0_units: {dense_0_units_best}, \n batch_size: {batch_size_best}, \n lambda_val: {lambda_val_best}, \n dropout: {dropout_best}")
    

model = MyCustomEfficientNetB0(pretrained=True, dense_0_units=dense_0_units_best, dense_1_units=dense_1_units_best).to(device)

#Testing
test_dataset = myDataset_test(transform = None)
test_dataloader = torch.utils.data.DataLoader(myDataset_test(transform = None),
                                    batch_size=8,
                                    shuffle=False,
                                    num_workers=0)


model.eval()
running_loss = 0.0
running_corrects = 0.0
all_preds = []
all_labels = []

for inputs, labels, masks in test_dataloader:
    inputs = inputs
    labels = labels
    masks = masks

    outputs, targets_, xe_loss_, gcam_losses_, imgs_feats = model(inputs, labels, masks, batch_size=inputs.size(0), dropout=nn.Dropout(0.8))

    loss = xe_loss_.mean() + 0.2 * gcam_losses_

    _, preds = torch.max(outputs, 1)

    all_preds.extend(preds)
    all_labels.extend(labels)

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

epoch_loss = running_loss / len(test_dataloader.dataset)
epoch_acc = running_corrects.double() / len(test_dataloader.dataset)
f1 = f1_score(all_labels, all_preds, average='macro')

print('Test loss: {:.4f}, acc: {:.4f}, F1 score: {:.4f}'.format(epoch_loss, epoch_acc, f1))


#Computing and plotting Grad-CAM

img_num =100
msk_num = 100
pat_num = 15

img_class = 'HGG_t2'

img_arr = np.load("/content/drive/MyDrive/T2_new_MRI_slices/test/" + img_class + "/"+str(img_num)+ '_' + str(pat_num) + ".npy")

img_msk_class = 'HGG'
img_msk = np.load("/content/drive/MyDrive/T2_new_Msk_slices/test/" + img_msk_class + "_masks/"+str(msk_num)+ '_' + str(pat_num)+".npy")

reshap_img = img_arr.reshape(-1, 3)
min_max_scaler = p.MinMaxScaler()
img_t = min_max_scaler.fit_transform(reshap_img)
img = img_t.reshape(img_arr.shape)

reshap_msk = img_msk.reshape(-1, 3)
msk_t = min_max_scaler.fit_transform(reshap_msk)
msk = msk_t.reshape(img_msk.shape)
msk_float32 = np.float32(msk)

img_float32 = np.float32(img)
img_color = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
img_tensor = val_transforms(img_color)
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.shape)

msk_color = cv2.cvtColor(msk_float32, cv2.COLOR_GRAY2RGB)
msk_tensor = val_transforms(msk_color)
msk_tensor = msk_tensor.unsqueeze(0)
print(msk_tensor.shape)

label = torch.tensor([1])

model.eval()
outputs, targets_, xe_loss_, gcam_losses_, imgs_feats = model(img_tensor.to(device), label.to(device), msk_tensor.to(device), batch_size = img_tensor.size(0), dropout=nn.Dropout(0.38))

cam = compute_gradcam(outputs, imgs_feats, targets_)
print(cam.shape)

img_grad = cam.permute(1, 2, 0)
img_grad_1 = img_grad.cpu()
img_grad_2 = img_grad_1.detach().numpy()
img_grad_3 = cv2.resize(img_grad_2, (224,224), cv2.INTER_LINEAR)

heatmap = overlay_gradCAM(img_rgb_plot,cam3)

img_color = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
img_color = img_color[:224, :224]

msk_rgb_plot = np.clip(msk_color, 0, 255).astype('float32')
msk_color = cv2.cvtColor(msk_float32, cv2.COLOR_GRAY2RGB)
msk_color = msk_color[:224, :224]

heatmap_mask = overlay_gradCAM(msk_color, img_grad_3)

img_rgb_plot = cv2.normalize(img_color, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
img_rgb_plot = img_rgb_plot *255
heatmap = overlay_gradCAM(img_rgb_plot,img_grad_3)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1,4,figsize=(16,32))
ax[0].imshow(img_float32)
ax[0].axis("off")
ax[0].set_title("Original Image " + img_class + ' ' + str(img_num))
ax[1].imshow(img_msk)
ax[1].axis("off")
ax[1].set_title(img_class + " Mask")
ax[2].imshow(rgb2gray(heatmap_mask))
ax[2].axis("off")
ax[2].set_title("GradCAM")
ax[3].imshow(rgb2gray(heatmap))
ax[3].axis("off")
ax[3].set_title("GradCAM-Mask")
plt.show()

#plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.savefig("/home/viktoriia.trokhova/plots/effnet_torch/loss.png")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train')
plt.plot(history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig("/home/viktoriia.trokhova/plots/effnet_torch/accuracy.png")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['f1_score'], label='Train')
plt.plot(history['val_f1_score'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score')
plt.legend()
plt.savefig("/home/viktoriia.trokhova/plots/effnet_torch/F1_Score.png")
