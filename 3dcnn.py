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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input

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
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
y_train = np.array([0] * len(HGG_train) + [1] * len(LGG_train))

X_val = np.array(HGG_val + LGG_val)
#X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], X_val.shape[3], 1))
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


#import torch.nn as nn

# datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=5,
#    #width_shift_range=0.1,
#    #height_shift_range=0.1,
#    #shear_range=0.1,
#     vertical_flip=True,
#     horizontal_flip=True,
#     fill_mode='nearest')

# train_generator = datagen.flow(
#     X_train, y_train, batch_size=32,
#     shuffle=True)

def custom_3d_cnn(input_shape=(240, 240, 155, 1)):
    inputs = Input(shape=input_shape)

    # Layer 1
    conv1 = Conv3D(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)

    # Layer 2
    conv2 = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(bn1)
    mp2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    bn2 = BatchNormalization()(mp2)

    # Layer 3
    conv3 = Conv3D(128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(bn2)
    mp3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    bn3 = BatchNormalization()(mp3)

    # Layer 4
    conv4 = Conv3D(256, kernel_size=(3, 3, 3), padding='same')(bn3)
    mp4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    bn4 = BatchNormalization()(mp4)

    # Layer 5
    conv5 = Conv3D(256, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(bn4)
    bn5 = BatchNormalization()(conv5)

    # Layer 6
    conv6 = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(UpSampling3D(size=(2, 2, 2))(bn5))
    bn6 = BatchNormalization()(conv6)

    # Layer 7
    conv7 = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(UpSampling3D(size=(2, 2, 2))(bn6))
    bn7 = BatchNormalization()(conv7)

    # Layer 8
    conv8 = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(bn7)

    # Layer 9
    flatten = Flatten()(conv8)
    fc1 = Dense(256)(flatten)

    # Layer 10
    fc2 = Dense(256)(fc1)

    # Layer 11
    softmax = Softmax()(fc2)

    # Create model
    model = Model(inputs=inputs, outputs=softmax)

    return model


model = custom_3d_cnn()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])


# datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=5,
#    #width_shift_range=0.1,
#    #height_shift_range=0.1,
#    #shear_range=0.1,
#     vertical_flip=True,
#     horizontal_flip=True,
#     fill_mode='nearest')

# train_generator = datagen.flow(
#     np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3])),
#     y_train, batch_size=32, shuffle=True)


checkpoint = ModelCheckpoint("resnet" + ".h5",monitor='val_auc',save_best_only=True,mode="max",verbose=1)
early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_auc', factor = 0.3, patience = 2, min_delta = 0.001, mode='max',verbose=1)

# Fit the model to the training data for 50 epochs using the best hyperparameters
# model.fit(
#     train_data = (X_train, y_train),
#     epochs=50,
#     validation_data=(X_val, y_val),
#     verbose=1,
#     callbacks=[checkpoint, early_stop, reduce_lr]
# )

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# create an instance of ImageDataGenerator with data augmentation
datagen = ImageDataGenerator(
    rotation_range=5,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #shear_range=0.1,
    #zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# add a new dimension with size 1 to the data
X_train_augmented = X_train[..., np.newaxis]
X_train_augmented = X_train_augmented[0]

# fit the model using the generator
history = model.fit(
    datagen.flow(X_train_augmented, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10
)

