# -*- coding: utf-8 -*-
"""Kopie von 3D_image_classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NQas2Tac8ak40wXN0BrmlB0kAjaKpTh9

# 3D image classification from CT scans

**Author:** [Hasib Zunair](https://twitter.com/hasibzunair)<br>
**Date created:** 2020/09/23<br>
**Last modified:** 2020/09/23<br>
**Description:** Train a 3D convolutional neural network to predict presence of pneumonia.

## Introduction

This example will show the steps needed to build a 3D convolutional neural network (CNN)
to predict the presence of viral pneumonia in computer tomography (CT) scans. 2D CNNs are
commonly used to process RGB images (3 channels). A 3D CNN is simply the 3D
equivalent: it takes as input a 3D volume or a sequence of 2D frames (e.g. slices in a CT scan),
3D CNNs are a powerful model for learning representations for volumetric data.

## References

- [A survey on Deep Learning Advances on Different 3D DataRepresentations](https://arxiv.org/pdf/1808.01462.pdf)
- [VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf)
- [FusionNet: 3D Object Classification Using MultipleData Representations](http://3ddl.cs.princeton.edu/2016/papers/Hegde_Zadeh.pdf)
- [Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction](https://arxiv.org/abs/2007.13224)

## Setup
"""

import os
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# from google.colab import drive
# drive.mount('/content/drive')

"""## Downloading the MosMedData: Chest CT Scans with COVID-19 Related Findings

In this example, we use a subset of the
[MosMedData: Chest CT Scans with COVID-19 Related Findings](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1).
This dataset consists of lung CT scans with COVID-19 related findings, as well as without such findings.

We will be using the associated radiological findings of the CT scans as labels to build
a classifier to predict presence of viral pneumonia.
Hence, the task is a binary classification problem.
"""

# # Download url of normal CT scans.
# url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
# filename = os.path.join(os.getcwd(), "CT-0.zip")
# keras.utils.get_file(filename, url)

# # Download url of abnormal CT scans.
# url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
# filename = os.path.join(os.getcwd(), "CT-23.zip")
# keras.utils.get_file(filename, url)

# # Make a directory to store the data.
# os.makedirs("MosMedData")

# # Unzip data in the newly created directory.
# with zipfile.ZipFile("CT-0.zip", "r") as z_fp:
#     z_fp.extractall("./MosMedData/")

# with zipfile.ZipFile("CT-23.zip", "r") as z_fp:
#     z_fp.extractall("./MosMedData/")

"""## Loading data and preprocessing

The files are provided in Nifti format with the extension .nii. To read the
scans, we use the `nibabel` package.
You can install the package via `pip install nibabel`. CT scans store raw voxel
intensity in Hounsfield units (HU). They range from -1024 to above 2000 in this dataset.
Above 400 are bones with different radiointensity, so this is used as a higher bound. A threshold
between -1000 and 400 is commonly used to normalize CT scans.

To process the data, we do the following:

* We first rotate the volumes by 90 degrees, so the orientation is fixed
* We scale the HU values to be between 0 and 1.
* We resize width, height and depth.

Here we define several helper functions to process the data. These functions
will be used when building training and validation datasets.
"""

import nibabel as nib

from scipy import ndimage


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = np.amin(volume)
    max = np.amax(volume)
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

"""Let's read the paths of the CT scans from the class directories."""

# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
HGG_train_paths = [
    os.path.join(os.getcwd(), "/home/viktoriia.trokhova/T2_split/train/HGG_t2", x)
    for x in os.listdir("/home/viktoriia.trokhova/T2_split/train/HGG_t2")
]
# Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# involvement of lung parenchyma.
LGG_train_paths = [
    os.path.join(os.getcwd(), "/home/viktoriia.trokhova/T2_split/train/LGG_t2", x)
    for x in os.listdir("/home/viktoriia.trokhova/T2_split/train/LGG_t2")
]

HGG_val_paths = [
    os.path.join(os.getcwd(), "/home/viktoriia.trokhova/T2_split/val/HGG_t2", x)
    for x in os.listdir("/home/viktoriia.trokhova/T2_split/val/HGG_t2")
]
# Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# involvement of lung parenchyma.
LGG_val_paths = [
    os.path.join(os.getcwd(), "/home/viktoriia.trokhova/T2_split/val/LGG_t2", x)
    for x in os.listdir("/home/viktoriia.trokhova/T2_split/val/LGG_t2")
]

print("CT scans with normal lung tissue: " + str(len(HGG_train_paths)))
print("CT scans with abnormal lung tissue: " + str(len(LGG_train_paths)))

"""## Build train and validation datasets
Read the scans from the class directories and assign labels. Downsample the scans to have
shape of 128x128x64. Rescale the raw HU values to the range 0 to 1.
Lastly, split the dataset into train and validation subsets.
"""

# Commented out IPython magic to ensure Python compatibility.
# Read and process the scans.
#Each scan is resized across height, width, and depth and rescaled.
HGG_train = np.array([process_scan(path) for path in HGG_train_paths])
LGG_train = np.array([process_scan(path) for path in LGG_train_paths])

HGG_val = np.array([process_scan(path) for path in HGG_val_paths])
LGG_val = np.array([process_scan(path) for path in LGG_val_paths])

#abnormal_scans_test = np.array([process_scan(path) for path in abnormal_scan_paths])
#normal_scans_test = np.array([process_scan(path) for path in normal_scan_paths])

# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
HGG_labels = np.array([1 for _ in range(len(HGG_train))])
LGG_labels = np.array([0 for _ in range(len(LGG_train))])
HGG_labels_val = np.array([1 for _ in range(len(HGG_val))])
LGG_labels_val = np.array([0 for _ in range(len(LGG_val))])

# Split data in the ratio 70-30 for training and validation.
x_train = np.concatenate((HGG_train, LGG_train), axis=0)
y_train = np.concatenate((HGG_labels, LGG_labels), axis=0)
x_val = np.concatenate((HGG_val, LGG_val), axis=0)
y_val = np.concatenate((HGG_labels_val, LGG_labels_val), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
#     % (x_train.shape[0], x_val.shape[0])
)

"""## Data augmentation

The CT scans also augmented by rotating at random angles during training. Since
the data is stored in rank-3 tensors of shape `(samples, height, width, depth)`,
we add a dimension of size 1 at axis 4 to be able to perform 3D convolutions on
the data. The new shape is thus `(samples, height, width, depth, 1)`. There are
different kinds of preprocessing and augmentation techniques out there,
this example shows a few simple ones to get started.
"""

import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

"""While defining the train and validation data loader, the training data is passed through
and augmentation function which randomly rotates volume at different angles. Note that both
training and validation data are already rescaled to have values between 0 and 1.
"""

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 16
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

"""Visualize an augmented CT scan."""

# import matplotlib.pyplot as plt

# data = train_dataset.take(1)
# images, labels = list(data)[0]
# images = images.numpy()
# image = images[0]
# print("Dimension of the CT scan is:", image.shape)
# plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")

# """Since a CT scan has many slices, let's visualize a montage of the slices."""

# def plot_slices(num_rows, num_columns, width, height, data):
#     """Plot a montage of 20 CT slices"""
#     data = np.rot90(np.array(data))
#     data = np.transpose(data)
#     data = np.reshape(data, (num_rows, num_columns, width, height))
#     rows_data, columns_data = data.shape[0], data.shape[1]
#     heights = [slc[0].shape[0] for slc in data]
#     widths = [slc.shape[1] for slc in data[0]]
#     fig_width = 12.0
#     fig_height = fig_width * sum(heights) / sum(widths)
#     f, axarr = plt.subplots(
#         rows_data,
#         columns_data,
#         figsize=(fig_width, fig_height),
#         gridspec_kw={"height_ratios": heights},
#     )
#     for i in range(rows_data):
#         for j in range(columns_data):
#             axarr[i, j].imshow(data[i][j], cmap="gray")
#             axarr[i, j].axis("off")
#     plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
#    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
#plot_slices(4, 10, 128, 128, image[:, :, :40])

"""## Define a 3D convolutional neural network

To make the model easier to understand, we structure it into blocks.
The architecture of the 3D CNN used in this example
is based on [this paper](https://arxiv.org/abs/2007.13224).
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Flatten, Dense, Softmax

# def get_model(width=240, height=240, depth=150):
#     """Build a 3D convolutional neural network model."""

#     inputs = keras.Input((width, height, depth, 1))

#     x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.GlobalAveragePooling3D()(x)
#     x = layers.Dense(units=512, activation="relu")(x)
#     x = layers.Dropout(0.3)(x)

#     outputs = layers.Dense(units=1, activation="sigmoid")(x)

#     # Define the model.
#     model = keras.Model(inputs, outputs, name="3dcnn")
#     return model

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch


import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters


from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

def build_model(hp):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input(shape=(128, 128, 64, 1))

    # Add the specified number of Conv3D layers
    num_conv_layers = hp.Int('num_conv_layers', min_value=3, max_value=10, step=1)
    x = inputs
    for i in range(num_conv_layers):
        x = layers.Conv3D(filters=hp.Int('filters_' + str(i+1), min_value=16, max_value=128, step=16), 
                          kernel_size=hp.Choice('kernel_size_' + str(i+1), values=[3, 5]),
                          padding="same",
                          activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=hp.Int('dense_units', min_value=32, max_value=512, step=32),
                      activation="relu")(x)
    x = layers.Dropout(hp.Float('dropout', min_value=0.2, max_value=0.8, step=0.1))(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")

    # Compile the model.
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import Hyperband

# Define the hyperparameters search space
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    seed=42,
    directory='hyperband_dir',
    project_name='3dcnn_hyperband')


# Run the hyperparameter search
tuner.search(train_dataset, epochs=20, validation_data=validation_dataset)

# Print the best hyperparameters found by the tuner
best_hyperparams = tuner.get_best_hyperparameters(1)[0]
print(f'Best hyperparameters: {best_hyperparams}')


# Build model.
# model = get_model(width=128, height=128, depth=64)
# model.summary()

# def custom_3d_cnn(width=240, height=240, depth=150):
#     inputs = keras.Input((width, height, depth, 1))

#     # Layer 1
#     conv1 = Conv3D(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(inputs)
#     bn1 = BatchNormalization()(conv1)

#     # Layer 2
#     conv2 = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(bn1)
#     mp2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
#     bn2 = BatchNormalization()(mp2)

#     # Layer 3
#     conv3 = Conv3D(128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(bn2)
#     mp3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
#     bn3 = BatchNormalization()(mp3)

#     # Layer 4
#     conv4 = Conv3D(256, kernel_size=(3, 3, 3), padding='same')(bn3)
#     mp4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
#     bn4 = BatchNormalization()(mp4)

#     # Layer 5
#     conv5 = Conv3D(256, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(bn4)
#     bn5 = BatchNormalization()(conv5)

#     # Layer 6
#     conv6 = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(UpSampling3D(size=(2, 2, 2))(bn5))
#     bn6 = BatchNormalization()(conv6)

#     # Layer 7
#     conv7 = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(UpSampling3D(size=(2, 2, 2))(bn6))
#     bn7 = BatchNormalization()(conv7)

#     # Layer 8
#     x = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(bn7)

#     # Layer 9
#     #flatten = Flatten()(conv8)
#     x = layers.GlobalAveragePooling3D()(x)
#     x = layers.Dense(units=512, activation="relu")(x)
#     x = layers.Dropout(0.8)(x)

#     outputs = layers.Dense(units=1, activation="sigmoid")(x)

#     # Define the model.
#     model = keras.Model(inputs, outputs, name="3dcnn")

#     return model

# model = custom_3d_cnn()
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])

"""## Train model"""

# # Compile model.
# initial_learning_rate = 0.0001
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
# )
# model.compile(
#     loss="binary_crossentropy",
#     optimizer=keras.optimizers.SGD(learning_rate=0.0001),
#     metrics=['accuracy', 'AUC'],
# )

# # Define callbacks.
# checkpoint_cb = keras.callbacks.ModelCheckpoint(
#     "3d_image_classification.h5", save_best_only=True
# )
# early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_auc", patience=15)

# # Train the model, doing validation at the end of each epoch
# epochs = 100
# model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=epochs,
#     shuffle=True,
#     verbose=2,
#     callbacks=[checkpoint_cb, early_stopping_cb],
# )

"""It is important to note that the number of samples is very small (only 200) and we don't
specify a random seed. As such, you can expect significant variance in the results. The full dataset
which consists of over 1000 CT scans can be found [here](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1). Using the full
dataset, an accuracy of 83% was achieved. A variability of 6-7% in the classification
performance is observed in both cases.

## Visualizing model performance

Here the model accuracy and loss for the training and the validation sets are plotted.
Since the validation set is class-balanced, accuracy provides an unbiased representation
of the model's performance.
"""

# fig, ax = plt.subplots(1, 2, figsize=(20, 3))
# ax = ax.ravel()

# for i, metric in enumerate(["acc", "loss"]):
#     ax[i].plot(model.history.history[metric])
#     ax[i].plot(model.history.history["val_" + metric])
#     ax[i].set_title("Model {}".format(metric))
#     ax[i].set_xlabel("epochs")
#     ax[i].set_ylabel(metric)
#     ax[i].legend(["train", "val"])

# """## Make predictions on a single CT scan"""

# # Commented out IPython magic to ensure Python compatibility.
# # Load best weights.
# model.load_weights("3d_image_classification.h5")
# prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
# scores = [1 - prediction[0], prediction[0]]

# class_names = ["normal", "abnormal"]
# for score, name in zip(scores, class_names):
#     print(
#         "This model is %.2f percent confident that CT scan is %s"
# #         % ((100 * score), name)
#     )
