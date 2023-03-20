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
import tensorflow_addons as tfa
from sklearn.metrics import classification_report,confusion_matrix
from PIL import Image
from keras.models import load_model
from skimage.color import rgb2gray
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow_addons import metrics



def save_to_dir(slices, path):
      for i in range(len(slices)):
          img = slices[i]
          save_path = path + '/' + str(i)
          np.save(save_path, img)

def load_from_dir(path):
      file_paths = glob.glob(os.path.join(path, '*.npy'))
      print(file_paths)

      slices_list=[]
      for img in range(len(file_paths)):
          new_img = np.load(file_paths[img])
          slices_list.append(new_img)

      return slices_list

def resize(slices_list, image_size):
    list_new = []

    for img in range(len(slices_list)):
      img_new=cv2.resize(slices_list[img],(image_size,image_size))
      #img_new = tf.expand_dims(img_new, axis=2)
      img_float32 = np.float32(img_new)
      img_color = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
      #img_new = np.stack((img_new,)*3, axis=-1) # add 3rd multichannel dimension, e.g. (224,224) to (224,224,3)
      list_new.append(img_color)

    return list_new

def add_labels(X, y, images_list, label):
  
    for img in images_list:
      X.append(img)
      y.append(label)

    return X, y


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# def create_model(num_conv_layers, num_pooling_layers, num_dense_layers, input_shape, num_classes, dropout_rate):
#     model = Sequential()
    
#     # Add convolutional layers
#     for i in range(num_conv_layers):
#         model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#         model.add(BatchNormalization())
    
#     # Add pooling layers
#     for i in range(num_pooling_layers):
#         model.add(MaxPooling2D(pool_size=(2, 2)))
    
#     # Flatten the output for the dense layers
#     model.add(Flatten())
    
#     # Add dense layers
#     for i in range(num_dense_layers):
#         model.add(Dense(units=128, activation='relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(dropout_rate))
    
#     # Add final output layer
#     model.add(Dense(units=num_classes, activation='softmax'))
    
#     # Compile the model
#     optimizer = Adam(learning_rate=0.001)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
#     return model


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras_tuner import Hyperband
from keras_tuner.engine.hyperparameters import HyperParameters
from sklearn.metrics import accuracy_score

# # Load the CIFAR-10 dataset
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

HGG_list_train = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/train/HGG_t2')
#HGG_list_train_mask = load_from_dir('/home/viktoriia.trokhova/Mask_slices/train/HGG_masks')
LGG_list_train = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/train/LGG_t2')
#LGG_list_train_mask = load_from_dir('/home/viktoriia.trokhova/Mask_slices/train/LGG_masks')

HGG_list_val = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/val/HGG_t2')
#HGG_list_val_masks = load_from_dir('/home/viktoriia.trokhova/Mask_slices/val/HGG_masks')
LGG_list_val = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/val/LGG_t2')
#LGG_list_val_masks = load_from_dir('/home/viktoriia.trokhova/Mask_slices/val/LGG_masks')

HGG_list_test = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/test/HGG_t2')
#HGG_list_test_masks = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/test/HGG_masks/')
LGG_list_test = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/test/LGG_t2')
#LGG_list_test_masks = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/test/LGG_masks/')

HGG_list_new_train = resize(HGG_list_train, image_size = 224)
LGG_list_new_train = resize(LGG_list_train, image_size = 224)

HGG_list_new_val = resize(HGG_list_val, image_size = 224)
LGG_list_new_val = resize(LGG_list_val, image_size = 224)
#HGG_list_masks_new_val = resize(HGG_list_val_masks, image_size = 224)
#LGG_list_masks_new_val = resize(LGG_list_val_masks, image_size = 224)

HGG_list_new_test = resize(HGG_list_test, image_size = 224)
LGG_list_new_test = resize(LGG_list_test, image_size = 224)


X_train = []
y_train = []

X_train, y_train = add_labels(X_train, y_train, HGG_list_new_train, label='HGG')
X_train, y_train = add_labels(X_train, y_train, LGG_list_new_train, label='LGG')

X_val = []
y_val = []
msk_val = []

X_val, y_val = add_labels(X_val, y_val, HGG_list_new_val, label='HGG')
X_val, y_val = add_labels(X_val, y_val, LGG_list_new_val, label='LGG')
#msk_val = HGG_list_masks_new_val + LGG_list_masks_new_val


X_test = []
y_test = []

X_test, y_test = add_labels(X_test, y_test, HGG_list_new_test, label='HGG')
X_test, y_test = add_labels(X_test, y_test, LGG_list_new_test, label='LGG')

X_train = np.array(X_train)
y_train = np.array(y_train)

X_val = np.array(X_val)
y_val = np.array(y_val)
msk_val = np.array(msk_val)

X_test = np.array(X_test)
y_test = np.array(y_test)

labels = ['HGG', 'LGG']
y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_val_new = []
for i in y_val:
    y_val_new.append(labels.index(i))
y_val = y_val_new
y_val = tf.keras.utils.to_categorical(y_val)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

X_train, y_train = shuffle(X_train,y_train, random_state=101)
X_val, y_val = shuffle(X_val,y_val, random_state=101)
X_test, y_test = shuffle(X_test, y_test, random_state=101)

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


def get_scaling_coefficients(phi):
    # phi is a user-defined coefficient that controls the model's depth, width, and resolution
    # phi = 0 corresponds to the smallest EfficientNet model, phi = 1 corresponds to the largest EfficientNet model
    # the scaling coefficients are taken from the EfficientNet paper
    depth_coefficient = 1.2 ** phi
    width_coefficient = 1.1 ** phi
    resolution_coefficient = 1.15 ** phi
    return depth_coefficient, width_coefficient, resolution_coefficient

def build_model(hp):
    model = Sequential()
    
    # Define input shape
    input_shape = X_train.shape[1:]
    
    # Add scaling hyperparameters
    phi = hp.Choice('phi', values=[0, 1, 2, 3, 4, 5, 6])
    alpha = hp.Float('alpha', min_value=0.0, max_value=1.0, default=1.0)
    beta = hp.Float('beta', min_value=0.0, max_value=1.0, default=1.0)
    
    # Get scaling coefficients
    depth_coefficient, width_coefficient, resolution_coefficient = get_scaling_coefficients(phi)
    
    # Add convolutional layers
    for i in range(int(depth_coefficient)):
        filters = int(32 * width_coefficient)
        kernel_size = int(3 * resolution_coefficient)
        model.add(Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(hp.Float('dropout_conv', 0.0, 0.5)))

    # Flatten the output for the dense layers
    model.add(Flatten())

    # Add dense layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(Dense(units=hp.Int('units_dense', 128, 512, 32), activation='relu'))
        model.add(Dropout(hp.Float('dropout_dense', 0.0, 0.5)))

    # Add final output layer
    model.add(Dense(units=2, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

#Define the model-building function
# def build_model(hp):
#     model = Sequential()

#     # Add convolutional layers
#     for i in range(hp.Int('num_conv_layers', 1, 3)):
#         model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(hp.Float('dropout_conv', 0.0, 0.5)))

#     # Flatten the output for the dense layers
#     model.add(Flatten())

#     # Add dense layers
#     for i in range(hp.Int('num_dense_layers', 1, 3)):
#         model.add(Dense(units=hp.Int('units_dense', 128, 512, 32), activation='relu'))
#         model.add(Dropout(hp.Float('dropout_dense', 0.0, 0.5)))

#     # Add final output layer
#     model.add(Dense(units=2, activation='softmax'))

#     # Compile the model
#     optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#     return model


# def build_model(hp):
#     phi = hp.Float('phi', 0.5, 1.5, 0.1)  # Scaling coefficient for network width
#     alpha = hp.Float('alpha', 0.2, 0.8, 0.1)  # Scaling coefficient for resolution
#     rho = hp.Float('rho', 0.2, 0.8, 0.1)  # Scaling coefficient for network depth

#     # Calculate network depth, width, and resolution based on scaling coefficients
#     b = round(alpha ** phi)
#     c = round(rho * (2 ** phi) * 32)
#     r = round(224 * alpha ** phi)

#     model = Sequential()

#     # Add convolutional layers
#     for i in range(b):
#         if i == 0:
#             model.add(Conv2D(filters=c, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=(r, r, 3)))
#         else:
#             model.add(Conv2D(filters=c, kernel_size=(3, 3), strides=(1, 1), padding='same'))
#         model.add(BatchNormalization())
#         model.add(Activation('swish'))

#     # Add pooling and dropout layers
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(hp.Float('dropout_conv', 0.0, 0.5)))

#     # Add dense layers
#     for i in range(hp.Int('num_dense_layers', 1, 3)):
#         model.add(Dense(units=round(hp.Int('units_dense', 128, 512, 32) * phi), activation='swish'))
#         model.add(Dropout(hp.Float('dropout_dense', 0.0, 0.5)))

#     # Add final output layer
#     model.add(Dense(units=2, activation='softmax'))

#     # Compile the model
#     optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'))
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#     return model



# Define the Hyperband search object
tuner = Hyperband(build_model, objective='val_accuracy', max_epochs=40, factor=3, seed=1, hyperparameters=HyperParameters())

# Search for the best hyperparameters
tuner.search(X_train, y_train, validation_data=(X_val, y_val), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

# Print the best hyperparameters found by the tuner
best_hyperparams = tuner.get_best_hyperparameters(1)[0]
print(f'Best hyperparameters: {best_hyperparams}')
