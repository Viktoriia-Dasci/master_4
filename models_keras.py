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
    
HGG_list_train = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/train/HGG_t2')
HGG_list_train_mask = load_from_dir('/home/viktoriia.trokhova/Mask_slices/train/HGG_masks')
LGG_list_train = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/train/LGG_t2')
LGG_list_train_mask = load_from_dir('/home/viktoriia.trokhova/Mask_slices/train/LGG_masks')

HGG_list_val = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/val/HGG_t2')
HGG_list_val_masks = load_from_dir('/home/viktoriia.trokhova/Mask_slices/val/HGG_masks')
LGG_list_val = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/val/LGG_t2')
LGG_list_val_masks = load_from_dir('/home/viktoriia.trokhova/Mask_slices/val/LGG_masks')

HGG_list_test = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/test/HGG_t2')
HGG_list_test_masks = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/test/HGG_masks/')
LGG_list_test = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/test/LGG_t2')
LGG_list_test_masks = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/test/LGG_masks/')


from sklearn.preprocessing import MinMaxScaler

def normalize_images(images_list):
    scaler = MinMaxScaler()
    normalized_images = []
    for image in images_list:
        image_norm = scaler.fit_transform(image)
        normalized_images.append(image_norm)
    return normalized_images

HGG_list_train_normalized = normalize_images(HGG_list_train)
LGG_list_train_normalized = normalize_images(LGG_list_train)
HGG_list_val_normalized = normalize_images(HGG_list_val)
LGG_list_val_normalized = normalize_images(LGG_list_val)
HGG_list_test_normalized = normalize_images(HGG_list_test)
LGG_list_test_normalized = normalize_images(LGG_list_test)


HGG_list_new_train = resize(HGG_list_train_normalized, image_size = 224)
LGG_list_new_train = resize(LGG_list_train_normalized, image_size = 224)

HGG_list_new_val = resize(HGG_list_val_normalized, image_size = 224)
LGG_list_new_val = resize(LGG_list_val_normalized, image_size = 224)
HGG_list_masks_new_val = resize(HGG_list_val_masks, image_size = 224)
LGG_list_masks_new_val = resize(LGG_list_val_masks, image_size = 224)

HGG_list_new_test = resize(HGG_list_test_normalized, image_size = 224)
LGG_list_new_test = resize(LGG_list_test_normalized, image_size = 224)


import numpy as np
from sklearn.utils import resample

# # Determine maximum number of samples in any class
# max_count = np.max([len(HGG_list_new_train), len(LGG_list_new_train)])

# # Oversample each class to match max_count
# HGG_list_new_train_resampled = resample(HGG_list_new_train, 
#                                          replace=True,
#                                          n_samples=max_count,
#                                          random_state=42)
# LGG_list_new_train_resampled = resample(LGG_list_new_train, 
#                                          replace=True,
#                                          n_samples=max_count,
#                                          random_state=42)

# print(len(HGG_list_new_train_resampled))
# print(len(LGG_list_new_train_resampled))

X_train = []
y_train = []

X_train, y_train = add_labels(X_train, y_train, HGG_list_new_train, label='HGG')
X_train, y_train = add_labels(X_train, y_train, LGG_list_new_train, label='LGG')

# X_train, y_train = add_labels(X_train, y_train, HGG_list_new_train, label='HGG')
# X_train, y_train = add_labels(X_train, y_train, LGG_list_new_train, label='LGG')

X_val = []
y_val = []
msk_val = []

X_val, y_val = add_labels(X_val, y_val, HGG_list_new_val, label='HGG')
X_val, y_val = add_labels(X_val, y_val, LGG_list_new_val, label='LGG')
msk_val = HGG_list_masks_new_val + LGG_list_masks_new_val


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


import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer


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

class_weights = generate_class_weights(y_train, multi_class=False, one_hot_encoded=True)
print(class_weights)


datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='nearest')


train_generator = datagen.flow(
    X_train, y_train,
    shuffle=True)

from sklearn.metrics import f1_score
import numpy as np




# def model_train(model_name, image_size = 224):
#     #model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))
#     model = model_name.output
#     model = tf.keras.layers.GlobalAveragePooling2D()(model)
#     model = tf.keras.layers.Dropout(rate=0.5)(model)
#     model = tf.keras.layers.Dense(128, activation='relu')(model)
#     model = tf.keras.layers.Dense(2,activation='softmax')(model)
#     model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
#     adam = tf.keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(loss='categorical_crossentropy', optimizer = adam, metrics= ['accuracy', 'AUC'])
#     #callbacks
#     tensorboard = TensorBoard(log_dir = 'logs')
#     checkpoint = ModelCheckpoint(str(model_name) + ".h5",monitor='val_auc',save_best_only=True,mode="max",verbose=1)
#     early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=5, verbose=1, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor = 'val_auc', factor = 0.3, patience = 2, min_delta = 0.001, mode='max',verbose=1)
#     #fitting the model
#     history = model.fit(train_generator, validation_data=(X_val, y_val), steps_per_epoch=len(train_generator), epochs=30, verbose=1,
#                    callbacks=[tensorboard, checkpoint, early_stop, reduce_lr])
  
#     return history

import keras_tuner
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

# def model_effnet(hp):
#     model_name = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
#     model = model_name.output
#     model = tf.keras.layers.GlobalAveragePooling2D()(model)
#     model = tf.keras.layers.Dense(128, activation='relu')(model)
#     model = tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.8, step=0.1))(model)
#     model = tf.keras.layers.Dense(2,activation='softmax')(model)
#     model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
    
#     # Define optimizer and batch size
#     optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])
#     learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.01)
#     batch_size = hp.Choice('batch_size', values=[8, 16, 32, 64])
    
#     #Set optimizer parameters based on user's selection
#     if optimizer == 'adam':
#         optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     else:
#         optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
#     # Compile the model with the optimizer and metrics
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', 'AUC'])
    
#     return model

# def model_resnet(hp):
#     model_name = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2)
#     model = model_name.output
#     model = tf.keras.layers.GlobalAveragePooling2D()(model)
#     model = tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.9, step=0.1))(model)
#     for i in range(hp.Int('num_layers', min_value=1, max_value=4)):
#        model = tf.keras.layers.Dense(hp.Int(f'dense_{i}_units', min_value=16, max_value=128, step=16), activation='relu')(model)
#     model = tf.keras.layers.Dense(2,activation='softmax')(model)
#     model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
#     sgd = SGD(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.01, 0.1]))
#     model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics= ['accuracy', 'AUC'])
#     return model

# def model_effnet(hp):
#     model_name = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
#     model = model_name.output
#     model = tf.keras.layers.GlobalAveragePooling2D()(model)
#     model = tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.8, step=0.1))(model)
#     for i in range(hp.Int('num_layers', min_value=1, max_value=3)):
#         model = tf.keras.layers.Dense(hp.Int(f'dense_{i}_units', min_value=16, max_value=128, step=16), activation='relu')(model)
#     model = tf.keras.layers.Dense(2,activation='softmax')(model)
#     model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
    
#     # Define optimizer and batch size
#     optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])
#     learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.001, 0.01, 0.1])
#     batch_size = hp.Choice('batch_size', values=[16, 32, 64])
    
#     #Set optimizer parameters based on user's selection
#     if optimizer == 'adam':
#         optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     else:
#         optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
#     # Compile the model with the optimizer and metrics
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', 'AUC'])
    
#     return model

# def model_inception(hp):
#     model_name = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2)
#     model = model_name.output
#     model = tf.keras.layers.GlobalAveragePooling2D()(model)
#     model = tf.keras.layers.Dense(128, activation='relu')(model)
#     model = tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.8, step=0.1))(model)
#     model = tf.keras.layers.Dense(2,activation='softmax')(model)
#     model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
    
#     # Define optimizer and batch size
#     optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])
#     learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.001, 0.01, 0.1])
#     batch_size = hp.Choice('batch_size', values=[16, 32, 64])
    
#     #Set optimizer parameters based on user's selection
#     if optimizer == 'adam':
#         optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     else:
#         optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
#     # Compile the model with the optimizer and metrics
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', 'AUC'])
    
#     return model

# def model_train(model_name, image_size = 224):
#     #model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))
#     model = model_name.output
#     model = tf.keras.layers.GlobalAveragePooling2D()(model)
#     model = tf.keras.layers.Dense(128, activation='relu')(model)
#     model = tf.keras.layers.Dropout(rate=0.4)(model)
#     model = tf.keras.layers.Dense(2,activation='softmax')(model)
#     model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
#     adam = tf.keras.optimizers.Adam(learning_rate=0.0009)
#     model.compile(loss='categorical_crossentropy', optimizer = adam, metrics= ['accuracy', 'AUC'])
#     #callbacks
#     #tensorboard = TensorBoard(log_dir = 'logs')
#     checkpoint = ModelCheckpoint("/home/viktoriia.trokhova/model_weights/effnet_weights" + ".h5",monitor='val_auc',save_best_only=True,mode="max",verbose=1)
#     early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=5, verbose=1, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor = 'val_auc', factor = 0.3, patience = 2, min_delta = 0.001, mode='max',verbose=1)
#     #fitting the model
#     history = model.fit(train_generator, validation_data=(X_val, y_val), epochs=50, batch_size=64, verbose=1,
#                    callbacks=[checkpoint, early_stop, reduce_lr], class_weight=class_weights)
  
#     return history


#def model_train(model_name, image_size = 224, learning_rate = 0.1, dropout=0.5):
    #model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))
#    model = model_name.output
#    model = tf.keras.layers.GlobalAveragePooling2D()(model)
#    model = tf.keras.layers.Dense(128, activation='relu')(model)
#    model = tf.keras.layers.Dropout(rate=dropout)(model)
#    model = tf.keras.layers.Dense(2,activation='softmax')(model)
#    model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
    #adam = tf.keras.optimizers.Adam(learning_rate=0.0009)
#    sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate)
#    model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics= ['accuracy', 'AUC'])
    #callbacks
    #tensorboard = TensorBoard(log_dir = 'logs')
#    checkpoint = ModelCheckpoint("/home/viktoriia.trokhova/model_weights/resnet_weights" + ".h5",monitor='val_auc',save_best_only=True,mode="max",verbose=1)
#    early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=10, verbose=1, restore_best_weights=True)
#    reduce_lr = ReduceLROnPlateau(monitor = 'val_auc', factor = 0.3, patience = 2, min_delta = 0.001, mode='max',verbose=1)
    #fitting the model
#    history = model.fit(train_generator, validation_data=(X_val, y_val), epochs=50, batch_size=64, verbose=1,
 #                  callbacks=[checkpoint, early_stop, reduce_lr], class_weight=class_weights)


#    return history


def model_train(model_name, image_size = 224, learning_rate = 0.1, dropout=0.5):
    #model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))
    model = model_name.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dense(128, activation='relu')(model)
    model = tf.keras.layers.Dropout(rate=dropout)(model)
    model = tf.keras.layers.Dense(2,activation='softmax')(model)
    model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
    #adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics= ['accuracy', 'AUC'])
    #callbacks
    #tensorboard = TensorBoard(log_dir = 'logs')
    checkpoint = ModelCheckpoint("/home/viktoriia.trokhova/model_weights/resnet_weughts_new" + ".h5",monitor='val_auc',save_best_only=True,mode="max",verbose=1)
    early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_auc', factor = 0.3, patience = 2, min_delta = 0.001, mode='max',verbose=1)
    #fitting the model
    history = model.fit(train_generator, validation_data=(X_val, y_val), epochs=50, batch_size=64, verbose=1,
                   callbacks=[checkpoint, early_stop, reduce_lr], class_weight=class_weights)
     
    return history



# # Define hp before calling tuner.search()
# # hp = HyperParameters()

# # # Add hyperparameters to search space
# # hp.Choice('optimizer', values=['adam', 'sgd'])
# # hp.Choice('learning_rate', values=[0.0001, 0.001, 0.01, 0.1])
# # hp.Choice('batch_size', values=[16, 32, 64])

# tuner = Hyperband(
#     model_inception,
#     objective=keras_tuner.Objective("val_auc", direction="max"),
#     #overwrite=True,
#     max_epochs=50,
#     factor=3,
#     hyperband_iterations=10
# )

# tuner.search(train_generator,
#              validation_data=(X_val, y_val),
#              steps_per_epoch=len(train_generator),
#              epochs=50,
#              verbose=1
#              )

#Print the best hyperparameters found by the tuner
#best_hyperparams = tuner.get_best_hyperparameters(1)[0]
#print(f'Best hyperparameters: {best_hyperparams}')

# tuner = Hyperband(
#     model_densenet,
#     objective=keras_tuner.Objective("val_auc", direction="max"),
#     max_epochs=50,
#     #overwrite=True,
#     factor=3,
#     hyperband_iterations=10
# )

# tuner.search(train_generator,
#              validation_data=(X_val, y_val),
#              steps_per_epoch=len(train_generator),
#              epochs=50,
#              verbose=1
#              )


# Print the best hyperparameters found by the tuner
#best_hyperparams = tuner.get_best_hyperparameters(1)[0]
#print(f'Best hyperparameters: {best_hyperparams}')



#Get the best model found by the tuner
#best_model = tuner.get_best_models(1)[0]

# checkpoint = ModelCheckpoint("/home/viktoriia.trokhova/model_weights/inception_keras" + ".h5",monitor='val_auc',save_best_only=True,mode="max",verbose=1)
# early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=10, verbose=1, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor = 'val_auc', factor = 0.3, patience = 2, min_delta = 0.001, mode='max',verbose=1)

# #Fit the model to the training data for 50 epochs using the best hyperparameters
# history_inception = best_model.fit(
#     train_generator,
#     epochs=50,
#     validation_data=(X_val, y_val),
#     verbose=1,
#     callbacks=[checkpoint, early_stop, reduce_lr]
# )

#history_resnet_weights = model_train(model_name = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), image_size = 224, learning_rate = 0.1, dropout=0.5)

#history_densenet_weights = model_train(model_name = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), image_size = 224, learning_rate = 0.1, dropout=0.5)

history_resnet_oversample = model_train(model_name = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), image_size = 224, learning_rate = 0.1, dropout=0.5)



def plot_acc_loss_auc(model_history, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'loss.png'))
    plt.close()

    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'accuracy.png'))
    plt.close()

    auc = model_history.history['auc']
    val_auc = model_history.history['val_auc']

    plt.plot(epochs, auc, 'y', label='Training AUC')
    plt.plot(epochs, val_auc, 'r', label='Validation AUC')
    plt.title('Training and validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'auc.png'))
    plt.close()


plot_acc_loss_auc(history_resnet_oversample,  '/home/viktoriia.trokhova/plots/resnet')
   
#history_effnet = model_train(model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3)))

#plot_acc_loss(history_effnet, '/home/viktoriia.trokhova/plots/effnet')

# history_resnet50 = model_train(model_name = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2))

# plot_acc_loss(history_resnet50, '/home/viktoriia.trokhova/plots/resnet')

#history_inceptionv3 = model_train(model_name = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2))

#plot_acc_loss(history_inceptionv3, '/home/viktoriia.trokhova/plots/inception')

#history_densenet121 = model_train(model_name = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2))

#plot_acc_loss(history_densenet121, '/home/viktoriia.trokhova/plots/densenet')

# history_vit = model_train(model_name = tfa.image.ViTModel(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2))

# plot_acc_loss(history_vit)

# history_deit = model_train(model_name = tfa.image.DeiT(include_top=False, pretrained=True, input_shape=(224,224,3), classes=2))

# plot_acc_loss(history_deit)

# history_Regnety = model_train(model_name = RegNetY(weights='imagenet', input_shape=(224, 224, 3), include_top=False))

# plot_acc_loss(history_Regnety)

# history_NFNet = model_train(model_name = tfa.image.NFNet(include_top=False, pretrained=True, input_shape=(224,224,3), classes=2))

# plot_acc_loss(history_NFNet)



# colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
# colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
# colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

# my_model_eff = load_model('/home/viktoriia.trokhova/master_4/effnetDA.h5')
# print('efficientnet')
# pred_eff = my_model_eff.predict(X_test)
# pred_ready_eff = np.argmax(pred_eff,axis=1)
# y_test_new_eff = np.argmax(y_test,axis=1)

# print(classification_report(y_test_new_eff,pred_ready_eff))

# # fig,ax=plt.subplots(1,1,figsize=(14,7))
# # sns.heatmap(confusion_matrix(y_test_new_eff,pred_ready_eff),ax=ax,xticklabels=labels,yticklabels=labels,annot=True, fmt='g',
# #            cmap=colors_green[::-1],alpha=0.7,linewidths=2,linecolor=colors_dark[3])
# # fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
# #              fontname='monospace',color=colors_dark[1],y=0.92,x=0.28,alpha=0.8)

# # plt.show()

# test_loss, test_acc = my_model_eff.evaluate(X_test, y_test, verbose=2)
# #acc = 0.81

# print(f' Test accuracy: {test_acc:.3f} \n Test loss {test_loss:.3f}')

# my_model_res = load_model('/home/viktoriia.trokhova/master_4/resnetDA.h5')
# print('resnet')
# pred_res = my_model_res.predict(X_test)
# pred_ready_res = np.argmax(pred_res,axis=1)
# y_test_new_res = np.argmax(y_test,axis=1)

# print(classification_report(y_test_new_res,pred_ready_res))

# # fig,ax=plt.subplots(1,1,figsize=(14,7))
# # sns.heatmap(confusion_matrix(y_test_new_res,pred_ready_res),ax=ax,xticklabels=labels,yticklabels=labels,annot=True, fmt='g',
# #            cmap=colors_green[::-1],alpha=0.7,linewidths=2,linecolor=colors_dark[3])
# # fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
# #              fontname='monospace',color=colors_dark[1],y=0.92,x=0.28,alpha=0.8)

# # plt.show()

# test_loss, test_acc = my_model_res.evaluate(X_test, y_test, verbose=2)

# print(f' Test accuracy: {test_acc:.3f} \n Test loss {test_loss:.3f}')

# my_model_inception = load_model('/home/viktoriia.trokhova/master_4/inceptionDA.h5')
# print('inception')
# pred_incep = my_model_inception.predict(X_test)
# pred_ready_incep = np.argmax(pred_incep,axis=1)
# y_test_new_incep = np.argmax(y_test,axis=1)

# print(classification_report(y_test_new_incep,pred_ready_incep))

# # fig,ax=plt.subplots(1,1,figsize=(14,7))
# # sns.heatmap(confusion_matrix(y_test_new_incep,pred_ready_incep),ax=ax,xticklabels=labels,yticklabels=labels,annot=True, fmt='g',
# #            cmap=colors_green[::-1],alpha=0.7,linewidths=2,linecolor=colors_dark[3])
# # fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
# #              fontname='monospace',color=colors_dark[1],y=0.92,x=0.28,alpha=0.8)

# # plt.show()

# test_loss, test_acc = my_model_inception.evaluate(X_test, y_test, verbose=2)

# print(f' Test accuracy: {test_acc:.3f} \n Test loss {test_loss:.3f}')

# my_model_densenet = load_model('/home/viktoriia.trokhova/master_4/densenetDA.h5')
# print('densenet')
# pred_dense = my_model_densenet.predict(X_test)
# pred_ready_dense = np.argmax(pred_dense,axis=1)
# y_test_new_dense = np.argmax(y_test,axis=1)


# print(classification_report(y_test_new_dense,pred_ready_dense))

# # fig,ax=plt.subplots(1,1,figsize=(14,7))
# # sns.heatmap(confusion_matrix(y_test_new_dense,pred_ready_dense),ax=ax,xticklabels=labels,yticklabels=labels,annot=True, fmt='g',
# #            cmap=colors_green[::-1],alpha=0.7,linewidths=2,linecolor=colors_dark[3])
# # fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
# #              fontname='monospace',color=colors_dark[1],y=0.92,x=0.28,alpha=0.8)

# # plt.show()

# test_loss, test_acc = my_model_densenet.evaluate(X_test, y_test, verbose=2)

# print(f' Test accuracy: {test_acc:.3f} \n Test loss {test_loss:.3f}')

# # inception: <keras.engine.functional.Functional object at 0x7f07681711b0>.h5
# # densenet: <keras.engine.functional.Functional object at 0x7f06ec417370>.h5

