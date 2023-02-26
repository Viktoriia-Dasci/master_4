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

HGG_list_new_train = resize(HGG_list_train, image_size = 224)
LGG_list_new_train = resize(LGG_list_train, image_size = 224)

HGG_list_new_val = resize(HGG_list_val, image_size = 224)
LGG_list_new_val = resize(LGG_list_val, image_size = 224)
HGG_list_masks_new_val = resize(HGG_list_val_masks, image_size = 224)
LGG_list_masks_new_val = resize(LGG_list_val_masks, image_size = 224)

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

labels = ['HGG', 'LGG', 'NB']
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

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

norm_image = cv2.normalize(X_train[0], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

def model_train(model_name, image_size = 224):
    #model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))
    model = model_name.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=0.79)(model)
    model = tf.keras.layers.Dense(2,activation='softmax')(model)
    model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
    sgd = SGD(learning_rate=0.004)
    model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics= ['accuracy'])
    #callbacks
    tensorboard = TensorBoard(log_dir = 'logs')
    checkpoint = ModelCheckpoint(str(model_name) + ".h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001, mode='auto',verbose=1)
    #fitting the model
    history = model.fit(X_train,y_train,validation_data=(X_val, y_val), epochs=30, verbose=1, batch_size=32,
                   callbacks=[tensorboard, checkpoint, reduce_lr])
    
return history


def plot_acc_loss(model_history):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
    
history_effnet = model_train(model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3)))

plot_acc_loss(history_effnet)

history_resnet50 = model_train(model_name = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2))

plot_acc_loss(history_resnet50)

history_inceptionv3 = model_train(model_name = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2))

plot_acc_loss(history_inceptionv3)

history_densenet121 = model_train(model_name = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2))

plot_acc_loss(history_densenet121)

colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']
