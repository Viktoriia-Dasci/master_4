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
from sklearn.preprocessing import MinMaxScaler
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
def add_labels(X, y, images_list, label):
  
    for img in images_list:
      X.append(img)
      y.append(label)
    return X, y
    
HGG_list_train = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/train/HGG_t2')
LGG_list_train = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/train/LGG_t2')


HGG_list_val = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/val/HGG_t2')
LGG_list_val = load_from_dir('/home/viktoriia.trokhova/Mri_slices_new/val/LGG_t2')



def preprocess(images_list):
    list_new = []
    for img in images_list:
        img_color = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB)
        img_cropped = tf.image.crop_to_bounding_box(img_color, 8, 8, 224, 224)
        img_processed = tf.keras.applications.imagenet_utils.preprocess_input(img_cropped)
        list_new.append(img_processed)
    return list_new
    
HGG_list_new_train = preprocess(HGG_list_train)
LGG_list_new_train = preprocess(LGG_list_train)
HGG_list_new_val = preprocess(HGG_list_val)
LGG_list_new_val = preprocess(LGG_list_val)
HGG_list_masks_new_val = preprocess(HGG_list_val_masks)
LGG_list_masks_new_val = preprocess(LGG_list_val_masks)


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
    rotation_range=90,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='nearest')
    
    
train_generator = datagen.flow(
    X_train, y_train,
    shuffle=True)
from sklearn.metrics import f1_score
import numpy as np

import keras_tuner
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

def model_train(model_name, image_size, learning_rate, dropout):
    #model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))
    model = model_name.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dense(128, activation='relu')(model)
    model = tf.keras.layers.Dropout(rate=dropout)(model)
    model = tf.keras.layers.Dense(2,activation='softmax')(model)
    model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
    #adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics= ['accuracy', 'AUC'])
    #callbacks
    #tensorboard = TensorBoard(log_dir = 'logs')
    checkpoint = ModelCheckpoint("/home/viktoriia.trokhova/model_weights/history_densenet_weights" + ".h5",monitor='val_auc',save_best_only=True,mode="max",verbose=1)
    early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=20, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_auc', factor = 0.3, patience = 10, min_delta = 0.001, mode='max',verbose=1)
    #fitting the model
    history = model.fit(train_generator, validation_data=(X_val, y_val), epochs=50, batch_size=8, verbose=1,
                   callbacks=[checkpoint, early_stop, reduce_lr], class_weight=class_weights)
     
    return history


#history_effnet = model_train(model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3)), image_size = 224, learning_rate = 0.0009, dropout=0.4)

#history_resnet_weights = model_train(model_name = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), image_size = 224, learning_rate = 0.1, dropout=0.5)

history_densenet_weights = model_train(model_name = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), image_size = 224, learning_rate = 0.1, dropout=0.5)

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
plot_acc_loss_auc(history_densenet_weights,  '/home/viktoriia.trokhova/plots/densenet')
