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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import keras_tuner
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters


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
  
  
  
def plot_acc_loss_f1_auc(model_history, folder_path):
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
    plt.savefig(os.path.join(folder_path, 'loss_t1ce.png'))
    plt.close()
    
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'accuracy_t1ce.png'))
    plt.close()
    
#     auc = model_history.history['auc']
#     val_auc = model_history.history['val_auc']
#     plt.plot(epochs, auc, 'y', label='Training AUC')
#     plt.plot(epochs, val_auc, 'r', label='Validation AUC')
#     plt.title('Training and validation AUC')
#     plt.xlabel('Epochs')
#     plt.ylabel('AUC')
#     plt.legend()
#     plt.savefig(os.path.join(folder_path, 'auc.png'))
#     plt.close()
    f1 = model_history.history['f1_score']
    val_f1 = model_history.history['val_f1_score']
    plt.plot(epochs, f1, 'y', label='Training F1 Score')
    plt.plot(epochs, val_f1, 'r', label='Validation F1 Score')
    plt.title('Training and validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'f1_score_t1ce.png'))
    plt.close()


def preprocess(images_list):
    list_new = []
    for img in images_list:
        img_color = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB)
#         mean_value = np.mean(img)
#         std_value = np.std(img)
#         normalized_image = (img - mean_value) / std_value
        #img_res = cv2.resize(img, (224, 224))
        img_cropped = tf.image.crop_to_bounding_box(img_color, 8, 8, 224, 224)
        img_processed = tf.keras.applications.imagenet_utils.preprocess_input(img_cropped)
        list_new.append(img_processed)
    return list_new


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
    
HGG_list_train = load_from_dir('/home/viktoriia.trokhova/T1CE_MRI_slices/train/HGG_t1ce')
LGG_list_train = load_from_dir('/home/viktoriia.trokhova/T1CE_MRI_slices/train/LGG_t1ce')


HGG_list_val = load_from_dir('/home/viktoriia.trokhova/T1CE_MRI_slices/val/HGG_t1ce')
LGG_list_val = load_from_dir('/home/viktoriia.trokhova/T1CE_MRI_slices/val/LGG_t1ce')

HGG_list_new_train = preprocess(HGG_list_train)
LGG_list_new_train = preprocess(LGG_list_train)
HGG_list_new_val = preprocess(HGG_list_val)
LGG_list_new_val = preprocess(LGG_list_val)
# Combine the HGG and LGG lists
X_train, y_train = add_labels([], [], HGG_list_new_train, label='HGG')
X_train, y_train = add_labels(X_train, y_train, LGG_list_new_train, label='LGG')
X_val, y_val = add_labels([], [], HGG_list_new_val, label='HGG')
X_val, y_val = add_labels(X_val, y_val, LGG_list_new_val, label='LGG')
# Convert labels to numerical values and one-hot encoding
labels = {'HGG': 0, 'LGG': 1}
y_train = tf.keras.utils.to_categorical([labels[y] for y in y_train])
y_val = tf.keras.utils.to_categorical([labels[y] for y in y_val])
# Convert data to arrays and shuffle
X_val, y_val = shuffle(np.array(X_val), y_val, random_state=101)
X_train, y_train = shuffle(np.array(X_train), y_train, random_state=101)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
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


def f1_score(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), dtype=tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0)), dtype=tf.float32))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate focal loss
    cross_entropy = -y_true * tf.math.log(y_pred)
    focal_loss = alpha * tf.pow(1.0 - y_pred, gamma) * cross_entropy
    
    return tf.reduce_mean(focal_loss, axis=-1)


# def model_train(model_name, image_size, learning_rate, dropout):
#     model = model_name.output
#     model = tf.keras.layers.GlobalAveragePooling2D()(model)
#     model = tf.keras.layers.Dropout(rate=dropout)(model)
#     model = tf.keras.layers.Dense(128, activation='relu')(model)
#     model = tf.keras.layers.Dense(16, activation='relu')(model)
#     model = tf.keras.layers.Dense(2, activation='softmax')(model)
#     model = tf.keras.models.Model(inputs=model_name.input, outputs=model)
#     adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     #sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate)
#     model.compile(loss=focal_loss, optimizer=adam, metrics=['accuracy', f1_score])
#     checkpoint = ModelCheckpoint("/home/viktoriia.trokhova/model_weights/inception_flair" + ".h5", monitor='val_f1_score', save_best_only=True, mode="max", verbose=1)
#     early_stop = EarlyStopping(monitor='val_f1_score', mode='max', patience=10, verbose=1, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_f1_score', factor=0.3, patience=2, min_delta=0.001, mode='max', verbose=1)
#     history = model.fit(train_generator, validation_data=(X_val, y_val), epochs=50, batch_size=64, verbose=1, callbacks=[checkpoint, early_stop, reduce_lr], class_weight=class_weights)
    
        
#     train_loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     train_accuracy = history.history['accuracy']
#     val_accuracy = history.history['val_accuracy']
#     train_f1_score = history.history['f1_score']
#     val_f1_score = history.history['val_f1_score']

#     print("Train Loss:", train_loss)
#     print("Val Loss:", val_loss)
#     print("Train Accuracy:", train_accuracy)
#     print("Val Accuracy:", val_accuracy)
#     print("Train F1 Score:", train_f1_score)
#     print("Val F1 Score:", val_f1_score)  
      
#     return history


import keras_tuner
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

def model_inception(hp):
    model_name = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2)
    model = model_name.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.8, step=0.1))(model)
    for i in range(hp.Int('num_layers', min_value=1, max_value=2)):
        model = tf.keras.layers.Dense(hp.Int(f'dense_{i}_units', min_value=16, max_value=128, step=16), activation='relu')(model)
    model = tf.keras.layers.Dense(2,activation='softmax')(model)
    model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
    
    # Define optimizer and batch size
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])
    learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.001, 0.01, 0.1])
    batch_size = hp.Choice('batch_size', values=[16, 32, 64])
    
    #Set optimizer parameters based on user's selection
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # Compile the model with the optimizer and metrics
    model.compile(loss=focal_loss, optimizer=optimizer, metrics=['accuracy', f1_score])
    
    return model

#Define hp before calling tuner.search()
hp = HyperParameters()


tuner = Hyperband(
    model_inception,
    objective=keras_tuner.Objective("val_f1_score", direction="max"),
    overwrite=True,
    max_epochs=30,
    factor=3,
    hyperband_iterations=5
)

tuner.search(train_generator,
             validation_data=(X_val, y_val),
             steps_per_epoch=len(train_generator),
             epochs=50,
             verbose=1
             )

#Print the best hyperparameters found by the tuner
best_hyperparams = tuner.get_best_hyperparameters(1)[0]
print(f'Best hyperparameters: {best_hyperparams}')


#Get the best model found by the tuner
best_model = tuner.get_best_models(1)[0]

checkpoint = ModelCheckpoint("/home/viktoriia.trokhova/model_weights/model_tuned" + ".h5",monitor='val_f1_score',save_best_only=True,mode="max",verbose=1)
early_stop = EarlyStopping(monitor='val_f1_score', mode='max', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_f1_score', factor = 0.3, patience = 2, min_delta = 0.001, mode='max',verbose=1)

#Fit the model to the training data for 50 epochs using the best hyperparameters
history_effnet = best_model.fit(
    train_generator,
    epochs=50,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[checkpoint, early_stop, reduce_lr]
)


  
  
#history_inception_weights = model_train(model_name = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), image_size = 224, learning_rate = 0.0001, dropout=0.3)
#history_effnet = model_train(model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3)), image_size = 224, learning_rate = 0.0001, dropout=0.7)
#history_densenet_weights = model_train(model_name = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), image_size = 224, learning_rate = 0.1, dropout=0.3)
#history_resnet_weights = model_train(model_name = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), image_size = 224, learning_rate = 0.1, dropout=0.5)
#plot_acc_loss_f1_auc(history_inception_weights,  '/home/viktoriia.trokhova/plots/inception')
#plot_acc_loss_f1_auc(history_densenet_weights,  '/home/viktoriia.trokhova/plots/densenet')
#plot_acc_loss_f1_auc(history_effnet,  '/home/viktoriia.trokhova/plots/effnet')
#history_densenet_weights = model_train(model_name = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), image_size = 224, learning_rate = 0.1, dropout=0.5)
#history_inception_weights = model_train(model_name = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), image_size = 224, learning_rate = 0.001, dropout=0.6)
