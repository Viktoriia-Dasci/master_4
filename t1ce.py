import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons import metrics
import keras_tuner
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
#custom functions
from Model_functions import *

home_dir = '/home/viktoriia.trokhova/'

base_dir = '/home/viktoriia.trokhova/Split_data/'

modality = 't1ce'

#load data
HGG_list_train = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/train/HGG_{modality}')
LGG_list_train = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/train/LGG_{modality}')
HGG_list_val = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/val/HGG_{modality}')
LGG_list_val = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/val/LGG_{modality}')
HGG_list_test = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/test/HGG_{modality}')
LGG_list_test = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/test/LGG_{modality}')


#preprocessing data
HGG_list_new_train = preprocess(HGG_list_train)
LGG_list_new_train = preprocess(LGG_list_train)

HGG_list_new_val = preprocess(HGG_list_val)
LGG_list_new_val = preprocess(LGG_list_val)

HGG_list_new_test = preprocess(HGG_list_test)
LGG_list_new_test = preprocess(LGG_list_test)

# Combining the HGG and LGG lists
X_train, y_train = add_labels([], [], HGG_list_new_train, label='HGG')
X_train, y_train = add_labels(X_train, y_train, LGG_list_new_train, label='LGG')

X_val, y_val = add_labels([], [], HGG_list_new_val, label='HGG')
X_val, y_val = add_labels(X_val, y_val, LGG_list_new_val, label='LGG')

X_test, y_test = add_labels([], [], HGG_list_new_test, label='HGG')
X_test, y_test = add_labels(X_test, y_test, LGG_list_new_test, label='LGG')

# Converting labels to numerical values and one-hot encoding
labels = {'HGG': 0, 'LGG': 1}
y_train = tf.keras.utils.to_categorical([labels[y] for y in y_train])
y_val = tf.keras.utils.to_categorical([labels[y] for y in y_val])
y_test = tf.keras.utils.to_categorical([labels[y] for y in y_test])

# Converting data to arrays and shuffle
X_val, y_val = shuffle(np.array(X_val), y_val, random_state=101)
X_train, y_train = shuffle(np.array(X_train), y_train, random_state=101)
X_test, y_test = shuffle(np.array(X_test), y_test, random_state=101)

#Calculating class_weights
class_weights = generate_class_weights(y_train, multi_class=False, one_hot_encoded=True)
print(class_weights)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


#Data augmentation
datagen = ImageDataGenerator(
    rotation_range=90,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='nearest')
    
train_generator = datagen.flow(
    X_train, y_train,
    shuffle=True)

#Hyperparameter tuning
hp = HyperParameters()

tuner_effnet = Hyperband(
    model_effnet,
    objective=keras_tuner.Objective("val_f1_score", direction="max"),
    overwrite=True,
    max_epochs=30,
    factor=3,
    hyperband_iterations=5
)

tuner_densenet = Hyperband(
    model_densenet,
    objective=keras_tuner.Objective("val_f1_score", direction="max"),
    overwrite=True,
    max_epochs=30,
    factor=3,
    hyperband_iterations=5
)

tuner_inception = Hyperband(
    model_inception,
    objective=keras_tuner.Objective("val_f1_score", direction="max"),
    overwrite=True,
    max_epochs=30,
    factor=3,
    hyperband_iterations=5
)

# Searching for best hyperparameters and models for each tuner
best_hyperparameters = {}
best_models = {}
tuners = {'EffNet': tuner_effnet, 'DenseNet': tuner_densenet, 'Inception': tuner_inception}

for name, tuner in tuners.items():
    tuner.search(train_generator,
                 validation_data=(X_val, y_val),
                 steps_per_epoch=len(train_generator),
                 epochs=50,
                 verbose=1
                 )
    
    best_hyperparameters[name] = tuner.get_best_hyperparameters(1)[0]
    best_models[name] = tuner.get_best_models(1)[0]

# Define callbacks
early_stop = EarlyStopping(monitor='val_f1_score', mode='max', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_f1_score', factor = 0.3, patience = 5, min_delta = 0.001, mode='max',verbose=1)

# Define the path for saving the plots
plot_folder_path = os.path.join(home_dir, f"model_plots/{modality}") 

# Fit the best model from each tuner to the training data for 50 epochs using the best hyperparameters
for name, model in best_models.items():
    print(f'Training {name}...')
    checkpoint = ModelCheckpoint(f"{home_dir}/model_weights/model_tuned_{name}.h5", monitor='val_f1_score',save_best_only=True, mode="max",verbose=1)
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # After training, plot the accuracy, loss, and f1 score
    plot_acc_loss_f1(history, plot_folder_path, name)

#Training models with the best hyperparameters inputted manually
# history_inception_weights = model_train(model_name = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), save_name = f"inception_{modality}", image_size = 224, dropout=0.7, optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), dense_0_units=16, dense_1_units=80, batch_size=64)  
# history_effnet = model_train(model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3)), save_name = f"effnet_{modality}", image_size = 224, dropout=0.6, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), dense_0_units=48, dense_1_units=80, batch_size=64)  
# history_densenet_weights = model_train(model_name = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), save_name = f"densenet_{modality}", image_size = 224, dropout=0.2, optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), dense_0_units=96, dense_1_units=None, batch_size=32)  
# plot_acc_loss_f1(history_inception_weights,  plot_folder_path, 'inception')  
# plot_acc_loss_f1(history_densenet_weights,  plot_folder_path, 'densenet') 
# plot_acc_loss_f1(history_effnet,  plot_folder_path, 'effnet')
