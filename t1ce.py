import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.applications import InceptionV3, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
#custom functions
from Functions import load_from_dir, preprocess, add_labels, generate_class_weights, model_train, plot_acc_loss_f1

#load data
HGG_list_train = load_from_dir('/content/drive/MyDrive/Split_data/t1ce_mri_slices/train/HGG_t1ce')
LGG_list_train = load_from_dir('/content/drive/MyDrive/Split_data/t1ce_mri_slices/train/LGG_t1ce')
HGG_list_val = load_from_dir('/content/drive/MyDrive/Split_data/t1ce_mri_slices/val/HGG_t1ce')
LGG_list_val = load_from_dir('/content/drive/MyDrive/Split_data/t1ce_mri_slices/val/LGG_t1ce')
HGG_list_test = load_from_dir('/content/drive/MyDrive/Split_data/t1ce_mri_slices/test/HGG_t1ce')
LGG_list_test = load_from_dir('/content/drive/MyDrive/Split_data/t1ce_mri_slices/test/LGG_t1ce')


#preprocess data
HGG_list_new_train = preprocess(HGG_list_train)
LGG_list_new_train = preprocess(LGG_list_train)

HGG_list_new_val = preprocess(HGG_list_val)
LGG_list_new_val = preprocess(LGG_list_val)

HGG_list_new_test = preprocess(HGG_list_test)
LGG_list_new_test = preprocess(LGG_list_test)

# Combine the HGG and LGG lists
X_train, y_train = add_labels([], [], HGG_list_new_train, label='HGG')
X_train, y_train = add_labels(X_train, y_train, LGG_list_new_train, label='LGG')

X_val, y_val = add_labels([], [], HGG_list_new_val, label='HGG')
X_val, y_val = add_labels(X_val, y_val, LGG_list_new_val, label='LGG')

X_test, y_test = add_labels([], [], HGG_list_new_test, label='HGG')
X_test, y_test = add_labels(X_test, y_test, LGG_list_new_test, label='LGG')

# Convert labels to numerical values and one-hot encoding
labels = {'HGG': 0, 'LGG': 1}
y_train = tf.keras.utils.to_categorical([labels[y] for y in y_train])
y_val = tf.keras.utils.to_categorical([labels[y] for y in y_val])
y_test = tf.keras.utils.to_categorical([labels[y] for y in y_test])

# Convert data to arrays and shuffle
X_val, y_val = shuffle(np.array(X_val), y_val, random_state=101)
X_train, y_train = shuffle(np.array(X_train), y_train, random_state=101)
X_test, y_test = shuffle(np.array(X_test), y_test, random_state=101)

class_weights = generate_class_weights(y_train, multi_class=False, one_hot_encoded=True)
print(class_weights)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

datagen = ImageDataGenerator(
    rotation_range=90,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='nearest')
    
train_generator = datagen.flow(
    X_train, y_train,
    shuffle=True)

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

# Search hyperparameters
tuners = [tuner_effnet, tuner_densenet, tuner_inception]
for tuner in tuners:
    tuner.search(train_generator,
                 validation_data=(X_val, y_val),
                 steps_per_epoch=len(train_generator),
                 epochs=50,
                 verbose=1
                 )
    # Print the best hyperparameters found by the tuner
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


history_inception_weights = model_train(model_name = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), save_name = "inception_t1ce", image_size = 224, dropout=0.7, optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), dense_0_units=16, dense_1_units=80, batch_size=64)  
history_effnet = model_train(model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3)), save_name = "effnet_t1ce", image_size = 224, dropout=0.6, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), dense_0_units=48, dense_1_units=80, batch_size=64)  
history_densenet_weights = model_train(model_name = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), save_name = "densenet_t1ce", image_size = 224, dropout=0.2, optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), dense_0_units=96, dense_1_units=None, batch_size=32)  
plot_acc_loss_f1_auc(history_inception_weights,  '/home/viktoriia.trokhova/plots/inception')
plot_acc_loss_f1_auc(history_densenet_weights,  '/home/viktoriia.trokhova/plots/densenet')
plot_acc_loss_f1_auc(history_effnet,  '/home/viktoriia.trokhova/plots/effnet')
