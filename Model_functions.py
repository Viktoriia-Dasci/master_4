import os
import glob
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras_tuner import HyperParameters as hp

class_weights = {0: 0.63, 1: 2.43}


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
  
  
def plot_acc_loss_f1(model_history, folder_path, model):

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f'loss_{model}.png'))
    plt.close()
    
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f'accuracy_{model}.png'))
    plt.close()
    f1 = model_history.history['f1_score']
    val_f1 = model_history.history['val_f1_score']
    plt.plot(epochs, f1, 'y', label='Training F1 Score')
    plt.plot(epochs, val_f1, 'r', label='Validation F1 Score')
    plt.title('Training and validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f'f1_score_{model}.png'))
    plt.close()


def preprocess(images_list):
    list_new = []
    for img in images_list:
        img_color = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB)
        img_cropped = tf.image.crop_to_bounding_box(img_color, 8, 8, 224, 224)
        img_processed = tf.keras.applications.imagenet_utils.preprocess_input(img_cropped)
        list_new.append(img_processed)
    return list_new

def preprocess_stack(images_list):
    list_new = []
    for img in images_list:
        img_cropped = tf.image.crop_to_bounding_box(img, 8, 8, 224, 224)
        img_cropped = tf.keras.applications.imagenet_utils.preprocess_input(img_cropped)
        list_new.append(img_cropped)
    return list_new

#Adapted from 'https://gist.github.com/angeligareta/83d9024c5e72ac9ebc34c9f0b073c64c'
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


def focal_loss(class_weights):
    def focal_loss_with_weights(y_true, y_pred, gamma=2.0):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = tf.pow(1.0 - y_pred, gamma) * cross_entropy

        # apply class weights
        class_weights_array = np.array(list(class_weights.values()))
        loss = loss * class_weights_array

        return tf.reduce_mean(loss, axis=-1)
    return focal_loss_with_weights    


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


def model_train(model_name, save_name, image_size, dropout, optimizer, dense_0_units, dense_1_units, batch_size):
    model = model_name.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=dropout)(model)
    model = tf.keras.layers.Dense(dense_0_units, activation='relu')(model)
    if dense_1_units is not None:
          model = tf.keras.layers.Dense(dense_1_units, activation='relu')(model)
          model = tf.keras.layers.Dense(2, activation='softmax')(model)
          model = tf.keras.models.Model(inputs=model_name.input, outputs=model)
          model.compile(loss=focal_loss, optimizer=optimizer, metrics=['accuracy', f1_score])
    else:
          model = tf.keras.layers.Dense(2, activation='softmax')(model)
          model = tf.keras.models.Model(inputs=model_name.input, outputs=model)
          model.compile(loss=focal_loss(class_weights), optimizer=optimizer, metrics=['accuracy', f1_score])
    
    checkpoint = ModelCheckpoint("/home/viktoriia.trokhova/model_weights/" + save_name + ".h5", monitor='val_f1_score', save_best_only=True, mode="max", verbose=1)
    early_stop = EarlyStopping(monitor='val_f1_score', mode='max', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_f1_score', factor=0.3, patience=5, min_delta=0.001, mode='max', verbose=1)
    history = model.fit(train_generator, validation_data=(X_val, y_val), epochs=50, batch_size=batch_size, verbose=1, callbacks=[checkpoint, early_stop, reduce_lr])
        
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_f1_score = history.history['f1_score']
    val_f1_score = history.history['val_f1_score']

    print("Train Loss:", train_loss)
    print("Val Loss:", val_loss)
    print("Train Accuracy:", train_accuracy)
    print("Val Accuracy:", val_accuracy)
    print("Train F1 Score:", train_f1_score)
    print("Val F1 Score:", val_f1_score)  
      
    return history

def model_effnet(hp):
    model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model = model_name.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.8, step=0.1))(model)
    for i in range(hp.Int('num_layers', min_value=1, max_value=2)):
        model = tf.keras.layers.Dense(hp.Int(f'dense_{i}_units', min_value=16, max_value=128, step=16), activation='relu')(model)
    model = tf.keras.layers.Dense(2, activation='softmax')(model)
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
    model.compile(loss=focal_loss(class_weights), optimizer=optimizer, metrics=['accuracy', f1_score])
    
    return model

def model_densenet(hp):
    model_name = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2)
    model = model_name.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.8, step=0.1))(model)
    for i in range(hp.Int('num_layers', min_value=1, max_value=2)):
        model = tf.keras.layers.Dense(hp.Int(f'dense_{i}_units', min_value=16, max_value=128, step=16), activation='relu')(model)
    model = tf.keras.layers.Dense(2, activation='softmax')(model)
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
    model.compile(loss=focal_loss(class_weights), optimizer=optimizer, metrics=['accuracy', f1_score])
    
    return model

def model_inception(hp):
    model_name = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2)
    model = model_name.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.8, step=0.1))(model)
    for i in range(hp.Int('num_layers', min_value=1, max_value=2)):
        model = tf.keras.layers.Dense(hp.Int(f'dense_{i}_units', min_value=16, max_value=128, step=16), activation='relu')(model)
    model = tf.keras.layers.Dense(2, activation='softmax')(model)
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
    model.compile(loss=focal_loss(class_weights), optimizer=optimizer, metrics=['accuracy', f1_score])
    
    return model
