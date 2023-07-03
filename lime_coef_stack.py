import numpy as np
import glob
import os
import random
import tensorflow as tf
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
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import img_as_float
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
#import lime




def load_from_dir(path):
      file_paths = glob.glob(os.path.join(path, '*.npy'))
      print(file_paths)

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

# def model_train(model_name, image_size = 224, learning_rate = 0.0009, dropout=0.4):
#       model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))
#       model = model_name.output
#       model = tf.keras.layers.GlobalAveragePooling2D()(model)
#       model = tf.keras.layers.Dense(128, activation='relu')(model)
#       model = tf.keras.layers.Dropout(rate=dropout)(model)
#       model = tf.keras.layers.Dense(2,activation='softmax')(model)
#       model = tf.keras.models.Model(inputs=model_name.input, outputs = model)
#       adam = tf.keras.optimizers.Adam(learning_rate=0.001)
#       #sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate)
#       model.compile(loss='categorical_crossentropy', optimizer = adam, metrics= ['accuracy', 'AUC'])
#       #callbacks
#       #tensorboard = TensorBoard(log_dir = 'logs')
#       #checkpoint = ModelCheckpoint("/home/viktoriia.trokhova/model_weights/resnet_weughts_new" + ".h5",monitor='val_auc',save_best_only=True,mode="max",verbose=1)
#       early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=10, verbose=1, restore_best_weights=True)
#       reduce_lr = ReduceLROnPlateau(monitor = 'val_auc', factor = 0.3, patience = 2, min_delta = 0.001, mode='max',verbose=1)
#       #fitting the model
#       model.fit(train_generator, validation_data=(X_val, y_val), epochs=50, batch_size=8, verbose=1,
#                       callbacks=[early_stop, reduce_lr], class_weight=class_weights)

#       return model


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



# def display_images(images_list):
#     for i, image in enumerate(images_list):
#         print(image.min())
#         print(image.max())
#         plt.figure(figsize=(6,6))
#         plt.imshow(rgb2gray(image))
#         plt.axis('off')
#         plt.title(f"Image {i}")
#         plt.show()
#         plt.close()




HGG_list_test = load_from_dir('/home/viktoriia.trokhova/Stacked_MRI_new/test/HGG_stack')
LGG_list_test = load_from_dir('/home/viktoriia.trokhova/Stacked_MRI_new/test/LGG_stack')


def preprocess(images_list):
    list_new = []
    for img in images_list:
        #img_color = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB)
        #img_cropped = cv2.resize(img_color,(224,224))
        #img_cropped = tf.convert_to_tensor(img_cropped)
        img_cropped = tf.image.crop_to_bounding_box(img, 8, 8, 224, 224)
        img_cropped = tf.keras.applications.imagenet_utils.preprocess_input(img_cropped)
        list_new.append(img_cropped)
    return list_new


HGG_list_new_test = preprocess(HGG_list_test)
LGG_list_new_test = preprocess(LGG_list_test)
# HGG_list_new_test = (HGG_list_test)
# LGG_list_new_test = resize(LGG_list_test)


#Combine the HGG and LGG lists
X_test, y_test = add_labels([], [], HGG_list_new_test, label='HGG')
X_test, y_test = add_labels(X_test, y_test, LGG_list_new_test, label='LGG')

#Convert labels to numerical values and one-hot encoding
labels = {'HGG': 0, 'LGG': 1}
y_test = tf.keras.utils.to_categorical([labels[y] for y in y_test])

#Convert data to arrays and shuffle
X_test, y_test = shuffle(np.array(X_test), y_test, random_state=101)


def f1_score(y_true, y_pred):
    y_pred = tf.cast(tf.math.greater_equal(y_pred, 0.5), dtype=tf.float32)
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


pretrained_model = load_model('/home/viktoriia.trokhova/model_weights/effnet_stacked_tuned.h5', custom_objects={'f1_score': f1_score, 'focal_loss': focal_loss})

model = 'effnet'

modality = 'Stacked'


pred_res = pretrained_model.predict(X_test)
pred_ready_res = np.argmax(pred_res,axis=1)
y_test_new_res = np.argmax(y_test,axis=1)


import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

# Reference: https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0
class GuidedBackprop:
    def __init__(self,model, layerName=None):
        self.model = model
        self.layerName = layerName
        if self.layerName == None:
            self.layerName = self.find_target_layer()

        self.gbModel = self.build_guided_model()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu

        return gbModel

    def guided_backprop(self, images, upsample_size):
        """Guided Backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x




import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model






from lime import lime_image
from skimage.segmentation import mark_boundaries

# explainer = lime_image.LimeImageExplainer()

# explanation = explainer.explain_instance(img_rgb,
#                                          pretrained_model.predict,
#                                          top_labels=1,
#                                          hide_color=0,
#                                          num_samples=3000)


# explanation= explainer.explain_instance
#             (images_inc_im[0].astype('double'), inet_model.predict,  top_labels=3, hide_color=0, num_samples=1000)

from mpl_toolkits.axes_grid1 import make_axes_locatable


# temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=True)
# temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=False)


def dice_coefficient(img1, img2):
    intersection = np.logical_and(img1, img2)
    return 2.0 * intersection.sum() / (img1.sum() + img2.sum())

def iou(img1, img2):
    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    iou_score = intersection / union
    return iou_score


lime_list = []

# Create an empty dataframe to store the data
#df = pd.DataFrame(columns=["img_class", "img_num", "pat_num", "dice_coef"])

from lime import lime_image
from skimage.segmentation import mark_boundaries

def explanation_heatmap(exp, exp_class):
    '''
    Using heat-map to highlight the importance of each super-pixel for the model prediction
    '''
    dict_heatmap = dict(exp.local_exp[exp_class])
    heatmap = np.vectorize(dict_heatmap.get)(exp.segments)
    # plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    # plt.colorbar()
    # plt.show()
    return heatmap


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


lime_list = []

# Create an empty dataframe to store the data
#df = pd.DataFrame(columns=["img_class", "img_num", "pat_num", "dice_coef"])

def lime_coef(img_class, img_msk_class):
    img_dir = "/home/viktoriia.trokhova/Stacked_MRI_new/test/" + img_class + "_stack"
    img_files = os.listdir(img_dir)
    msk_dir = "/home/viktoriia.trokhova/T2_new_Msk_slices/test/"  + img_class + "_masks"
    msk_files = os.listdir(msk_dir)

    # Get the last number of the file names to set pat_num


    # Loop through all the images and process them
    for i, img_file in enumerate(img_files):
        # Extract the image name and number
        img_name = img_file.split(".")[0]
        print(img_file)
        print(img_name)
        img_num = int(img_name.split("_")[0])
        print(img_num)
        pat_num = int(img_name.split("_")[1])
        print(pat_num)

        # Load the image array
        img_arr = np.load(os.path.join(img_dir, img_file))
        img_arr = cv2.resize(img_arr, (224, 224))
        img_rgb = np.float32(img_arr)
        #img_rgb = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
        plt.show()

        # Process the image
        #plt.imshow(img_arr)
        # plt.axis("off")
        # save_results_to = '/content/' + str(img_num) + '_' + str(pat_num)
        # plt.savefig(save_results_to, bbox_inches='tight', pad_inches=0)
        # img = cv2.imread('/content/' + str(img_num) + '_' + str(pat_num) + '.png')
        # img_rgb = cv2.resize(img, (224, 224))

        #Process the mask
        img_msk =np.load(os.path.join(msk_dir, img_file))
        img_msk = cv2.resize(img_msk, (224, 224))
        msk_float32 = np.float32(img_msk)
        msk_rgb = cv2.cvtColor(msk_float32, cv2.COLOR_GRAY2RGB)
        #plt.imshow(img_msk)
        # plt.axis("off")
        # save_results_to = '/content/' + str(msk_num) + '_' + str(pat_num) + 'mask'
        # plt.savefig(save_results_to, bbox_inches='tight', pad_inches=0)
        # msk=cv2.imread('/content/' + str(msk_num) + '_' + str(pat_num) + 'mask' + '.png')
        # msk_rgb=cv2.resize(msk,(224,224))

        #calculate Lime
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img_rgb,
                                                pretrained_model.predict,
                                                top_labels=1,
                                                hide_color=0,
                                                num_samples=1000)

        temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=True)
        img_hide = mark_boundaries(temp_1, mask_1)
        img_hide = img_hide/np.amax(img_hide)
        img_hide = np.clip(img_hide, 0, 1)
        # gradCAM = GradCAM(model=pretrained_model)
        # cam3 = gradCAM.compute_heatmap(image=np.expand_dims(img_rgb,axis=0),classIdx=0,upsample_size=target_size)

        # guidedBP = GuidedBackprop(model=pretrained_model)
        # gb_cam = guidedBP.guided_backprop(np.expand_dims(img_rgb,axis=0),target_size)
        # guided_gradcam = deprocess_image(gb_cam*cam3)

        # cam3 = gradCAM.compute_heatmap(image=np.expand_dims(img_rgb,axis=0),classIdx=0,upsample_size=target_size)
        msk_rgb = cv2.normalize(msk_rgb, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        #convert GradCam and mask to binary
        msk_rgb[np.mean(msk_rgb, axis=-1)<0.2] = 0
        msk_rgb[np.mean(msk_rgb, axis=-1)>=0.2] = 1
        #plt.imshow(msk_rgb)

        img_hide = cv2.normalize(img_hide, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img_hide[np.mean(img_hide, axis=-1)<0.2] = 0
        img_hide[np.mean(img_hide, axis=-1)>=0.2] = 1
        # guided_gradcam = cv2.normalize(guided_gradcam, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        # guided_gradcam[np.mean(guided_gradcam, axis=-1)<0.6] = 0
        # guided_gradcam[np.mean(guided_gradcam, axis=-1)>=0.6] = 1
        #plt.imshow(cam3)

        dice_coef = dice_coefficient(msk_rgb, img_hide)
        iou_coef = iou(msk_rgb, img_hide)

        # Append a new row to the dataframe with the data
        new_row = {"img_class": img_class, "img_num": img_num, "pat_num": pat_num, "dice_coef": dice_coef, "iou_coef": iou_coef}
        #df_row = pd.DataFrame(new_row, index=[0])
        #print(df_row)
        lime_list.append(new_row)
        #df.append(new_row, ignore_index=True)
        #print(df)

    df = pd.DataFrame(lime_list)

    return df

df_lime_LGG = lime_coef('LGG', pretrained_model)
df_lime_LGG.to_csv("/home/viktoriia.trokhova/explain_datasets/" + "/lime_coef_stack_" + model + ".csv", index=False)

df_lime_HGG = lime_coef('HGG', pretrained_model)
df_lime_HGG.to_csv("/home/viktoriia.trokhova/explain_datasets/" + "/lime_coef_stack_" + model + ".csv", index=False)
