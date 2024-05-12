# -*- coding: utf-8 -*-

#importing libraries
import os
import pathlib
from glob import glob
import random
import numpy as np
import PIL #Python Imaging Library
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report
import numpy as np

#finding image files
img_dir = '/path'
img_dir = pathlib.Path(img_dir)

#checking directories to ensure we have the correct # of images for each class
total_files = 0
for root, dirs, files in os.walk(str(img_dir)):
    level = root.replace(str(img_dir), '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f'{indent}{os.path.basename(root)}/ ({len(files)} files)')
    total_files += len(files)
print(f'There are {total_files} images in this dataset')

#retrieving labels
indoor_dir = sorted([ name for name in list(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name)) ])
print(f' Indoor Labels: {indoor_dir}')


#fixing seed to reproduce results
SEED = 1001
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from IPython.display import display
#loop over the first three indoor scene labels to display images
for i in range(3):
    #/* get all imgage files
    image_file = list(img_dir.glob(indoor_dir[i]+'/*'))
    # we open the first image  only using PIL library for handleing 3D data
    img = PIL.Image.open(str(image_file[0]))
    # channel number= img.mode
    print(f'(Image size  = ({img.size[0]}, {img.size[1]}, {len(img.mode)}) ; Indoor Scene = {indoor_dir[i]})')
    display(img)

#setting parameters for preprocessing function
batch_size = 16
image_height = 256
image_width = 256
train_test_split = 0.2

#training dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
  img_dir, # parent folder contains all the folders contaning fruits
  labels='inferred', # labels are generated from the directory structure
  label_mode='int', #'int': means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).
  validation_split= train_test_split,
  subset="training",
  seed= 1001, #fix the seed
  image_size=(image_height, image_width),
  batch_size=batch_size)

#validation datase
val_data = tf.keras.preprocessing.image_dataset_from_directory(
  img_dir,
  labels='inferred',
  label_mode='int',
  validation_split= train_test_split,
  subset="validation",
  seed=1001,
  image_size=(image_height, image_width),
  batch_size=batch_size)

#importing libraries
import matplotlib.pyplot as plt

#visualizing training data
# we resize all the original images with different sizes to the same size
plt.figure(figsize=(12, 12))
for img, lab in train_data.take(1):
  # we only plot 8 images out of 16 images (16 images = 1 batch)
  for i in range(8):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(img[i].numpy().astype("uint16"))
    # Map the label index to name
    #lab[i]: label encoding
    plt.title(indoor_dir[lab[i]])
    plt.axis("off")

#finding shapes of image_batch and labels_batch
for image_batch, labels_batch in train_data:
  print(f'''image_batch.shape = {image_batch.shape};
labels_batch.shape = {labels_batch.shape } ''')
  break

#configuring the dataset for better performance
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

#assigning variables to used in the CNN model later
num_labels = len(indoor_dir) #61 indoor classes
image_channel = 3 #

#creating first CNN model
model_1 = tf.keras.Sequential([
  #normalizing pixel values so they fall within the range of [0,1]
  layers.experimental.preprocessing.Rescaling(1.0/255.0, input_shape=(image_height, image_width, image_channel)),

  #adding first convolution layer
  layers.Conv2D(32, 3, padding='same', activation='relu'),

  #adding the first max pooling layer
  layers.MaxPooling2D((3)),

  #adding second convolution layer
  layers.Conv2D(64, 3, padding='same', activation='relu'),

  #adding the second max pooling layer
  layers.MaxPooling2D((3)),

  #adding third convolution layer
  layers.Conv2D(128, 3, padding='same', activation='relu'),

  #we need to change the dimensions of the output array to 2D to make it work with the classification algorithm,
  #so we will add a flatten layer
  layers.Flatten(),

  #adding first layer for classification
  layers.Dense(256, activation='relu'),

  #adding second layer for classification
  layers.Dense(128, activation='relu'),

  #since we have 61 labels, we need to have 61 neurons for the final output classification layer; activation=None
  #we will use num_labels we created earlier since it's already set to the number of indoor scene classes
  layers.Dense(num_labels, activation=None)
])

#building model to specify input shape
model_1.build(input_shape=(None, image_height, image_width, image_channel))

#summarizing model
model_1.summary()

#configuring the model
model_1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) #we want to use accuracy as the metric for the model

#adding early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience= 5)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #training the model and storing the history in a variable to plot later
# history = model_1.fit(train_data,
#                     epochs=50,
#                     validation_data=val_data,
#                     callbacks=[callback],
#                     verbose = 1)

#plotting the model's history
import seaborn as sns
import pandas as pd

train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
#plotting train loss
sns.lineplot(x='epoch', y ='loss', data =train_history)
#plotting validation loss
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
#adding legends
plt.legend(labels=['train_loss', 'val_loss'])
plt.title('Training and Validation Loss Over Epochs')
plt.show()

#plotting training accuracy
sns.lineplot(x='epoch', y ='accuracy', data =train_history)
#Plot validation accuracy
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
#Add legends
plt.legend(labels=['train_accuracy', 'val_accuracy'])
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()


#creating array of predicted values (as integers)
pred_prod = model_1.predict(val_data)
pred_int = np.argmax(pred_prod, axis=-1)

#creating array of actual values from val_data (as integers)
val_labels = val_data.map(lambda x, y: y).unbatch() #extracting labels from val_data
actual_int = np.array(list(val_labels.as_numpy_iterator())) #converting labels to int

#mapping integers to labels
indoor_labels = np.array(indoor_dir) #converting indoor_dir to numpy array to map labels
pred_labels = np.array(indoor_labels)[pred_int]
actual_labels = np.array(indoor_labels)[actual_int]

#creating classification report
print(classification_report(pred_labels, actual_labels, target_names=indoor_labels))


#specifying dropout rate
dropout_rate = 0.3

model_2 = tf.keras.Sequential([

  #adding data augmentation layers
  layers.experimental.preprocessing.RandomTranslation(height_factor=0.2,width_factor = 0.2),
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape=(image_height,image_width,image_channel)),
  layers.experimental.preprocessing.RandomRotation(0.3),

  #normalizing pixel values so they fall within the range of [0,1]
  layers.experimental.preprocessing.Rescaling(1.0/255.0, input_shape=(image_height, image_width, image_channel)),

  #adding first convolution layer
  layers.Conv2D(32, 3, padding='same', activation='relu'),

  #adding the first max pooling layer
  layers.MaxPooling2D((3)),

  #adding second convolution layer
  layers.Conv2D(64, 3, padding='same', activation='relu'),

  #adding the second max pooling layer
  layers.MaxPooling2D((3)),

  #adding third convolution layer
  layers.Conv2D(128, 3, padding='same', activation='relu'),

  #we need to change the dimensions of the output array to 2D to make it work with the classification algorithm,
  #so we will add a flatten layer
  layers.Flatten(),

  #adding first layer for classification
  layers.Dense(256, activation='relu'),

  #adding dropout layer
  layers.Dropout(rate = dropout_rate),

  #adding second layer for classification
  layers.Dense(128, activation='relu'),

  #since we have 61 labels, we need to have 61 neurons for the final output later; activation=None
  #we will use num_labels we created earlier since it's already set to the number of indoor scene classes
  layers.Dense(num_labels, activation=None)
])

#building model to specify input shape
model_2.build(input_shape=(None, image_height, image_width, image_channel))

#summarizing model
model_2.summary()

#configuring the model
model_2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) #we want to use accuracy as the metric for the model

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #training the model and storing the history in a variable to plot later
# history = model_2.fit(train_data,
#                     epochs=50,
#                     validation_data=val_data,
#                     callbacks=[callback],
#                     verbose = 1)

#plotting the model's history
import seaborn as sns
import pandas as pd

train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
#plotting train loss
sns.lineplot(x='epoch', y ='loss', data =train_history)
#plotting validation loss
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
#adding legends
plt.legend(labels=['train_loss', 'val_loss'])
plt.title('Training and Validation Loss Over Epochs')
plt.show()

#plotting training accuracy
sns.lineplot(x='epoch', y ='accuracy', data =train_history)
#Plot validation accuracy
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
#Add legends
plt.legend(labels=['train_accuracy', 'val_accuracy'])
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()

#creating array of predicted values (as integers)
pred_prod = model_2.predict(val_data)
pred_int = np.argmax(pred_prod, axis=-1)

#creating array of actual values from val_data (as integers)
val_labels = val_data.map(lambda x, y: y).unbatch() #extracting labels from val_data
actual_int = np.array(list(val_labels.as_numpy_iterator())) #converting labels to int

#mapping integers to labels
indoor_labels = np.array(indoor_dir) #converting indoor_dir to numpy array to map labels
pred_labels = np.array(indoor_labels)[pred_int]
actual_labels = np.array(indoor_labels)[actual_int]

#creating classification report
print(classification_report(pred_labels, actual_labels, target_names=indoor_labels))


#importing pre-trained ResNet50 model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

#specifiying the image size for our dataset
IMG_SHAPE = (image_height, image_width, image_channel)

resnet_model = ResNet50(weights='imagenet',
                        include_top=False,
                        input_shape = IMG_SHAPE)

#freezing convolutional base
resnet_model.trainable = False


#specifying the input size of the images
inputs = tf.keras.Input(shape = IMG_SHAPE)

#creating data augmentation layer
data_aug = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.2,width_factor = 0.2),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", input_shape=(image_height,image_width,image_channel)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)
])
#passing inputs through data_aug layer
x = data_aug(inputs)

#passing inputs through preprocess layer that we imported earlier
x = preprocess_input(x)

#freeze the convolutional base so that we don't retrain the weights
#passing inputs through model
x = resnet_model(x, training=False)

#converting 4D to a 2D using Flatten; we also used flatten instead of global average pooling to retain information
flatten_layer = tf.keras.layers.Flatten()
x= flatten_layer(x)

#applying the classification layer since we did not include it when we retrieved the ResNet50 model (include_top=False).
#we have 61 indoor classes; we'll use num_labels since it contains that information
classification_layer = tf.keras.layers.Dense(num_labels)
outputs = classification_layer(x)

#using the functional API approach to combine the inputs and outputs together
model_3 = tf.keras.Model(inputs, outputs)

#configuring model
model_3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_3.summary()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #training the model and storing the history in a variable to plot later
# history = model_3.fit(train_data,
#                     epochs=50,
#                     validation_data=val_data,
#                     callbacks=[callback],
#                     verbose = 1)

#plotting model's history
train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
#plotting train loss
sns.lineplot(x='epoch', y ='loss', data =train_history)
#plotting validation loss
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
#adding legends
plt.legend(labels=['train_loss', 'val_loss'])
plt.title('Training and Validation Loss Over Epochs')
plt.show()

#plotting training accuracy
sns.lineplot(x='epoch', y ='accuracy', data =train_history)
#Plot validation accuracy
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
#Add legends
plt.legend(labels=['train_accuracy', 'val_accuracy'])
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()

#creating array of predicted values (as integers)
pred_prod = model_3.predict(val_data)
pred_int = np.argmax(pred_prod, axis=-1)

#creating array of actual values from val_data (as integers)
val_labels = val_data.map(lambda x, y: y).unbatch() #extracting labels from val_data
actual_int = np.array(list(val_labels.as_numpy_iterator())) #converting labels to int

#mapping integers to labels
indoor_labels = np.array(indoor_dir) #converting indoor_dir to numpy array to map labels
pred_labels = np.array(indoor_labels)[pred_int]
actual_labels = np.array(indoor_labels)[actual_int]

#creating classification report
print(classification_report(pred_labels, actual_labels, target_names=indoor_labels))
