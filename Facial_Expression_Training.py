#!pip install -q livelossplot

""" INTRODUCTION OF LIBRARIES"""
#first of all import all the necessary libraries which are going to be used further.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
%matplotlib inline

"""EXPLORE THE DATASET"""
#it will load the images into the platform, the dataset of all the images
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_size = 48   #the data contains only 48x48 px images
plt.figure(0, figsize=(12,20))
ctr = 0

#in the training dataset, it will show us all the images of all the classes randomly.
for expression in os.listdir("train/"): 
    for i in range(1,6):
        ctr += 1
        plt.subplot(7,5,ctr)
        img = load_img("train/" + expression + "/" +os.listdir("train/" 
                                                    + expression)[i], target_size=(img_size, img_size))
        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show()  #show the images in a subplot of (7,5)


#this will show us that how many images are there in a particular class 
#ie. how many images are there in the angry class, sad class, happy class, etc.
for expression in os.listdir("train/"):
    print(str(len(os.listdir("train/" + expression))) + " " + expression + " images")
    

import tensorflow as tf
tf.config.list_physical_devices('GPU')


"""GENERATE TRAINING AND VALIDATION BATCHES"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 64 #we choose 64 as a batch size so it will efficiently increase the training process

#datagen_train = ImageDataGenerator()
#image generator for the images of training set and randomly flip all the images along the horizontal axis
datagen_train = ImageDataGenerator(horizontal_flip=True)    

#it is going to take in the batches of images from the directory that we specify
train_generator = datagen_train.flow_from_directory("train/",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)
#same case will be applied to validation set and it will take the batches of images from the directory that we specify
datagen_validation = ImageDataGenerator()
validation_generator = datagen_validation.flow_from_directory("test/",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)
#it will return how many images are there in the training and test set of 7 classes. 
# Considering the dataset, 28709 images are from training set and 7178 images from validation set


from tensorflow.keras.layers import Activation, Convolution2D, Dropout, Conv2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Input, MaxPooling2D, SeparableConv2D
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Sequential, Model


"""CREATE A CONVOLUTIONAL NEURAL NETWORK"""
def big_XCEPTION(input_shape, num_classes):
    # module 1 (base)
    img_input = Input(input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)   #batch normalization works well with CNN
    x = Activation('relu', name='block1_conv1_act')(x)  #use the activation function relu for non-linearity
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)
    x = Dropout(0.3)(x)     #we use dropout to prevent the overfitting of the training data
    
    # module 2
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)    #padding should be same so that we don't lose the information about images
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = Dropout(0.3)(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)  #it effectively shrinks the height and width dimentions by a factor of 3
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = Dropout(0.3)(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = Dropout(0.3)(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)   #the output can be predicted according to the probabilitic scores,
                                                            #that's why we use activation function as softmax function

    model = Model(img_input, output)
    return model 

model = big_XCEPTION((48, 48, 1), 7)
optimizer = Adamax()    #the Adam can be used as an optimizer

#compilation of the model, since we are working on a multiclass classification, it should be categorical_crosentropy 
# and accuracy can be calculated by the matrix used as a list
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


"""TRAINING AND EVALUATION OF THE MODEL"""
from IPython.display import SVG, Image
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
Image('model.png')

%%time

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

#reduce the learning rate when we observe a plateau in the validation loss if we dont see any improvement 
# in the validation loss after certain number of epochs, we can reduce the learning rate
#reduce the learning rate by a factor of 0.1
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=4, min_lr=0.0001, mode='auto')

epochs = 30     #no. of iterations

#we have to find the steps required for training and validation to carry out the training processes.
steps_per_epoch = train_generator.n//train_generator.batch_size     #no. of examples in our training generator // batch_size
validation_steps = validation_generator.n//validation_generator.batch_size  #no. of examples in our test generator // batch_size

#define callbacks

#save the model with highest accuracy when we reach at a particular epoch in the model as we keep 
# training, we are going to save the weights to this once we have found the model with higgest accuracy
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', save_weights_only=True, mode='max', verbose=0)  
callbacks = [checkpoint, reduce_lr]

#fit the training model
history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    callbacks=callbacks
)

print(history.history.keys())

"""SUMMARIZATION OF ACCURACY"""

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for learning rate
plt.plot(history.history['lr'])
plt.title('Learning Rate')
plt.ylabel('learning rate')
plt.xlabel('epoch')
plt.legend(['learning rate'], loc='upper left')
plt.show()
#this will return the log-loss function (cost function) graph and accuracy graph of the model

"""REPRESENT THE MODEL AS JSON STRING"""
model_json = model.to_json()
with open("model.json", "w") as json_file:  #architecture to a JSON file
    json_file.write(model_json)