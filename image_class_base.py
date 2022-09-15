# -*- coding: utf-8 -*-
"""
Created on Wed May 30 22:51:01 2018

@author: Tarunbir Singh
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from PIL import Image

# Step 1 - Data collection
        
batch_size = 50

train_path = 'run1/train'
test_path = 'run1/test'
valid_path = 'run1/validation'

# this is the augmentation configuration we will use for our training 
train_batch = ImageDataGenerator().flow_from_directory(train_path, target_size=(150,150),classes=['dog','cat'] , batch_size=batch_size)
valid_batch = ImageDataGenerator().flow_from_directory(valid_path, target_size=(150,150),classes=['dog','cat'] , batch_size=batch_size)
test_batch = ImageDataGenerator().flow_from_directory(test_path, target_size=(150,150),classes=['dog','cat'] , batch_size=132)

#Step 2 - Build Model
        
model = Sequential()

# adding convolutional layers
# classic CNN model : input->Conv->ReLU->conv->ReLU->Pool->ReLU->Conv->ReLU->Pool->Fully Connected

model.add(Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
model.add(Flatten()) #turns negative values to zero

model.add(Dense(2,activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',          # performs gradient descent
              metrics=['accuracy'])

# Step 3 - Training the model

model.fit_generator(
        train_batch,
        steps_per_epoch=2000 // batch_size,
        epochs=5,
        validation_data=valid_batch,
        validation_steps=800 // batch_size,
        verbose=1)

#model.save_weights('first_try.h5') # always save your weights after training or during training

# predicting test batch
pred = model.predict_generator(test_batch)

print('Test accuracy: {:.2f} %'.format(np.mean(np.equal(pred,next(test_batch)[1]))))
