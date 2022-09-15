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
import pandas as pd
# Step 1 - Data collection
        
batch_size = 10

dataDir = 'run2'

train_path = dataDir+'/train'
test_path = dataDir+'/test'
valid_path = dataDir+'/validation'

# this is the augmentation configuration we will use for our training 
train_batch = ImageDataGenerator(rescale=1./255).flow_from_directory(train_path, target_size=(150,150),class_mode='binary' , batch_size=batch_size)
valid_batch = ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path, target_size=(150,150),class_mode='binary' , batch_size=batch_size)
test_batch = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path, target_size=(150,150),class_mode='binary' , batch_size=440)

#Step 2 - Build Model
        
model = Sequential()

# adding convolutional layers
# classic CNN model : input->Conv->ReLU->conv->ReLU->Pool->ReLU->Conv->ReLU->Pool->Fully Connected

model.add(Conv2D(32,(3,3),input_shape=(150,150,3)))
model.add(Activation('relu')) #turns negative values to zero
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu')) #turns negative values to zero
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu')) #turns negative values to zero
model.add(MaxPooling2D(pool_size=(2,2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu')) #turns negative values to zero
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',          # performs gradient descent
              metrics=['accuracy'])

# Step 3 - Training the model

model.fit_generator(train_batch, epochs=50, validation_data=valid_batch, verbose=1)

# model.save_weights('first_try.h5') # always save your weights after training or during training


# predicting test batch
X,y = next(test_batch)
pred = model.predict(X)
pred = np.array([int(np.squeeze(pred>0.5)[i]) for i in range(len(pred))])

print('\nTest accuracy: {:.2f} %\n'.format(100*np.mean(np.equal(pred,y))))

cm = pd.crosstab(y,pred, margins=True, rownames=['True'], colnames=['Predicted'])
    
print("***** Confusion Matrix *****\n",cm)