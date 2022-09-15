# -*- coding: utf-8 -*-
"""
Created on Wed May 30 22:51:01 2018

@author: Tarunbir Singh
"""
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
# Step 1 - Data collection
        
batch_size = 10

train_path = 'run1/train'
test_path = 'run1/test'
valid_path = 'run1/validation'

# this is the augmentation configuration we will use for our training 
train_batch = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['dog','cat'] , batch_size=batch_size)
valid_batch = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),classes=['dog','cat'] , batch_size=batch_size)
test_batch = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),classes=['dog','cat'] , batch_size=40)

#Step 2 - Getting keras model from vgg16
        
vgg16 = keras.applications.vgg16.VGG16()

vgg16.summary()
# classic CNN model : input->Conv->ReLU->conv->ReLU->Pool->ReLU->Conv->ReLU->Pool->Fully Connected

#Step 3 - changing the model object(vgg16) to sequential object(model)

model = Sequential()

for layer in vgg16.layers:
    model.add(layer)

# removing the last output layer with 1000 categories
model.layers.pop()

# fixing the pretrained vgg16 parameter so that they arent changed while training
for layer in model.layers:
    layer.trainable = False

# adding our custom required outplut layer
model.add(Dense(2,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',          # performs gradient descent
              metrics=['accuracy'])

# Step 3 - Training the model

model.fit_generator(
        train_batch,
        steps_per_epoch=120 // batch_size,
        epochs=5,
        validation_data=valid_batch,
        validation_steps=40 // batch_size,
        verbose=1)

#model.save_weights('first_try.h5') # always save your weights after training or during training

# predicting test batch
X,y = next(test_batch)
pred = model.predict(X)

print('\nTest accuracy: {:.2f} %\n'.format(100*np.mean(np.equal(np.argmax(pred,1),np.argmax(y,1)))))

cm = pd.crosstab(np.argmax(y,1),np.argmax(pred,1), margins=True, rownames=['True'], colnames=['Predicted'])
    
print("***** Confusion Matrix *****\n",cm)
