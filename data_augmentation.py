# -*- coding: utf-8 -*-
"""
Created on Thu May 31 00:01:22 2018

@author: Tarunbir Singh
"""

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Data pre-processing and Data augmentation - random transformations,
# so that our model would never see twice the exact same picture
datagen = ImageDataGenerator(
        rotation_range = 40,        # range in which randomly rotate pictures
        width_shift_range = 0.2,    # range within which to randomly translate 
        height_shift_range = 0.2,   # pictures vertically or horizontally
        rescale = 1./255,           # scale down 0-255 to 0-1
        shear_range = 0.2,          # randomly applying shearing tranformations
        zoom_range = 0.2,           # randomly zooming inside pictures
        horizontal_flip = True,     # random horizontal flipping, real world pictures- no symmetry
        fill_mode = 'nearest')      # filling newly created pixels after rotation or width/height shift

img = load_img('data/cats/cat.3.jpg') # this is a PIL image
x = img_to_array(img) # this is a numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape) # this is a numpy array with shape (1, 3, 150, 150)

# the .flow commad below generates batches of randomly transformed images
# and saves the results to the folder 'preview/' directory
i=0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i>20:
        break # otherwise the generator would loop indefinitely