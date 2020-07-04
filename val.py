# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:38:15 2020

@author: vivek vivian
"""

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
model = load_model('model_lungs.h5')
img = image.load_img('val/NORMAL/NORMAL2-IM-1436-0001.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)