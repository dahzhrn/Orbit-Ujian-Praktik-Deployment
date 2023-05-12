import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense 

from tensorflow.keras.applications.vgg19 import VGG19

base_model = VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3))

def make_model():
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(2048, activation="relu"))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(10, activation="softmax" , name="classification"))
    
    return model