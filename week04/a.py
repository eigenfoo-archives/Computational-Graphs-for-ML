'''
ECE471, Selecte Topics in Machine Learning - Assignment 4
Submit by Oct. 4, 10pm
tldr: Classify cifar10. Acheive performance similar to the state of the art.
Classify cifar100. Achieve a top-5 accuracy of 70%
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.cifar10 import load_data

# Fix seed
np.random.seed(1618)
tf.set_random_seed(1618)

# Data parameters
NUM_CLASSES = 10
HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3

# Load data and split into train, val, test sets
(x_train, y_train), (x_test, y_test) = load_data()
(x_train, y_train), (x_val, y_val) = \
    (x_train[:40000], y_train[:40000]), (x_train[40000:], y_train[40000:])

# Normalize and reshape data and labels
x_train, x_val, x_test = \
    map(lambda x: (x / 255.0).reshape([-1, HEIGHT, WIDTH, NUM_CHANNELS]),
        [x_train, x_val, x_test])
y_train, y_val, y_test = \
    map(lambda y: keras.utils.to_categorical(y, NUM_CLASSES),
        [y_train, y_val, y_test])


def conv_layer(filters, kernel_size, model=None):
    model.add(keras.layers.Conv2D(filters, kernel_size, padding='same'))
    model.add(keras.layers.MaxPool2D(2))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.BatchNormalization())


def dense_layer(units, model=None):
    model.add(keras.layers.Dense(units))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.BatchNormalization())


# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS_ADAM = 30
NUM_EPOCHS_SGD = 20

model = keras.Sequential()

conv_layer(64, 3, model=model)
conv_layer(128, 3, model=model)
conv_layer(256, 3, model=model)
conv_layer(512, 3, model=model)

model.add(keras.layers.Flatten())

dense_layer(2048, model=model)
dense_layer(1024, model=model)
dense_layer(512, model=model)
dense_layer(256, model=model)

model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS_ADAM,
          verbose=1,
          validation_data=[x_val, y_val])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS_SGD,
          verbose=1,
          validation_data=[x_val, y_val])

loss, acc = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', loss)
print('Test accuracy:', acc)
