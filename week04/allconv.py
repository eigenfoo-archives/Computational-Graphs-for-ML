'''
ECE471, Selected Topics in Machine Learning - Assignment 4
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

# Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS_ADAM = 20
NUM_EPOCHS_SGD = 30

model = keras.Sequential()

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(96, 3, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(1e-3)))
model.add(keras.layers.Conv2D(96, 3, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(1e-3)))
model.add(keras.layers.Conv2D(96, 3, activation='relu', strides=2,
                              kernel_regularizer=keras.regularizers.l2(1e-3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Conv2D(192, 3, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(1e-3)))
model.add(keras.layers.Conv2D(192, 3, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(1e-3)))
model.add(keras.layers.Conv2D(192, 3, activation='relu', strides=2,
                              kernel_regularizer=keras.regularizers.l2(1e-3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Conv2D(192, 3, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(1e-3)))
model.add(keras.layers.Conv2D(192, 1, activation='relu',
                              kernel_regularizer=keras.regularizers.l2(1e-3)))
model.add(keras.layers.Conv2D(  10, 1, activation='relu', strides=2,
                              kernel_regularizer=keras.regularizers.l2(1e-3)))

model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(NUM_CLASSES, activation=keras.activations.softmax))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS_ADAM,
          verbose=1,
          validation_data=[x_val, y_val])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(momentum=0.9),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS_SGD,
          verbose=1,
          validation_data=[x_val, y_val])

loss, acc, top5_acc = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', loss)
print('Test accuracy:', acc)
print('Test top5 accuracy:', top5_acc)
