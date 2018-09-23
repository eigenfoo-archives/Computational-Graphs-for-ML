'''
ECE471 Selected Topics in Machine Learning - Assignment 2
Submit by Sept. 26, 10pm
tldr: Classify mnist digits with a convolutional neural network. Get at least
95.5% accuracy on the test test.
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.mnist import load_data

# Fix seed
np.random.seed(1618)
tf.set_random_seed(1618)

# Data parameters
NUM_CLASSES = 10
HEIGHT = 28
WIDTH = 28
NUM_CHANNELS = 1

# Load data and split into train, val, test sets
(x_train, y_train), (x_test, y_test) = load_data()
(x_train, y_train), (x_val, y_val) = \
    (x_train[:50000], y_train[:50000]), (x_train[50000:], y_train[50000:])

# Normalize and reshape data and labels
x_train, x_val, x_test = \
    map(lambda x: (x / 255.0).reshape([-1, HEIGHT, WIDTH, NUM_CHANNELS]),
        [x_train, x_val, x_test])
y_train, y_val, y_test = \
    map(lambda y: keras.utils.to_categorical(y, NUM_CLASSES),
        [y_train, y_val, y_test])

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Create CNN using Keras API
activation = keras.activations.relu

model = keras.Sequential()
model.add(keras.layers.Conv2D(5, 14, activation=activation,
                              strides=4))
model.add(keras.layers.MaxPool2D(3))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          verbose=1,
          validation_data=[x_val, y_val])

loss, acc = model.evaluate(x_test, y_test, verbose=1)
num_params = np.sum([np.prod(v.get_shape().as_list())
                     for v in tf.trainable_variables()])

print('Test loss:', loss)
print('Test accuracy:', acc)
print('Number of parameters:', num_params)

''' Output
Train on 50000 samples, validate on 10000 samples
Epoch 1/10
2018-09-22 20:50:51.831604: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
50000/50000 [==============================] - 6s 114us/step - loss: 1.6335 - acc: 0.4260 - val_loss: 1.1370 - val_acc: 0.7064
Epoch 2/10
50000/50000 [==============================] - 6s 112us/step - loss: 1.3468 - acc: 0.5217 - val_loss: 0.9773 - val_acc: 0.7578
Epoch 3/10
50000/50000 [==============================] - 6s 112us/step - loss: 1.2587 - acc: 0.5592 - val_loss: 0.8627 - val_acc: 0.7916
Epoch 4/10
50000/50000 [==============================] - 6s 112us/step - loss: 1.2032 - acc: 0.5790 - val_loss: 0.8156 - val_acc: 0.7997
Epoch 5/10
50000/50000 [==============================] - 6s 110us/step - loss: 1.1760 - acc: 0.5913 - val_loss: 0.8024 - val_acc: 0.7997
Epoch 6/10
50000/50000 [==============================] - 6s 110us/step - loss: 1.1646 - acc: 0.5922 - val_loss: 0.7703 - val_acc: 0.8126
Epoch 7/10
50000/50000 [==============================] - 6s 110us/step - loss: 1.1585 - acc: 0.5962 - val_loss: 0.7569 - val_acc: 0.8145
Epoch 8/10
50000/50000 [==============================] - 6s 112us/step - loss: 1.1410 - acc: 0.6024 - val_loss: 0.7505 - val_acc: 0.8169
Epoch 9/10
50000/50000 [==============================] - 6s 111us/step - loss: 1.1377 - acc: 0.6040 - val_loss: 0.7424 - val_acc: 0.8157
Epoch 10/10
50000/50000 [==============================] - 6s 111us/step - loss: 1.1364 - acc: 0.6068 - val_loss: 0.7325 - val_acc: 0.8179
10000/10000 [==============================] - 0s 35us/step
Test loss: 0.734625018501
Test accuracy: 0.8101
Number of parameters: 3144.0
'''
