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

# Downsample images
x_train, x_val, x_test = \
    map(lambda x: x[:, ::2, ::2, :], [x_train, x_val, x_test])

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 8

# Create CNN using Keras API
activation = keras.activations.relu

model = keras.Sequential()
model.add(keras.layers.Conv2D(3, 5, activation=activation))
model.add(keras.layers.MaxPool2D(5))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax',
                             use_bias=False))

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
Epoch 1/8
2018-09-22 23:07:55.242510: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
50000/50000 [==============================] - 8s 169us/step - loss: 1.8248 - acc: 0.3619 - val_loss: 1.1677 - val_acc: 0.7233
Epoch 2/8
50000/50000 [==============================] - 8s 169us/step - loss: 1.3431 - acc: 0.5451 - val_loss: 0.9330 - val_acc: 0.7678
Epoch 3/8
50000/50000 [==============================] - 7s 144us/step - loss: 1.2295 - acc: 0.5837 - val_loss: 0.8394 - val_acc: 0.7876
Epoch 4/8
50000/50000 [==============================] - 7s 137us/step - loss: 1.1758 - acc: 0.6036 - val_loss: 0.7799 - val_acc: 0.7997
Epoch 5/8
50000/50000 [==============================] - 7s 136us/step - loss: 1.1485 - acc: 0.6116 - val_loss: 0.7594 - val_acc: 0.8093
Epoch 6/8
50000/50000 [==============================] - 7s 140us/step - loss: 1.1237 - acc: 0.6192 - val_loss: 0.7379 - val_acc: 0.8150
Epoch 7/8
50000/50000 [==============================] - 7s 141us/step - loss: 1.1188 - acc: 0.6210 - val_loss: 0.7259 - val_acc: 0.8218
Epoch 8/8
50000/50000 [==============================] - 7s 140us/step - loss: 1.1012 - acc: 0.6292 - val_loss: 0.7054 - val_acc: 0.8185
10000/10000 [==============================] - 1s 57us/step
Test loss: 0.694395569706
Test accuracy: 0.8176
Number of parameters: 602.0
'''
