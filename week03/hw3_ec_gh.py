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
model.add(keras.layers.Conv2D(5, 5, activation=activation,
                              strides=3))
model.add(keras.layers.MaxPool2D(4))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax', use_bias=False))

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
2018-09-22 21:53:26.773677: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
50000/50000 [==============================] - 6s 114us/step - loss: 1.6963 - acc: 0.4042 - val_loss: 1.0029 - val_acc: 0.7349
Epoch 2/10
50000/50000 [==============================] - 5s 108us/step - loss: 1.1864 - acc: 0.6001 - val_loss: 0.8231 - val_acc: 0.7835
Epoch 3/10
50000/50000 [==============================] - 5s 110us/step - loss: 1.1098 - acc: 0.6269 - val_loss: 0.7607 - val_acc: 0.7949
Epoch 4/10
50000/50000 [==============================] - 5s 109us/step - loss: 1.0720 - acc: 0.6413 - val_loss: 0.7385 - val_acc: 0.7979
Epoch 5/10
50000/50000 [==============================] - 5s 109us/step - loss: 1.0592 - acc: 0.6452 - val_loss: 0.7168 - val_acc: 0.7999
Epoch 6/10
50000/50000 [==============================] - 5s 110us/step - loss: 1.0444 - acc: 0.6505 - val_loss: 0.7102 - val_acc: 0.7992
Epoch 7/10
50000/50000 [==============================] - 6s 110us/step - loss: 1.0355 - acc: 0.6546 - val_loss: 0.6986 - val_acc: 0.8031
Epoch 8/10
50000/50000 [==============================] - 5s 109us/step - loss: 1.0228 - acc: 0.6568 - val_loss: 0.6891 - val_acc: 0.8042
Epoch 9/10
50000/50000 [==============================] - 6s 112us/step - loss: 1.0181 - acc: 0.6609 - val_loss: 0.6911 - val_acc: 0.8045
Epoch 10/10
50000/50000 [==============================] - 6s 114us/step - loss: 1.0247 - acc: 0.6604 - val_loss: 0.6926 - val_acc: 0.8085
10000/10000 [==============================] - 0s 49us/step
Test loss: 0.671688363457
Test accuracy: 0.8189
Number of parameters: 998.0
'''
