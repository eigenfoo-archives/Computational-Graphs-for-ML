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
REGULARIZER = keras.regularizers.l2(l=1e-7)


def resnext_module(input, output_dim, bottleneck_dim=16, cardinality=16):
    '''
    Adds a ResNeXt bottleneck module.

    Parameters
    ----------
    input : tf.Tensor
        Input tensor.
    cardinality : int
        Cardinality.
    bottleneck_dim : int
        Number of channels that form the so-called bottleneck.
    output_dim : int
        Number of channels that form the output.

    Returns
    -------
    x : tf.Tensor
        Output tensor.
    '''

    xfm_list = []

    for c in range(cardinality):
        x = keras.layers.Conv2D(bottleneck_dim, 1, padding='same',
                                kernel_regularizer=REGULARIZER)(input)
        x = keras.layers.Conv2D(bottleneck_dim, 3, padding='same',
                                kernel_regularizer=REGULARIZER)(x)
        x = keras.layers.Conv2D(output_dim, 1, padding='same',
                                kernel_regularizer=REGULARIZER)(x)
        xfm_list.append(x)

    x = keras.layers.add(xfm_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # "The shortcuts are identity connections except for those increasing
    # dimensions which are projections."
    if input.get_shape().as_list()[-1] != output_dim:
        projection = keras.layers.Dense(output_dim, use_bias=False,
                                        kernel_regularizer=REGULARIZER)(input)
        x = keras.layers.add([x, projection])
    else:
        x = keras.layers.add([x, input])

    return x


input = keras.layers.Input(shape=[32, 32, 3])

# Initial convolutional and maxpool layer
x = keras.layers.Conv2D(64, 3, padding='same',
                        kernel_regularizer=REGULARIZER)(input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)

x = resnext_module(x, 128)
x = resnext_module(x, 128)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Dropout(0.5)(x)
x = resnext_module(x, 256)
x = resnext_module(x, 256)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Dropout(0.5)(x)
x = resnext_module(x, 256)
x = resnext_module(x, 256)

x = keras.layers.GlobalAveragePooling2D()(x)

x = keras.layers.Dense(256, activation='relu')(x)
output = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.models.Model(input, output)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS_ADAM,
          verbose=1,
          validation_data=[x_val, y_val])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS_SGD,
          verbose=1,
          validation_data=[x_val, y_val])

loss, acc, top_k_acc = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', loss)
print('Test accuracy:', acc)
print('Test top5 accuracy:', top_k_acc)

model.save('resnext.h5')
