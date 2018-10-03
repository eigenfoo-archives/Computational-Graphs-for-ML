import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

datagen_train = ImageDataGenerator(
    featurewise_center=True,
    zca_whitening=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1)
datagen_train.fit(x_train)

datagen_val = ImageDataGenerator(
    featurewise_center=True,
    zca_whitening=True)
datagen_val.fit(x_val)

datagen_test = ImageDataGenerator(
    featurewise_center=True,
    zca_whitening=True)
datagen_test.fit(x_test)

# Normalize and reshape data and labels
x_train, x_val, x_test = \
    map(lambda x: x.reshape([-1, HEIGHT, WIDTH, NUM_CHANNELS]),
        [x_train, x_val, x_test])
y_train, y_val, y_test = \
    map(lambda y: keras.utils.to_categorical(y, NUM_CLASSES),
        [y_train, y_val, y_test])

# Adapted from
# https://github.com/philipperemy/tensorflow-maxout/blob/master/maxout.py
def maxout(inputs):
    shape = inputs.get_shape().as_list()
    shape[0] = -1
    shape[-1] = shape[-1] // 2
    shape += [2]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs


def conv_layer(filters, kernel_size, maxpool=False, dropout=False, model=None):
    model.add(keras.layers.Conv2D(filters, kernel_size,
                                  padding='same', activation=maxout,
                                  kernel_regularizer=tf.keras.regularizers.l2(5e-7)))
    if maxpool:
        model.add(keras.layers.MaxPool2D(maxpool))
    if dropout:
        model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())


# Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS_ADAM = 20
NUM_EPOCHS_SGD = 30

model = keras.Sequential()

model.add(keras.layers.Conv2D(96, 5,
                              padding='same', activation=maxout,
                              kernel_regularizer=tf.keras.regularizers.l2(5e-7),
                              input_shape=[32, 32, 3]))
conv_layer(96, 5, model=model)
conv_layer(96, 5, maxpool=2, dropout=True, model=model)
conv_layer(192, 3, model=model)
conv_layer(192, 3, model=model)
conv_layer(192, 3, maxpool=2, dropout=True, model=model)
conv_layer(192, 3, model=model)
conv_layer(192, 1, model=model)
conv_layer(10, 1, model=model)

model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(NUM_CLASSES, activation=keras.activations.softmax))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

model.fit_generator(datagen_train.flow(x_train, y_train,
                                       batch_size=BATCH_SIZE),
                    epochs=NUM_EPOCHS_ADAM,
                    verbose=1,
                    validation_data=datagen_val.flow(x_val, y_val,
                                                     batch_size=BATCH_SIZE))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,
                                             momentum=True),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

model.fit_generator(datagen_train.flow(x_train, y_train,
                                       batch_size=BATCH_SIZE),
                    epochs=NUM_EPOCHS_SGD,
                    verbose=1,
                    validation_data=datagen_val.flow(x_val, y_val,
                                                     batch_size=BATCH_SIZE))

loss, acc, top5_acc = \
    model.evaluate_generator(datagen_test.flow(x_test, y_test,
                                               batch_size=BATCH_SIZE),
                             verbose=1)

print('Test loss:', loss)
print('Test accuracy:', acc)
print('Test top5 accuracy:', top5_acc)
