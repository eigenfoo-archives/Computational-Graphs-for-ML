'''
ECE471, Selecte Topics in Machine Learning - Assignment 4
Submit by Oct. 4, 10pm
tldr: Classify cifar10. Acheive performance similar to the state of the art.
Classify cifar100. Achieve a top-5 accuracy of 70%
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data


# https://github.com/philipperemy/tensorflow-maxout/blob/master/maxout.py
def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels,
                                                              num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs


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
    map(lambda y: tf.keras.utils.to_categorical(y, NUM_CLASSES),
        [y_train, y_val, y_test])

x_train_batches = np.split(x_train, 400)
y_train_batches = np.split(y_train, 400)

# Hyperparameters
NUM_EPOCHS = 1
REGULARIZER = None  # keras.regularizers.l2(0.0)

input = tf.placeholder(tf.float32, [None, 32, 32, 3])
labels = tf.placeholder(tf.float32, [None, 10])

x = tf.layers.conv2d(input, 64, 3, kernel_regularizer=REGULARIZER)
x = max_out(x, 2)

x = tf.layers.conv2d(x, 128, 3, kernel_regularizer=REGULARIZER)
x = max_out(x, 2)

x = tf.layers.conv2d(x, 256, 3, kernel_regularizer=REGULARIZER)
x = max_out(x, 2)

x = tf.layers.flatten(x)
x = tf.layers.dense(x, 128, activation=tf.nn.relu)
logits = tf.layers.dense(x, NUM_CLASSES)
output = tf.nn.softmax(logits)

loss = tf.losses.softmax_cross_entropy(labels, logits)
accuracy = tf.metrics.accuracy(labels, output)
train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(NUM_EPOCHS):
    for x_batch, y_batch in zip(x_train_batches, y_train_batches):
        _, loss_ = sess.run([train_step, loss],
                            feed_dict={input: x_batch, labels: y_batch})
        print(loss_)

# sess.run()
