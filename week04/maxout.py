'''
ECE471, Selected Topics in Machine Learning - Assignment 4
Submit by Oct. 4, 10pm
tldr: Classify cifar10. Acheive performance similar to the state of the art.
Classify cifar100. Achieve a top-5 accuracy of 70%
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.cifar100 import load_data
from tqdm import tqdm


# Adapted from
# https://github.com/philipperemy/tensorflow-maxout/blob/master/maxout.py
def max_out(inputs):
    shape = inputs.get_shape().as_list()
    shape[0] = -1
    shape[-1] = shape[-1] // 2
    shape += [2]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs


# Fix seed
np.random.seed(1618)
tf.set_random_seed(1618)

# Data parameters
NUM_CLASSES = 100
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

# This means batch size is 40000 / 400 = 100
x_train_batches = np.split(x_train, 400)
y_train_batches = np.split(y_train, 400)

# Hyperparameters
NUM_EPOCHS = 15
REGULARIZER = tf.keras.regularizers.l2(1e-7)

input = tf.placeholder(tf.float32, [None, 32, 32, 3])
labels = tf.placeholder(tf.float32, [None, NUM_CLASSES])

x = tf.layers.conv2d(input, 128, 3, kernel_regularizer=REGULARIZER)
x = max_out(x)

x = tf.layers.conv2d(x, 256, 3, kernel_regularizer=REGULARIZER)
x = max_out(x)

x = tf.layers.conv2d(x, 512, 3, kernel_regularizer=REGULARIZER)
x = max_out(x)

x = tf.layers.conv2d(x, 1024, 3, kernel_regularizer=REGULARIZER)
x = max_out(x)

x = tf.layers.flatten(x)
x = tf.layers.dense(x, 256, activation=max_out)
logits = tf.layers.dense(x, NUM_CLASSES)
output = tf.nn.softmax(logits)

# Metrics
accuracy = tf.keras.metrics.categorical_accuracy(labels, output)
top5_accuracy = tf.keras.metrics.top_k_categorical_accuracy(labels,
                                                            output, k=5)

# Loss
loss = tf.losses.softmax_cross_entropy(labels, logits)
train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

LOGFILE = '100metrics.txt'
saver = tf.train.Saver()

for i in range(NUM_EPOCHS):
    for j, x_batch, y_batch in tqdm(enumerate(zip(x_train_batches,
                                                  y_train_batches))):
        if j % 40 == 0:
            _, accuracy_, top5_accuracy = \
                sess.run([train_step, accuracy, top5_accuracy],
                         feed_dict={input: x_batch, labels: y_batch})
            with open(LOGFILE, 'a') as f:
                f.write('Minibatch: {}\t'.format(j))
                f.write('Train Loss: {}\t'.format(loss_))
                f.write('Train Accuracy: {}\t'.format(accuracy_))
                f.write('Train Top5 Accuracy: {}\n'.format(top5_accuracy_))
        else:
            sess.run(train_step, feed_dict={input: x_batch, labels: y_batch})

    loss_, accuracy_, top5_accuracy_ = \
        sess.run([loss, accuracy, top5_accuracy],
                 feed_dict={input: x_val, labels: y_val})

    with open(LOGFILE, 'a') as f:
        f.write('Epoch: {}\t'.format(i))
        f.write('Loss: {}\t'.format(loss_))
        f.write('Accuracy: {}\t'.format(accuracy_))
        f.write('Top5 Accuracy: {}\n'.format(top5_accuracy_))
        f.write(80*'=')
        f.write('\n')

    save_path = saver.save(sess, "./tmp/100model{}.ckpt".format(i))
