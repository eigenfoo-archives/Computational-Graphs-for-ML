'''
ECE471 Selected Topics in Machine Learning - Assignment 2
Submit by Sept. 19, 10PM
tldr: Perform binary classification on the spirals dataset using a multi-layer
perceptron. You must generate the data yourself.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical

# Make data
t = np.linspace(1, 20, 100)
r = t
theta = 0.8*t

x1 = r*np.cos(theta) + np.random.normal(0, 0.3, 100)
y1 = r*np.sin(theta) + np.random.normal(0, 0.3, 100)
x2 = r*np.cos(theta + np.pi) + np.random.normal(0, 0.3, 100)
y2 = r*np.sin(theta + np.pi) + np.random.normal(0, 0.3, 100)

data = np.concatenate([np.vstack([x1, y1]).T,
                       np.vstack([x2, y2]).T])
labels = to_categorical(np.append(np.zeros(100), np.ones(100)))

data, labels = shuffle(data, labels)

# Multi-layered perceptron
model = keras.Sequential()
model.add(keras.layers.Dense(2, activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

model.fit(data, labels, epochs=100, batch_size=32)

# Plot
plt.scatter(x1, y1, c='r')
plt.scatter(x2, y2, c='b')

plt.title('Spirals')
plt.axis('equal')

#plt.show()
