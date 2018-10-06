'''
ECE471, Selected Topics in Machine Learning - Assignment 5
Submit by Oct. 10, 10pm.
tldr: Classify the AG News dataset posted on the course website. Achieve an
accuracy similar to the state of the art.
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Hyperparameters
EMBEDDING_DIM = 8
BATCH_SIZE = 32
NUM_EPOCHS = 1


def load_data():
    train = pd.read_csv('ag_news_csv/train.csv',
                        names=['label', 'headline', 'blurb'])
    test = pd.read_csv('ag_news_csv/test.csv',
                       names=['label', 'headline', 'blurb'])

    train['text'] = train['headline'] + train['blurb']
    test['text'] = test['headline'] + test['blurb']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train.text)
    vocab_size = len(tokenizer.word_index)

    x_train = pad_sequences(tokenizer.texts_to_sequences(train.text))
    # Data 1-indexes, Keras 0-indexes
    y_train = to_categorical(train.label - 1)
    _, sequence_length = x_train.shape

    x_test = pad_sequences(tokenizer.texts_to_sequences(test.text),
                           maxlen=sequence_length)
    # Data 1-indexes, Keras 0-indexes
    y_test = to_categorical(test.label - 1)

    return (x_train, y_train), (x_test, y_test), vocab_size, sequence_length


(x_train, y_train), (x_test, y_test), vocab_size, sequence_length = load_data()

model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size+1, EMBEDDING_DIM,
                                 input_length=sequence_length))

model.add(keras.layers.Reshape([sequence_length*EMBEDDING_DIM, 1]))

model.add(keras.layers.Conv1D(16, EMBEDDING_DIM, dilation_rate=1,
                              activation='relu'))
model.add(keras.layers.MaxPool1D(2))
model.add(keras.layers.Conv1D(32, EMBEDDING_DIM, dilation_rate=2,
                              activation='relu'))
model.add(keras.layers.MaxPool1D(2))
model.add(keras.layers.Conv1D(64, EMBEDDING_DIM, dilation_rate=3,
                              activation='relu'))
model.add(keras.layers.MaxPool1D(2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          verbose=1,
          validation_data=[x_test, y_test])
