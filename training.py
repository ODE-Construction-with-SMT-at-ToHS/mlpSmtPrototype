import os, sys
import tensorflow as tf

from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' Simple function to train a simple MLP.
    Should receive an object with a static 
    function and a name attribute.'''

def train1d(funcClass):

    # Sample some training data.
    X_samples = np.linspace(-10, 10, 1000+1)
    y_samples = []
    for x in X_samples:
        y_samples.append(funcClass.f(x))

    X, y = shuffle(X_samples, y_samples, random_state=0)
    X_samples_shuffled = np.array(X)
    y_samples_shuffled = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X_samples_shuffled, y_samples_shuffled, test_size=0.2)

    # Create a simple mlp model.
    model = keras.Sequential([
        keras.layers.Dense(10, activation=tf.nn.relu, input_shape=[1]),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(5, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    # Use the MSE regression loss 
    optimizer = tf.keras.optimizers.RMSprop(0.01)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    
    # Train & save the model.
    model.fit(X_train, y_train, epochs = 500, validation_data=(X_test, y_test))
    model_filename = 'models/' + funcClass.name + '_model.h5'
    model.save(model_filename)

    # Plot the results.
    y_preds = model.predict(X_test)

    plt.plot(X_samples, y_samples, 'r')
    plt.scatter(X_test, y_preds)
    plt.savefig('plots/' + funcClass.name + '_learned.png')
    plt.clf()

''' Sample funtion to demonstrate how models trained to fit a
    specified funtion can be restored. '''

def openModel(funcClass):

    path = 'models/' + funcClass.name + '_model.h5'
    if not(os.path.isfile(path)):
        print('No model trained yet.')
        return

    model = keras.models.load_model(path)
    it = 0
    for layer in model.layers:
        it += 1
        weights = layer.get_weights()
        print('Layer ' + str(it) + ' weights:')
        print(weights)

if __name__ == '__main__':

    #train1d(linearA)
    #train1d(quadraticA)
    #train1d(quadraticB)

    openModel(quadraticA)
