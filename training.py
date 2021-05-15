import os

import numpy
import tensorflow as tf

from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from functions import *
from plot3d import *

import numpy as np
import matplotlib.pyplot as plt

def train1d(func_class):
    ''' Simple function to train a simple MLP.
    Should receive an object with a static 
    function and a name attribute.
    '''

    # Sample some training data.
    x_samples = np.linspace(-10, 10, 1000+1).T
    y_samples = []
    for x in x_samples:
        y_samples.append(func_class.f(x))
    y_samples = np.array(y_samples)

    # process samples, get test training data
    x, y = shuffle(x_samples, y_samples, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Create a simple mlp model.
    model = keras.Sequential([
        keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(1,)),
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
    
    # Train, validate & save the model.
    model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
    model_filename = 'models/' + func_class.name + '_model.h5'
    model.save(model_filename)

    # Plot the results.
    y_predictions = model.predict(x_test)

    plt.plot(x_samples, y_samples, 'r')
    plt.scatter(x_test, y_predictions)
    plt.savefig('plots/' + func_class.name + '_learned.png')
    plt.clf()

def open_model(func_class):
    ''' Sample function to demonstrate how models trained to fit a
    specified function can be restored.
    '''

    path = 'models/' + func_class.name + '_model.h5'
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

def train2d(func_class):

    # Sample 2D training data.
    a, b = np.mgrid[-10:10:100j, -10:10:100j]
    x_samples = np.vstack((a.flatten(), b.flatten())).T
    y_samples = [func_class.f(x) for x in x_samples]
    y_samples = np.array(y_samples)

    # process samples, get test training data
    x, y = shuffle(x_samples, y_samples, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Create a simple mlp model.
    model = keras.Sequential([
        keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense( 5, activation=tf.nn.relu),
        keras.layers.Dense( 2)
    ])

    # Use the MSE regression loss
    optimizer = tf.keras.optimizers.RMSprop(0.01)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    # Train, validate & save the model.
    model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test))
    model_filename = 'models/' + func_class.name + '_model.h5'
    model.save(model_filename)

    # Plot the results.
    y_prediction = model.predict(x_test)

    # plot distances of y_test and y_prediction
    plot_dist_map(x_test, y_test, y_prediction, func_class.name)

if __name__ == '__main__':

    # train2d(LinearA2D)
    train1d(LinearA)
    # train1d(QuadraticA)
    # train1d(QuadraticB)

    open_model(LinearA)
