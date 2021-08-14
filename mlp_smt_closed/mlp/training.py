"""
This module can be used to create an MLP, train it  a function `f`, save it, and plot the error. So far,
`f` :math:`\in \mathbb{R} \\rightarrow \mathbb{R}` or `f` :math:`\in \mathbb{R}^2 \\rightarrow \mathbb{R}^2` is
supported
"""
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from mlp_smt_closed.mlp.functions import *
from mlp_smt_closed.mlp.plot3d import *
from mlp_smt_closed.mlp.sampling import *


def train1d(func_class):
    """Create an MLP, train it on a function `f` :math:`\in \mathbb{R} \\rightarrow \mathbb{R}` contained in
    `func_class`, save it, and plot the error

    Args:
        func_class:
            a class which must contain a static function `f` :math:`\in \mathbb{R} \\rightarrow \mathbb{R}` and a `name`
            attribute
    """

    # load samples
    x_samples, y_samples, x_train, x_test, y_train, y_test = load_samples(func_class)

    # Create a simple mlp model.
    model = keras.Sequential([
        keras.layers.Dense(5, activation=tf.nn.relu, input_shape=(1,)),
        keras.layers.Dense(5, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    # Use the MSE regression loss (learning rate !?)
    optimizer = tf.keras.optimizers.RMSprop(0.01)

    # configure model
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    
    # Train, validate & save the model.
    model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
    model_filename = 'models/' + func_class.name + '_model.h5'
    model.save(model_filename)


    # calculate predictions
    y_predictions = model.predict(x_test)

    # Plot the results
    plt.plot(x_samples, y_samples, 'r')
    plt.scatter(x_test, y_predictions)
    plt.savefig('plots/' + func_class.name + '_learned.png')
    plt.show()
    plt.clf()


def train2d(func_class):
    """Create an MLP, train it on a function `f` :math:`\in \mathbb{R}^2 \\rightarrow \mathbb{R}^2` contained in
    `func_class`, save it, and plot the error

    Args:
        func_class:
            a class which must contain a static function `f` :math:`\in \mathbb{R}^2 \\rightarrow \mathbb{R}^2` and a
            `name` attribute
    """

    # load samples
    x_samples, y_samples, x_train, x_test, y_train, y_test = load_samples(func_class)

    # Create a simple mlp model.
    model = keras.Sequential([
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.relu),
        keras.layers.Dense(2)
    ])

    # Use the MSE regression loss (learning rate!?)
    optimizer = tf.keras.optimizers.RMSprop(0.01)

    # configure model
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    # Train, validate & save the model.
    model.fit(x_train, y_train, epochs=400, validation_data=(x_test, y_test))
    model_filename = 'models/' + func_class.name + '_model.h5'
    model.save(model_filename)

    # calculate predictions
    y_prediction = model.predict(x_test)

    # plot distances of y_test and y_prediction
    plot_dist_map(x_test, y_test, y_prediction, func_class.name)


def open_model(func_class):
    """
    Print the saved model/parameters of an MLP trained on `func_class` if the model was trained on `func_class` already

    Args:
        func_class: function for which the model should be printed
    """

    # get path to model
    path = 'models/' + func_class.name + '_model.h5'
    if not(os.path.isfile(path)):
        print('No model trained yet.')
        return

    # load the model
    model = keras.models.load_model(path)

    # iterate over layers to print them
    it = 0
    for layer in model.layers:
        it += 1
        weights = layer.get_weights()
        print('Layer ' + str(it) + ' weights:')
        print(weights)


if __name__ == '__main__':
    # sample_1d(LinearA, 1000+1, [-10, 10], 0.2)
    # sample_1d(QuadraticA, 1000+1, [-10, 10], 0.2)
    # sample_1d(QuadraticB, 1000+1, [-10, 10], 0.2)
    # sample_2d(LinearA2D, 100, 100, [-10, 10], [-10, 10], 0.05)
    # train1d(LinearA)
    # train1d(QuadraticA)
    # train1d(QuadraticB)
    # train2d(LinearA2D)
    # train1d(LinearA)
    x_samples, y_samples, x_train, x_test, y_train, y_test = load_samples(LinearA)
    print(x_samples, y_samples, x_train, x_test, y_train, y_test)
    print(x_samples.shape)

    # open_model(LinearA)
