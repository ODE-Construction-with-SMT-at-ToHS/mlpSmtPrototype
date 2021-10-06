"""
This module can be used to create an MLP, train it  a function `f`, save it, and plot the error. So far,
`f` :math:`\in \mathbb{R} \\rightarrow \mathbb{R}` or `f` :math:`\in \mathbb{R}^2 \\rightarrow \mathbb{R}^2` is
supported
"""
import sys
import inspect

import tensorflow as tf
from tensorflow import keras
from mlp_smt_closed.mlp.functions import *
from mlp_smt_closed.mlp.plot3d import *
from mlp_smt_closed.mlp.sampling import *

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)


def train(func_class, nodes_l1, nodes_l2, learning_rate, epochs):
    """Create an MLP, train it on a function `f` :math:`\in \mathbb{R}^n \\rightarrow \mathbb{R}^n` contained in
    `func_class`, save it, and plot the error. The MLP has 3 layers, the last layer automatically has
    `func_class.dimension()` nodes.

    Args:
        func_class:
            a class which must contain a static function `f` :math:`\in \mathbb{R}^2 \\rightarrow \mathbb{R}^2` and a
            `name` attribute
        nodes_l1:
            number of nodes in layer one of the network
        nodes_l2:
            number of nodes in layer two of the network
        learning_rate:
            learning rate during the training
        epochs:
            number of epochs in the training
    """

    # load samples
    x_samples, y_samples, x_train, x_test, y_train, y_test = load_samples(func_class)

    # Create a simple mlp model.
    model = keras.Sequential([
        keras.layers.Dense(nodes_l1, activation=tf.nn.relu),
        keras.layers.Dense(nodes_l2, activation=tf.nn.relu),
        keras.layers.Dense(func_class.dimension())
    ])

    # Use the MSE regression loss (learning rate!?)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    # boundaries = [50]
    # values = [0.01, 0.0001]
    # learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    # Later, whenever we perform an optimization step, we pass in the step.

    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    # configure model
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    # Train, validate & save the model.
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    model_filename = 'models_infeasible/' + func_class.name() + '_model.h5'
    model.save(model_filename)

    # calculate predictions
    y_prediction = model.predict(x_test)

    if func_class.dimension() == 1:
        # Plot the results
        plt.plot(x_samples, y_samples, 'r')
        plt.scatter(x_test, y_prediction)
        plt.savefig('plots_infeasible/' + func_class.name() + '_learned.png')
        plt.show()
        plt.clf()
    elif func_class.dimension() == 2:
        # plot distances of y_test and y_prediction
        plot_dist_map(x_test, y_test, y_prediction, func_class.name())


def open_model(func_class):
    """
    Print the saved model/parameters of an MLP trained on `func_class` if the model was trained on `func_class` already

    Args:
        func_class: function for which the model should be printed
    """

    # get path to model
    path = 'models/' + func_class.name() + '_model.h5'
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
    # sample(LinearA2D(), [[-10, 10], [-10, 10]], [100+1, 100+1], 0.02)
    # sample(PolyDeg3(), [[-10, 10]], [1001], 0.02)
    # sample(QuadraticA(), [[-10, 10]], [1001], 0.02)
    platoon_intervals = [[-10, 10] for _ in range(15)]
    platoon_sizes = [2 for _ in range(15)]
    sample(Platoon(), platoon_intervals, platoon_sizes, 0)
    train(Platoon(), 20, 20, 0.01, 100)
    # train(LinearA2D(), 20, 20, 0.0001, 500)
    # train(PolyDeg3(), 20, 20, 0.001, 500)
    # train(QuadraticA(), 20, 20, 0.001, 500)
    # open_model(LinearB2D())
