import os, sys
import tensorflow as tf

from tensorflow import keras
from sklearn.utils import shuffle
from functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train1d(funcClass):

    # Sample some training data.
    xValues = np.linspace(0, 10, 50)
    yValues = []
    for x in xValues:
        yValues.append(funcClass.f(x))

    X, y = shuffle(xValues, yValues, random_state=0)
    xVals = np.array(X)
    yVals = np.array(y)

    # Create a simple mlp model.
    model = keras.Sequential([
        keras.layers.Dense(20, activation=tf.nn.relu, input_shape=[1]),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    # Use the MSE regression loss 
    optimizer = tf.keras.optimizers.RMSprop(0.01)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    
    # Train & save the model.
    model.fit(xVals, yVals, epochs = 100)
    model_filename = funcClass.name + '_model.h5'
    model.save(model_filename)

    # Plot the results.
    predsY = model.predict(xVals)

    plt.plot(xValues, yValues, 'r')
    plt.scatter(xVals, predsY)
    plt.show()


if __name__ == '__main__':
    print(linearA.name)
    train1d(linearA)
    print('no error so faar')
