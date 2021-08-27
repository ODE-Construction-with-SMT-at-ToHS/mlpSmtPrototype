"""
This module contains functions to

* take samples from a given function and store the samples in a file
* load saved samples for a given function
"""

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def sample(func_class, intervals, sizes, scatter):
    """This function takes ``sizes[0]`` * ...  * ``sizes[n-1]`` samples of the function ``func_class``
    :math:`\in \mathbb{R}^n \\rightarrow \mathbb{R}^n` within ``intervals[0]`` :math:`\\times` ... :math:`\\times`
    ``intervals[n-1]``. It also scatters the function values (normally distributed) according to ``scatter``. Then, it
    divides the samples into training and test samples and stores everything in a file in the folder ``/samples``.

    Args:
        func_class: function to sample
        intervals: tuple of intervals for the corresponding dimension
        sizes: tuple of number of samples to take in the corresponding dimension
        scatter: standard deviation for normal distribution (choose 0 if no scattering desired)
    """

    # sanity check
    if len(intervals) != func_class.dimension():
        print('Error: dimension of', func_class.name(), 'is', func_class.dimension(), 'but you provided',
              len(intervals), 'intervals')
    if len(sizes) != func_class.dimension():
        print('Error: dimension of', func_class.name(), 'is', func_class.dimension(), 'but you provided',
              len(sizes), 'sizes')

    # use intervals and their sizes to create samples for each dimension separately
    interval_vectors = [np.linspace(intervals[dim][0], intervals[dim][1], sizes[dim]) for dim in range(func_class.dimension())]

    # use the array of interval vectors to get an n-dimensional grid
    x_samples = np.vstack(np.meshgrid(*interval_vectors)).reshape(func_class.dimension(), -1).T

    # calculate y values for x-samples
    y_samples = [func_class.f(x) for x in x_samples]
    y_samples = np.array(y_samples)

    # add noise to y-values
    y_scatter = np.random.normal(0, scatter, y_samples.shape)
    y_samples = y_samples + y_scatter

    # process samples, divide into test and training data
    x, y = shuffle(x_samples, y_samples, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # save samples to file
    sample_path = 'samples/' + func_class.name()
    np.savez(sample_path, x_samples=x_samples, y_samples=y_samples, x_train=x_train, x_test=x_test, y_train=y_train,
             y_test=y_test)


def load_samples(func_class):
    """
    Load saved samples of the specified ``func_class``

    Args:
        func_class: class for which to load the samples

    Returns:
        (list): List containing ``x_samples``, ``y_samples``, ``x_train``, ``x_test``, ``y_train``, ``y_test`` in that order
    """
    sample_path = 'samples/' + func_class.name() + '.npz'
    samples = np.load(sample_path)

    x_samples = samples['x_samples']
    y_samples = samples['y_samples']
    x_train = samples['x_train']
    x_test = samples['x_test']
    y_train = samples['y_train']
    y_test = samples['y_test']

    return x_samples, y_samples, x_train, x_test, y_train, y_test
