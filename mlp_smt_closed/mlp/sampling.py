"""
This module contains functions to

* take samples from a given function and store the samples in a file
* load saved samples for a given function
"""

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def sample_1d(func_class, size, interval, scatter):
    """This function takes ``size`` samples of the function ``func_class``
    :math:`\in \mathbb{R} \\rightarrow \mathbb{R}` within ``interval``. It also scatters the function values (normally
    distributed) according to ``scatter``. Then, it divides the samples into training and test samples and stores
    everything in a file in the folder ``/samples``.

    Args:
        func_class: function to sample
        size: number of samples to be taken
        interval: interval in which the samples should be taken
        scatter: standard deviation for normal distribution (choose 0 if no scattering desired)
    """

    # TODO: interval/dimenstion sanity check ?

    lower, upper = interval

    # x-values of the samples
    x_samples = np.linspace(lower, upper, size).T
    # y=f(x) values of the samples
    y_samples = [func_class.f(x) for x in x_samples]
    y_samples = np.array(y_samples)

    # Add noise to y-values
    y_scatter = np.random.normal(0.0, scatter, y_samples.shape)
    y_samples = y_samples + y_scatter

    # process samples, get test training data
    x, y = shuffle(x_samples, y_samples, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # save samples to file
    sample_path = 'samples/' + func_class.name
    np.savez(sample_path, x_samples=x_samples, y_samples=y_samples, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def sample_2d(func_class, x1_size, x2_size, x1_interval, x2_interval, scatter):
    """This function takes ``x1_size`` * ``x2_size`` samples of the function ``func_class``
    :math:`\in \mathbb{R}^2 \\rightarrow \mathbb{R}^2` within ``x1_interval`` :math:`\\times` ``x2_interval``.
    It also scatters the function values (normally distributed) according to ``scatter``. Then, it divides the samples
    into training and test samples and stores everything in a file in the folder ``/samples``.

    Args:
        func_class: function to sample
        x1_size: number of samples to be taken in the first input dimension
        x2_size: number of samples to be taken in the second input dimension
        x1_interval: interval of the first input dimension from which the samples should be taken
        x2_interval: interval of the second input dimension from which the samples should be taken
        scatter: standard deviation for normal distribution (choose 0 if no scattering desired)
    """
    # Sample 2D training data.

    # np.mgrid interprets the complex part of the number as absolute number of samples. a real number would indicate the
    # step size
    x1_size_complex = complex(0, x1_size)
    x2_size_complex = complex(0, x2_size)

    x1_lower, x1_upper = x1_interval
    x2_lower, x2_upper = x2_interval

    # format magic
    a, b = np.mgrid[x1_lower:x1_upper:x1_size_complex, x2_lower:x2_upper:x2_size_complex]
    x_samples = np.vstack((a.flatten(), b.flatten())).T

    # calculation
    y_samples = [func_class.f(x) for x in x_samples]
    y_samples = np.array(y_samples)

    # Add noise to y-values
    y_scatter = np.random.normal(0, scatter, y_samples.shape)
    y_samples = y_samples + y_scatter

    # process samples, get test training data
    x, y = shuffle(x_samples, y_samples, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # save samples to file
    sample_path = 'samples/' + func_class.name
    np.savez(sample_path, x_samples=x_samples, y_samples=y_samples, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def load_samples(func_class):
    """
    Load saved samples of the specified ``func_class``

    Args:
        func_class: class for which to load the samples

    Returns:
        (list): List containing ``x_samples``, ``y_samples``, ``x_train``, ``x_test``, ``y_train``, ``y_test`` in that order
    """
    sample_path = 'samples/' + func_class.name + '.npz'
    samples = np.load(sample_path)

    x_samples = samples['x_samples']
    y_samples = samples['y_samples']
    x_train = samples['x_train']
    x_test = samples['x_test']
    y_train = samples['y_train']
    y_test = samples['y_test']

    return x_samples, y_samples, x_train, x_test, y_train, y_test
