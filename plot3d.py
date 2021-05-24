"""
This module can be used to plot the (2 norm) distance between two 2D points for a given grid. Its aim is to visualize
the prediction error of an MLP with 2D input and 2D output.
"""

import matplotlib.pyplot as plt
import numpy as np
import pylab


def plot_dist_map(x_test, y_test, y_prediction, name):
    """
    Plot the distance (2 norm) between `y_test[i]` and `y_prediction[i]` for all points `x_test[i]` on the grid `x_test`

    Parameters
    ----------
    x_test : (n,2) array
             list of all points in the grid
    y_test : (n,2) array
    TODO: add other parameters
    """

    # reshape sample-points
    plot_dim_1 = [i[0] for i in x_test]
    plot_dim_2 = [i[1] for i in x_test]

    # calculate errors
    errors = [np.linalg.norm(y_test[i]-y_prediction[i]) for i in range(y_test.shape[0])]
    errors = np.array(errors)
    plot_dim_3 = errors

    # plot errors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colmap = pylab.cm.ScalarMappable(cmap=pylab.cm.coolwarm)
    colmap.set_array(plot_dim_3)
    ax.scatter(plot_dim_1, plot_dim_2, plot_dim_3, c=pylab.cm.coolwarm(plot_dim_3/max(plot_dim_3)), marker='o')
    fig.colorbar(colmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('prediction error')
    plt.draw()
    fig.savefig('plots/' + name + '_learned.png')
    plt.show()
    plt.clf()
