"""Plot the distance between pairs of 2D points on a 2D grid in 3D."""

import matplotlib.pyplot as plt
import numpy as np
import pylab


def plot_dist_map(x_test, y_test, y_prediction, name):
    """Plot the distance between pairs of 2D points on a 2D grid in 3D.

    This module can be used to plot the (2 norm) distance between two points :math:`y, y' \in \mathbb{R}^2` for a given
    grid (:math:`\subset \mathbb{R}^2`) of pairs. Its aim is to visualize the prediction error of an MLP trained on a
    function :math:`f \in \mathbb{R}^2 \\rightarrow \mathbb{R}^2` with input :math:`x \in \mathbb{R}^2` and output
    :math:`y' \in \mathbb{R}^2`.

    Args:
        x_test:
            (n,2)-array; list of all points ``x_test[i]`` :math:`\in \mathbb{R}^2` in the grid. (inputs for the MLP)
        y_test:
            (n,2)-array; list of all points ``y_test[i] = f(x_test[i])`` :math:`\in \mathbb{R}^2`, where ``f`` is the
            function on which the MLP was trained
        y_prediction:
            (n,2)-array; list of all points ``y_prediction[i]`` :math:`\in \mathbb{R}^2`, where ``y_prediction[i] is the
            prediction of the MLP trained on ``f`` for input ``x_test[i]``
        name:
            name of the function ``f`` on which the MLP was trained, used to name the file in which the visualization is
            saved.
    """

    # reshape sample-points
    plot_dim_1 = [i[0] for i in x_test]
    plot_dim_2 = [i[1] for i in x_test]

    # calculate errors
    errors = [np.linalg.norm(y_test[i]-y_prediction[i]) for i in range(y_test.shape[0])]
    errors = np.array(errors)
    plot_dim_3 = errors

    # create and configure plot
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
    # save plot
    fig.savefig('plots/' + name + '_learned.png')
    # show plot
    plt.show()
    plt.clf()
