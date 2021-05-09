import matplotlib.pyplot as plt
import numpy as np
import pylab


def plot_dist_map(x_test, y_test, y_prediction):

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
    plt.show()
