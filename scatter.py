from pylab import *
import matplotlib.pyplot as plt

# get sample-positions
A, B = np.mgrid[-2:2:5j, -2:2:5j]
x_test = np.vstack((A.flatten(), B.flatten())).T

y_test = np.array([[1, 1],
                   [2, 3],
                   [1, 5]])

y_prediction = np.array([[4, 6],
                         [2, 2],
                         [5, 2]])


# plot prediction-error
def plot_dist_map(x_test, y_test, y_prediction):

    # reshape sample-points
    plot_dim_1 = [i[0] for i in x_test]
    plot_dim_2 = [i[1] for i in x_test]

    # print(y_test[1])
    # calculate errors
    errors = []
    for i in range(y_test.shape[0]):
        errors.append(np.linalg.norm(y_test[i]-y_prediction[i]))
    errors = np.array(errors)
    print(errors)

    plot_dim_3 = np.zeros(25)

    plot_dim_3[12] = 1
    plot_dim_3[13] = 0.5

    # plot errors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colmap = cm.ScalarMappable(cmap=cm.coolwarm)
    colmap.set_array(plot_dim_3)
    ax.scatter(plot_dim_1, plot_dim_2, plot_dim_3, c=cm.coolwarm(plot_dim_3/max(plot_dim_3)), marker='o')
    fig.colorbar(colmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('prediction error')
    plt.show()


plot_dist_map(x_test, y_test, x_test)
