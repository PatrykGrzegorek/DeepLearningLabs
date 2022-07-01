import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def sample_gmm_2d(k, c, n):
    '''
    returns:
      X  ... data in a matrix [K·N x 2 ]
      Y_ ... class indices of data [K·N]
    '''
    x = list()
    y = list()

    for _ in range(k):
        x.append(np.random.normal(loc=np.random.uniform(0, 7), size=(n, 2)))
        y.append(np.random.randint(low=0, high=c, size=n))

    return np.vstack(x), np.hstack(y)


def graph_data(X, Y_, Y, special=None):
    if special is None:
        special = []

    palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]

    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good],
                s=sizes[good], marker='o', edgecolors='black')

    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad],
                s=sizes[bad], marker='s', edgecolors='black')


def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    values = function(grid).reshape((width, height))

    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    plt.pcolormesh(xx0, xx1, values,
                   vmin=delta - maxval, vmax=delta + maxval)

    if offset is not None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])


def convert_to_one_hot(datapoints, n_classes: int or None = None):
    if n_classes is None:
        n_classes = np.max(datapoints) + 1

    _datapoints = list()

    for datapoint in datapoints:
        _t = np.zeros(n_classes)
        _t[datapoint] = 1

        _datapoints.append(_t)

    return np.array(_datapoints)


if __name__ == "__main__":
    np.random.seed(100)
    # tf.random.set_seed(100)

    X, Y_ = sample_gmm_2d(4, 2, 30)

    Y = myDummyDecision(X) > 0.5

    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0)
    graph_data(X, Y_, Y)

    plt.show()
