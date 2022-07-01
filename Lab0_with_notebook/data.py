import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:

    min_x = 0
    max_x = 10
    min_y = 0
    max_y = 10

    def __init__(self):
        x_interval, y_interval = self.max_x - self.min_x, self.max_y - self.min_y
        self.mean = (x_interval, y_interval) * np.random.random_sample(2)
        self.mean += (self.min_x, self.min_y)
        eigvalx = (np.random.random_sample() * (x_interval) / 5) ** 2
        eigvaly = (np.random.random_sample() * (y_interval) / 5) ** 2
        eigenvalues = (eigvalx, eigvaly)
        theta = np.random.random_sample()
        rotation_matrix = [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
        self.covariance_matrix = np.dot(
            np.dot(np.transpose(rotation_matrix), np.diag(eigenvalues)),
            rotation_matrix,
        )

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.covariance_matrix, n)


def eval_AP(ranked_labels):
    """Recovers AP from ranked labels"""

    n = len(ranked_labels)
    pos = sum(ranked_labels)
    neg = n - pos

    tp = pos
    tn = 0
    fn = 0
    fp = neg

    sumprec = 0
    # IPython.embed()
    for x in ranked_labels:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if x:
            sumprec += precision

        # print (x, tp,tn,fp,fn, precision, recall, sumprec)
        # IPython.embed()

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec / pos


def sample_gauss_2d(nclasses, nsamples):
    # create the distributions and groundtruth labels
    Gs = []
    Ys = []
    for i in range(nclasses):
        Gs.append(Random2DGaussian())
        Ys.append(i)

    # sample the dataset
    X = np.vstack([G.get_sample(nsamples) for G in Gs])
    Y_ = np.hstack([[Y] * nsamples for Y in Ys])

    return X, Y_


def eval_perf_binary(Y, Y_):
    tp = sum(np.logical_and(Y == Y_, Y_ == True))
    fn = sum(np.logical_and(Y != Y_, Y_ == True))
    tn = sum(np.logical_and(Y == Y_, Y_ == False))
    fp = sum(np.logical_and(Y != Y_, Y_ == False))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return accuracy, recall, precision


def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_)+1
    M = np.bincount(n * Y_ + Y, minlength=n*n).reshape(n, n)
    for i in range(n):
        tp_i = M[i,i]
        fn_i = np.sum(M[i,:]) - tp_i
        fp_i = np.sum(M[:,i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append((recall_i, precision_i))

    accuracy = np.trace(M)/np.sum(M)

    return accuracy, pr, M


def graph_data(X, Y_, Y, special=[]):
    """Creates a scatter plot (visualize with plt.show)

    Arguments:
        X  ... data (np. array Nx2)
        Y_ ... true classes (np.array Nx1)
        Y  ... predicted classes (np.array Nx1)
    Returns:
        None
    """
    # colors of the datapoint markers
    palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    # draw the correctly classified datapoints
    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good],
                s=sizes[good], marker='o', edgecolors='black')

    # draw the incorrectly classified datapoints
    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad],
                s=sizes[bad], marker='s', edgecolors='black')


def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    """Creates a surface plot (visualize with plt.show)

    Arguments:
      fun    ... the decision function (Nx2)->(Nx1)
      rect   ... he domain in which we plot the data:
                 ([x_min,y_min], [x_max,y_max])
      offset ... the value of the decision function
             on the border between the classes;
             we typically have:
             offset = 0.5 for probabilistic models
                (e.g. logistic regression)
             offset = 0 for models which do not squash
                classification scores (e.g. SVM)
  width,height ... coordinate grid resolution

    Returns:
      None
    """

    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    # get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values,
                   vmin=delta - maxval, vmax=delta + maxval)

    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])


if __name__ == "__main__":
    np.random.seed(100)

    # # task 1
    # G = Random2DGaussian()
    # X = G.get_sample(100)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    # task 3
    # get the training dataset
    X, Y_ = sample_gauss_2d(2, 100)
    # get the class predictions
    Y = myDummyDecision(X) > 0.5

    # task 4
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0)

    # graph the data points
    graph_data(X, Y_, Y)

    # show the results
    plt.show()
