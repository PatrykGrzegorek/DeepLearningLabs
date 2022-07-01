import numpy as np
import matplotlib.pyplot as plt

import data


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def binlogreg_train(X, Y_):
    """
    Arguments
      X:  data, np.array NxD
      Y_: class indices, np.array Nx1

    Return values
      w, b: parameters of binary logistic regression
    """
    N = len(Y_)
    w = np.transpose(np.random.randn(2))
    b = 0
    param_niter = 1000
    param_delta = 0.01

    # gradient descent (param_niter iterations)
    for i in range(param_niter):
        # classification scores
        scores = np.dot(X, w) + b  # N x 1

        # a posteriori class probabilities c_1
        probs = sigmoid(scores)  # N x 1

        # loss
        loss = np.sum(-np.log(probs))  # scalar

        # trace
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        predictions = [1 if i > 0.5 else 0 for i in probs]

        # derivative of the loss function with respect to classification scores
        dL_dscores = probs - Y_  # N x 1

        # gradients with respect to parameters
        grad_w = 1 / N * np.dot(X.T, dL_dscores)  # D x 1
        grad_b = 1/N * sum(dL_dscores)  # 1 x 1

        # # modifying the parameters
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    """
    Arguments
    X:    data, np.array NxD
    w, b: logistic regression parameters

    Return values
    probs: a posteriori probabilities for c1, dimensions Nx1
    """
    s = np.dot(X, w) + b
    probs = sigmoid(s)  # N x 1
    return probs


def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)

    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = [1 if i > 0.5 else 0 for i in probs]

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y)

    # show the plot
    plt.show()
