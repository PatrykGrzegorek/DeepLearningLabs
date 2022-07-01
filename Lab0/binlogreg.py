import numpy as np
import data
import matplotlib.pyplot as plt

def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs


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
    param_delta = 0.01
    m = X.shape[0]
    w = np.random.randn(X.shape[1])
    b = 0
    # gradient descent (param_niter iteratons)
    for i in range(900):
        # classification scores
        scores = np.dot(X, w) + b

        # a posteriori class probabilities c_1
        probs = sigmoid(scores)     # N x 1

        # loss (scalar)
        loss = np.sum(-np.log(probs))

        # trace
        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivative of the loss funciton with respect to classification scores
        dL_dscores = probs - Y_     # N x 1

        # gradijents with respect to parameters
        grad_w = 1/m * np.dot(X.T, dL_dscores)   # D x 1
        grad_b = 1/m * np.sum(dL_dscores)     # 1 x 1

        # modifying the parameters
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    # classification scores
    scores = np.dot(X, w) + b

    # a posteriori class probabilities c_1
    probs = sigmoid(scores)

    return probs


def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify


if __name__ == '__main__':

    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    print(X.shape, Y_.shape)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = [1 if i > 0.5 else 0 for i in probs]

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)

    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
