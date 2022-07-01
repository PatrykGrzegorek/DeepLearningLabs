import numpy as np
import data
import matplotlib.pyplot as plt


def logreg_train(X, Y_):

    c = max(Y_) + 1
    m = X.shape[0]
    w = np.random.randn(c, X.shape[1])
    b = np.zeros(c)
    param_delta = 0.1

    for i in range(100):
        scores = np.dot(X, w.T) + b
        expscores = np.exp(scores)
        sumexp = np.sum(expscores)
        probs = expscores/sumexp
        logprobs = np.log(probs)
        loss = np.mean(-logprobs)

        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))

        grad_w = np.dot(probs.T, X)
        grad_b = np.sum(probs)

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def logreg_classify(X, w, b):
    scores = np.dot(X, w.T) + b
    expscores = np.exp(scores)
    sumexp = np.sum(expscores)
    probs = expscores / sumexp

    return probs


def logreg_decfun(w, b):
    def classify(X):
        return logreg_classify(X, w, b)
    return classify


if __name__ == '__main__':

    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 100)

    # train the model
    w, b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, w, b)
    Y = [np.argmax(i) for i in probs]

    # report performance
    accuracy, recall_precision, confusion_matrix = data.eval_perf_multi(Y, Y_)
    #AP = data.eval_AP(Y_[probs.argsort()])
    print(f"Accuracy: {accuracy}")
    print(f"Recall and precision for each class: {recall_precision}")
    print(f"Confusion matrix:\n{confusion_matrix}")

    # decfun = logreg_decfun(w, b)
    # bbox = (np.min(X, axis=0), np.max(X, axis=0))
    # data.graph_surface(decfun, bbox, offset=0.5)

    # # graph the data points
    # data.graph_data(X, Y_, Y, special=[])

    # # show the plot
    # plt.show()