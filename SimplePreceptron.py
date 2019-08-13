import numpy as np;


class SimplePerceptron:
    def __init__(self, alpha):
        self.__alpha = alpha;
        self.__theta = None;


    def train(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        X = np.hstack((np.ones((dataSet.shape[0], 1)), dataSet[:, :-1]));
        y = dataSet[:, -1].T * self.__alpha;
        n = np.mat(np.zeros((1, dataSet.shape[0])));
        gram = X * X.T;

        count = range(0, X.shape[0]);

        while True:
            hasError = False;

            for i in count:
                if y[0, i] * np.multiply(np.multiply(gram[i, :], y), n).sum(axis = 1)[0, 0] <= 0:
                    hasError = True;
                    break;

            if hasError:
                n[0, i] += 1;
            else:
                break;

        self.__theta = np.multiply(np.multiply(X, y.T), n.T).sum(axis = 0).T;


    def predict(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        X = np.hstack((np.ones((dataSet.shape[0], 1)), dataSet));
        Y = X * self.__theta;

        return Y;


