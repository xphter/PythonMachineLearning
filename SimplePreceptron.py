import numpy as np;


class SimplePerceptron:
    def __init__(self, alpha):
        self.__alpha = alpha;
        self.__theta = None;


    def train(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        X = np.hstack((np.ones((dataSet.shape[0], 1)), dataSet[:, :-1]));
        y = dataSet[:, -1] * self.__alpha;
        n = np.mat(np.zeros((dataSet.shape[0], 1)));
        gram = X * X.T;

        count = range(0, X.shape[0]);

        while True:
            hasError = False;
            a = np.multiply(n, y);

            for i in count:
                if y[i, 0] * (gram[i, :] * a) <= 0:
                    hasError = True;
                    break;

            if hasError:
                n[i, 0] += 1;
            else:
                break;

        self.__theta = np.multiply(X, a).sum(axis = 0).T;


    def predict(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        X = np.hstack((np.ones((dataSet.shape[0], 1)), dataSet));
        Y = X * self.__theta;

        return Y;


