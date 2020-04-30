import math;
import numpy as np;


class LDAClassifier:
    def __init__(self):
        self.__pc = None;
        self.__logPC = None;
        self.__mu = None;
        self.__sigmaI = None;
        self.__muSigma = None;
        self.__classes = None;


    def train(self, X, y):
        if X is None or y is None:
            raise ValueError();

        listY = np.sort(y.A.flatten()).tolist();
        classes = list(set(listY));
        centralizedX = X - X.mean(0);
        n, p = X.shape;

        self.__mu = np.vstack(tuple([X[(y == c).A.flatten(), :].mean(0) for c in classes]));
        self.__sigmaI = (centralizedX.T * centralizedX / (n - len(classes))).I;
        self.__pc = np.mat([listY.count(c) for c in classes]) / n;
        self.__logPC = np.log(self.__pc);
        self.__muSigma = np.multiply(self.__mu * self.__sigmaI, self.__mu).sum(1).T / 2;
        self.__classes = np.array(classes);


    def predict(self, X):
        if X is None or not isinstance(X, np.matrix):
            raise ValueError();

        K = self.__classes.shape[0];
        delta = self.__logPC + X * self.__sigmaI * self.__mu.T - self.__muSigma;
        posterior = np.hstack(tuple([1 / np.exp(delta - delta[:, k]).sum(1) for k in range(0, K)]));

        return np.mat(self.__classes[delta.argmax(1).A.flatten().tolist()]).T, posterior;


class QDAClassifier:
    def __init__(self):
        self.__pc = None;
        self.__logPC = None;
        self.__mu = None;
        self.__sigmaI = None;
        self.__sigmaD = None;
        self.__classes = None;


    def __getSigma(self, X, y, mu, c):
        Z = X[(y == c).A.flatten(), :] - mu;

        return (Z.T * Z) / (Z.shape[0] - 1);


    def train(self, X, y):
        if X is None or y is None:
            raise ValueError();

        listY = np.sort(y.A.flatten()).tolist();
        classes = list(set(listY));
        n, p = X.shape;

        self.__mu = np.vstack(tuple([X[(y == c).A.flatten(), :].mean(0) for c in classes]));
        self.__sigmaI = [self.__getSigma(X, y, self.__mu[k, :], classes[k]).I for k in range(0, len(classes))];
        self.__sigmaD = np.mat([math.log(np.linalg.det(item)) / 2 for item in self.__sigmaI]);
        self.__pc = np.mat([listY.count(c) for c in classes]) / n;
        self.__logPC = np.log(self.__pc);
        self.__classes = np.array(classes);


    def predict(self, X):
        if X is None or not isinstance(X, np.matrix):
            raise ValueError();

        K = self.__classes.shape[0];
        delta = self.__logPC + self.__sigmaD - np.hstack(tuple([np.multiply((X - self.__mu[k, :]) * self.__sigmaI[k], X - self.__mu[k, :]).sum(1) for k in range(0, K)])) / 2;
        posterior = np.hstack(tuple([1 / np.exp(delta - delta[:, k]).sum(1) for k in range(0, K)]));

        return np.mat(self.__classes[delta.argmax(1).A.flatten().tolist()]).T, posterior;


