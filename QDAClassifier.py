import math;
import numpy as np;


class QDAClassifier:
    def __init__(self):
        self.__X = None;
        self.__y = None;

        self.__pc = None;
        self.__mu = None;
        self.__sigmaI = None;
        self.__sigmaD = None;
        self.__classes = None;


    def __getSigma(self, c):
        X = self.__X[(self.__y == c).A.flatten(), :];

        return (X.T * X) / (X.shape[0] - 1);


    def train(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        self.__X = dataSet[:, :-1];
        self.__y = dataSet[:, -1];

        y = self.__y.A.flatten().tolist();
        y.sort();
        classes = set(y);
        centralizedX = self.__X - self.__X.mean(0);
        n, p = self.__X.shape;

        self.__mu = np.vstack(tuple([self.__X[(self.__y == c).A.flatten(), :].mean(0) for c in classes]));
        self.__sigmaI = [self.__getSigma(c).I for c in classes];
        self.__sigmaD = np.mat([math.log(np.linalg.det(item)) / 2 for item in self.__sigmaI]);
        self.__pc = np.log(np.mat([y.count(c) for c in classes]) / n);
        self.__classes = np.array(list(classes));


    def predict(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        delta = self.__pc + self.__sigmaD - np.hstack(tuple([np.multiply((dataSet - self.__mu[i, :]) * self.__sigmaI[i], dataSet - self.__mu[i, :]).sum(1) for i in range(0, len(self.__classes))])) / 2;

        return np.mat(self.__classes[delta.argmax(1).A.flatten().tolist()]).T;
