import math;
import numpy as np;


class LDAClassifier:
    def __init__(self):
        self.__X = None;
        self.__y = None;

        self.__pc = None;
        self.__mu = None;
        self.__sigmaI = None;
        self.__classes = None;


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
        self.__sigmaI = (centralizedX.T * centralizedX / (n - len(classes))).I;
        self.__pc = np.log(np.mat([y.count(c) for c in classes]) / n) - np.multiply(self.__mu * self.__sigmaI, self.__mu).sum(1).T / 2;
        self.__classes = np.array(list(classes));



    def predict(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        delta = self.__pc + dataSet * self.__sigmaI * self.__mu.T;

        return np.mat(self.__classes[delta.argmax(1).A.flatten().tolist()]).T;
