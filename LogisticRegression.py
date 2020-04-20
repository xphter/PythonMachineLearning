import sys;
import math;
import numpy as np;

from Optimizer import *;


class LogisticRegression(IOptimizerTarget, IGradientProvider, IHessianMatrixProvider):
    __MAX_POWER = math.log(sys.float_info.max) - 1;

    def __init__(self, optimizer):
        if optimizer is None or not isinstance(optimizer, OptimizerBase):
            raise ValueError();

        self.__X = None;
        self.__y = None;
        self.__theta = None;
        self.__lambda = None;
        self.__optimizer = optimizer;
        self.__optimizer.setTarget(self);


    @property
    def theta(self):
        return self.__theta;


    def __sigmoid(self, X, theta):
        return 1 / (1 + np.exp(-1 * X * theta));


    def __log1AddExp(self, theta):
        value = self.__X * theta;

        if np.all(value < LogisticRegression.__MAX_POWER):
            return np.log(1 + np.exp(value));

        item = None;
        result = np.mat(np.zeros(value.shape));
        for i in range(value.shape[0]):
            item = value[i, 0];
            result[i, 0] = item if item >= LogisticRegression.__MAX_POWER else math.log(1 + math.exp(item));

        return result;


    def getTargetValue(self, theta):
        return self.__log1AddExp(theta).sum() - (self.__y.T * self.__X * theta)[0, 0] + \
               (theta.T * theta)[0, 0] * self.__lambda / 2;


    def getGradient(self, theta):
        return self.__X.T * (self.__sigmoid(self.__X, theta) - self.__y) + self.__lambda * theta;


    def getHessianMatrix(self, theta):
        h = np.exp(-1 * self.__X * theta);
        h = np.divide(h, 1 + 2 * h + np.power(h, 2));

        return np.multiply(self.__X, h).T * self.__X;


    def train(self, dataSet, lmd = 1):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        self.__X = np.hstack((np.mat(np.ones((dataSet.shape[0], 1))), dataSet[:, :-1]));
        self.__y = dataSet[:, -1];
        self.__theta = np.mat(np.zeros((dataSet.shape[1], 1)));
        self.__lambda = max(0, lmd);

        self.__theta, costValue, gradient = self.__optimizer.search(self.__theta);

        return costValue;


    def predict(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        return (self.__sigmoid(np.hstack((np.mat(np.ones((dataSet.shape[0], 1))), dataSet)), self.__theta) > 0.5) + 0;
