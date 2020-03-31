import sys;
import math;
import numpy as np;

from GradientDescent import *;


class LogisticRegression:
    __MAX_POWER = math.log(sys.float_info.max) - 1;

    def __init__(self):
        self.__X = None;
        self.__Y = None;
        self.__theta = None;
        self.__lambda = None;
        self.__gradientDescent = GradientDescent(self._costFunction, self._getGradient);


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


    def _costFunction(self, theta):
        return self.__log1AddExp(theta).sum() - (self.__Y.T * self.__X * theta)[0, 0] + \
               (theta.T * theta)[0, 0] * self.__lambda / 2;


    def _getGradient(self, theta):
        return self.__X.T * (self.__sigmoid(self.__X, theta) - self.__Y) + self.__lambda * theta;


    def train(self, dataSet, alpha, minDifference, maxCount = None, lmd = 1):
        self.__X = np.hstack((np.mat(np.ones((dataSet.shape[0], 1))), dataSet[:, :-1]));
        self.__Y = dataSet[:, -1];
        self.__theta = np.mat(np.ones((dataSet.shape[1], 1)));
        self.__lambda = max(0, lmd);

        self.__theta, costValue, gradient = self.__gradientDescent.search(self.__theta, alpha, minDifference, maxCount);


    def predict(self, dataSet):
        return (self.__sigmoid(np.hstack((np.mat(np.ones((dataSet.shape[0], 1))), dataSet)), self.__theta) > 0.5) + 0;
