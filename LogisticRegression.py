import sys;
import math;
import numpy as np;
from scipy.stats import norm;

from Optimizer import *;


class LogisticRegression(IOptimizerTarget, IGradientProvider, IHessianMatrixProvider):
    __DEFAULT_SIG_LEVEL = 0.05;
    __MAX_POWER = math.log(sys.float_info.max) - 1;

    def __init__(self, optimizer):
        if optimizer is None or not isinstance(optimizer, OptimizerBase):
            raise ValueError();

        self.__X = None;
        self.__y = None;
        self.__lambda = None;
        self.__optimizer = optimizer;
        self.__optimizer.setTarget(self);

        self.__theta = None;
        self.__thetaStd = None;
        self.__thetaZ = None;
        self.__thetaP = None;
        self.__thetaValue = None;


    def __repr__(self):
        p = self.__theta.shape[0];

        return "logit = {0}{1}".format(
            self.__thetaValue[0, 0],
            "".join([" {0} {1} * x{2:.0f}".format("+" if item[1] >= 0 else "-", math.fabs(item[1]), item[0])
                     for item in
                     np.hstack((np.mat(range(1, p)).T, self.__thetaValue[1:, :])).tolist()])
        );


    def __str__(self):
        p = self.__theta.shape[0];

        return "P(y=1|X) = e^z / (1 + e^z); z = θ0{0}\r\n{1}".format(
            "".join([" + θ{0:.0f} * x{0:.0f}".format(i) for i in range(1, p)]),
            "\r\n".join(
                ["θ{0:.0f} = {1}, std = {2}, z-value = {3}, p-value = {4}".format(*item)
                 for item in
                 np.hstack((np.mat(range(0, p)).T, self.__theta, self.__thetaStd, self.__thetaZ, self.__thetaP)).tolist()])
        );


    @property
    def theta(self):
        return self.__thetaValue;


    @property
    def thetaP(self):
        return self.__thetaP;


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


    def train(self, X, y, lmd = 1, sigLevel = None):
        if X is None or y is None:
            raise ValueError();

        if sigLevel is None:
            sigLevel = LogisticRegression.__DEFAULT_SIG_LEVEL;

        n, p = X.shape[0], X.shape[1] + 1;
        self.__X = np.hstack((np.mat(np.ones((n, 1))), X));
        self.__y = y;
        self.__theta = np.mat(np.zeros((p, 1)));
        self.__lambda = max(0, lmd);

        self.__theta, costValue, gradient = self.__optimizer.search(self.__theta);
        self.__thetaStd = np.sqrt(self.getHessianMatrix(self.__theta).I)[range(0, p), range(0, p)].T;
        self.__thetaZ = np.divide(self.__theta, self.__thetaStd);
        self.__thetaP = np.mat(2 * (1 - norm.cdf(np.abs(self.__thetaZ))));
        self.__thetaValue = self.__theta.copy();
        # self.__thetaValue[(self.__thetaP >= sigLevel).A.flatten(), :] = 0;

        return costValue;


    def predictValue(self, X):
        if X is None or not isinstance(X, np.matrix):
            raise ValueError();

        return self.__sigmoid(np.hstack((np.mat(np.ones((X.shape[0], 1))), X)), self.__thetaValue);
