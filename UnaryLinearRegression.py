import math;
import numpy as np;
from scipy.stats import t;


class UnaryLinearRegression:
    __DEFAULT_SIG_LEVEL = 0.05;

    def __init__(self):
        self.__n = None;
        self.__avgX = None;

        self.__beta0 = None;
        self.__beta1 = None;

        self.__sigma = None;
        self.__rss = None;
        self.__r2 = None;
        self.__beta0Std = None;
        self.__beta1Std = None;
        self.__beta1P = None;


    def __repr__(self):
        return "Y = {0} {3} {1} * (X - {2})".format(self.__beta0, math.fabs(self.__beta1), self.__avgX, "+" if self.__beta1 >= 0 else "-");


    def __str__(self):
        return "Y = beta0 + beta1 * (X - avgX)\r\nbeta0 = {0}, std = {1}\r\nbeta1 = {2}, std = {3}, p = {4}\r\nsigma = {5}, R^2 = {6}".format(self.__beta0, self.__beta0Std, self.__beta1, self.__beta1Std, self.__beta1P, self.__sigma, self.__r2);


    @property
    def intercept(self):
        return self.__beta0 - self.__beta1 * self.__avgX;


    @property
    def slop(self):
        return self.__beta1;


    @property
    def rss(self):
        return self.__rss;


    @property
    def r2(self):
        return self.__r2;


    def fit(self, X, Y, sigLevel = None):
        if X is None or Y is None:
            raise ValueError();

        if sigLevel is None:
            sigLevel = UnaryLinearRegression.__DEFAULT_SIG_LEVEL;

        n = X.shape[0];
        avgX, avgY = X.mean(), Y.mean();
        centralizedX = X - avgX;
        centralizedY = Y - avgY;
        sx2 = (centralizedX.T * centralizedX)[0, 0];

        self.__n = n;
        self.__avgX = avgX;
        self.__beta0 = avgY;
        self.__beta1 = (centralizedX.T * Y)[0, 0] / sx2;

        residual = Y - self.__beta0 - self.__beta1 * centralizedX;
        rss = (residual.T * residual)[0, 0];
        tss = (centralizedY.T * centralizedY)[0, 0];

        self.__sigma = math.sqrt(rss / (n - 2));
        self.__rss = rss;
        self.__r2 = 1 - rss/tss if tss != 0 else 0;
        self.__beta0Std = self.__sigma / math.sqrt(n);
        self.__beta1Std = self.__sigma / math.sqrt(sx2);
        self.__beta1P = 2 * (1 - t.cdf(math.fabs(self.__beta1 / self.__beta1Std), n - 2)) if self.__beta1Std != 0 else 0;
        if self.__beta1P > sigLevel:
            self.__beta1 = 0;


    def predict(self, X):
        if X is None:
            raise ValueError();

        return self.__beta0 + self.__beta1 * (X - self.__avgX);


    def inverse(self, Y):
        if Y is None:
            raise ValueError();
        if self.__beta1 == 0:
            raise NotImplementedError();

        return (Y - self.__beta0) / self.__beta1 + self.__avgX;


    def calcRss(self, X, Y):
        residual = Y - self.predict(X);
        return (residual.T * residual)[0, 0];
