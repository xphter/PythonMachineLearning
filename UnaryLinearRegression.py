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
        self.__beta0T = None;
        self.__beta0P = None;
        self.__beta1Std = None;
        self.__beta1T = None;
        self.__beta1P = None;


    def __repr__(self):
        return "Y = {0} {3} {1} * (X - {2})".format(self.__beta0, math.fabs(self.__beta1), self.__avgX, "+" if self.__beta1 >= 0 else "-");


    def __str__(self):
        return "Y = beta0 + beta1 * (X - avgX)\r\n{0}\r\n{1}\r\n{2}".format(
            "beta0 = {0}, std = {1}, t-value = {2}, p-value = {3}".format(self.__beta0, self.__beta0Std, self.__beta0T, self.__beta0P),
            "beta1 = {0}, std = {1}, t-value = {2}, p-value = {3}".format(self.__beta1, self.__beta1Std, self.__beta1T, self.__beta1P),
            "sigma = {0}, R^2 = {1}".format(self.__sigma, self.__r2));


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


    def __getP(self, value, degree):
        return 2 * (1 - t.cdf(math.fabs(value), degree));


    def fit(self, x, y, sigLevel = None):
        if x is None or y is None:
            raise ValueError();

        if sigLevel is None:
            sigLevel = UnaryLinearRegression.__DEFAULT_SIG_LEVEL;

        n = x.shape[0];
        avgX, avgY = x.mean(), y.mean();
        centralizedX = x - avgX;
        centralizedY = y - avgY;
        sx2 = (centralizedX.T * centralizedX)[0, 0];

        self.__n = n;
        self.__avgX = avgX;
        self.__beta0 = avgY;
        self.__beta1 = (centralizedX.T * y)[0, 0] / sx2;

        residual = y - self.__beta0 - self.__beta1 * centralizedX;
        rss = (residual.T * residual)[0, 0];
        tss = (centralizedY.T * centralizedY)[0, 0];

        self.__sigma = math.sqrt(rss / (n - 2));
        self.__rss = rss;
        self.__r2 = 1 - rss/tss if tss != 0 else 0;
        self.__beta0Std = self.__sigma / math.sqrt(n);
        self.__beta0T = self.__beta0 / self.__beta0Std if self.__beta0Std != 0 else math.nan;
        self.__beta0P = self.__getP(self.__beta0T, n - 2) if self.__beta0Std != 0 else 0;
        if self.__beta0P > sigLevel:
            self.__beta0 = 0;

        self.__beta1Std = self.__sigma / math.sqrt(sx2);
        self.__beta1T = self.__beta1 / self.__beta1Std if self.__beta1Std != 0 else math.nan;
        self.__beta1P = self.__getP(self.__beta1T, n - 2) if self.__beta1Std != 0 else 0;
        if self.__beta1P > sigLevel:
            self.__beta1 = 0;


    def predict(self, x):
        if x is None:
            raise ValueError();

        return self.__beta0 + self.__beta1 * (x - self.__avgX);


    def inverse(self, y):
        if y is None:
            raise ValueError();
        if self.__beta1 == 0:
            raise NotImplementedError();

        return (y - self.__beta0) / self.__beta1 + self.__avgX;


    def calcRss(self, x, y):
        residual = y - self.predict(x);
        return (residual.T * residual)[0, 0];