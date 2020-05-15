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
        self.__residual = None;
        self.__rss = None;
        self.__r2 = None;
        self.__sx2 = None;
        self.__beta0Std = None;
        self.__beta0T = None;
        self.__beta0P = None;
        self.__beta0Value = None;
        self.__beta1Std = None;
        self.__beta1T = None;
        self.__beta1P = None;
        self.__beta1Value = None;

        self.__sigLevel = None;


    def __repr__(self):
        return "y = {0} {3} {1} * (x - {2})".format(self.__beta0Value, math.fabs(self.__beta1Value), self.__avgX, "+" if self.__beta1Value >= 0 else "-");


    def __str__(self):
        return "y = β0 + β1 * (x - {0})\r\n{1}\r\n{2}\r\n{3}".format(
            self.__avgX,
            "β0 = {0}, std = {1}, t-value = {2}, p-value = {3}".format(self.__beta0, self.__beta0Std, self.__beta0T, self.__beta0P),
            "β1 = {0}, std = {1}, t-value = {2}, p-value = {3}".format(self.__beta1, self.__beta1Std, self.__beta1T, self.__beta1P),
            "σ = {0}, R^2 = {1}".format(self.__sigma, self.__r2)
        );


    @property
    def intercept(self):
        return self.__beta0Value - self.__beta1Value * self.__avgX;


    @property
    def slop(self):
        return self.__beta1Value;


    @property
    def slopP(self):
        return self.__beta1P;


    @property
    def sigma(self):
        return self.__sigma;


    @property
    def residual(self):
        return self.__residual;


    @property
    def rss(self):
        return self.__rss;


    @property
    def r2(self):
        return self.__r2;


    @property
    def sigLevel(self):
        return self.__sigLevel;


    @sigLevel.setter
    def sigLevel(self, value):
        if self.__beta0P is None or self.__beta1P is None:
            return;

        if value is None:
            value = UnaryLinearRegression.__DEFAULT_SIG_LEVEL;

        self.__sigLevel = value;
        self.__beta0Value = self.__beta0 if self.__beta0P < value else 0;
        self.__beta1Value = self.__beta1 if self.__beta1P < value else 0;


    def __getP(self, value, degree):
        return 2 * (1 - t.cdf(math.fabs(value), degree));


    def fit(self, x, y):
        if x is None or y is None:
            raise ValueError();

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
        self.__residual = residual;
        self.__rss = rss;
        self.__r2 = 1 - rss/tss if tss != 0 else 0;
        self.__sx2 = sx2;
        self.__beta0Std = self.__sigma / math.sqrt(n);
        self.__beta0T = self.__beta0 / self.__beta0Std if self.__beta0Std != 0 else math.nan;
        self.__beta0P = self.__getP(self.__beta0T, n - 2) if self.__beta0Std != 0 else 0;
        self.__beta0Value = self.__beta0;

        self.__beta1Std = self.__sigma / math.sqrt(sx2);
        self.__beta1T = self.__beta1 / self.__beta1Std if self.__beta1Std != 0 else math.nan;
        self.__beta1P = self.__getP(self.__beta1T, n - 2) if self.__beta1Std != 0 else 0;
        self.__beta1Value = self.__beta1;


    def predictValue(self, x):
        if x is None:
            raise ValueError();

        return self.__beta0Value + self.__beta1Value * (x - self.__avgX);


    def predictInterval(self, x, confidence = None, prediction = True):
        if x is None:
            raise ValueError();
        if confidence is not None and (confidence <= 0 or confidence >= 1):
            raise ValueError();

        alpha = 1 - confidence if confidence is not None else UnaryLinearRegression.__DEFAULT_SIG_LEVEL;
        tValue = t.ppf(1 - alpha / 2, self.__n - 2);
        interval = np.sqrt((1 if prediction else 0) + 1 / self.__n + np.power(x - self.__avgX, 2) / self.__sx2) * self.__sigma * tValue;
        value = self.predictValue(x);

        return np.mat(np.hstack((value - interval, value, value + interval)));


    def inverse(self, y):
        if y is None:
            raise ValueError();
        if self.__beta1Value == 0:
            raise NotImplementedError();

        return (y - self.__beta0Value) / self.__beta1Value + self.__avgX;


    def calcRss(self, x, y):
        residual = y - self.predictValue(x);
        return (residual.T * residual)[0, 0];
