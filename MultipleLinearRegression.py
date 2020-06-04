import math;
import numpy as np;
from scipy.stats import t, f;


class MultipleLinearRegression:
    __DEFAULT_SIG_LEVEL = 0.05;

    def __init__(self):
        self.__n = None;
        self.__p = None;
        self.__avgX = None;

        self.__beta0 = None;
        self.__beta = None;

        self.__sigma = None;
        self.__residual = None;
        self.__rss = None;
        self.__r2 = None;
        self.__aic = None;
        self.__bic = None;
        self.__adjustedR2 = None;
        self.__c = None;
        self.__allF = None;
        self.__allP = None;
        self.__beta0Std = None;
        self.__beta0T = None;
        self.__beta0P = None;
        self.__beta0Value = None;
        self.__betaStd = None;
        self.__betaT = None;
        self.__betaP = None;
        self.__betaValue = None;

        self.__sigLevel = None;


    def __repr__(self):
        return "y = {0}{1}".format(
            self.__beta0Value,
            "".join([" {0} {1} * (x{2:.0f} - {3})".format("+" if item[1] >= 0 else "-", math.fabs(item[1]), item[0], item[2])
                     for item in
                     np.hstack((np.mat(range(1, self.__p + 1)).T, self.__betaValue, self.__avgX.T)).tolist()])
        );


    def __str__(self):
        return "y = β0{0}\r\n{1}\r\n{2}\r\n{3}".format(
            "".join(
                [" + β{0:.0f} * (x{0:.0f} - {1})".format(*item)
                 for item in
                 np.hstack((np.mat(range(1, self.__p + 1)).T, self.__avgX.T)).tolist()]),
            "β0 = {0}, std = {1}, t-value = {2} p-value = {3}".format(self.__beta0, self.__beta0Std, self.__beta0T, self.__beta0P),
            "\r\n".join(
                ["β{0:.0f} = {1}, std = {2}, t-value = {3}, p-value = {4}".format(*item)
                 for item in
                 np.hstack((np.mat(range(1, self.__p + 1)).T, self.__beta, self.__betaStd, self.__betaT, self.__betaP)).tolist()]),
            "σ = {0}, R^2 = {1}, AIC = {2}, BIC = {3}, adjusted R^2 = {4}, F-value = {5}, F p-value = {6}".format(self.__sigma, self.__r2, self.__aic, self.__bic, self.__adjustedR2, self.__allF, self.__allP)
        );


    @property
    def intercept(self):
        return self.__beta0Value - np.multiply(self.__betaValue, self.__avgX).sum();


    @property
    def beta(self):
        return self.__betaValue;


    @property
    def betaP(self):
        return self.__betaP;


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
    def aic(self):
        return self.__aic;


    @property
    def bic(self):
        return self.__bic;


    @property
    def adjustedR2(self):
        return self.__adjustedR2;


    @property
    def sigLevel(self):
        return self.__sigLevel;


    @sigLevel.setter
    def sigLevel(self, value):
        if self.__beta0P is None or self.__betaP is None or self.__allP is None:
            return;

        if value is None:
            value = MultipleLinearRegression.__DEFAULT_SIG_LEVEL;

        self.__sigLevel = value;
        self.__beta0Value = self.__beta0 if self.__beta0P < value else 0;
        self.__betaValue[(self.__betaP >= value).A.flatten(), :] = 0;
        if self.__allP >= value:
            self.__betaValue[:, :] = 0;


    def __getP(self, value, degree):
        if isinstance(value, np.matrix):
            return np.mat(2 * (1 - t.cdf(np.abs(value.A.flatten()), degree))).T;
        else:
            return 2 * (1 - t.cdf(math.fabs(value), degree));


    def fit(self, X, y):
        if X is None or y is None:
            raise ValueError();

        n, p = X.shape;
        avgX, avgY = X.mean(0), y.mean();
        centralizedX = X - avgX;
        centralizedY = y - avgY;
        C = (centralizedX.T * centralizedX).I;

        self.__n = n;
        self.__p = p;
        self.__avgX = avgX;
        self.__beta0 = avgY;
        self.__beta = C * centralizedX.T * y;

        residual = y - self.__beta0 - centralizedX * self.__beta;
        rss = (residual.T * residual)[0, 0];
        tss = (centralizedY.T * centralizedY)[0, 0];
        sigma2 = rss / (n - p - 1);

        self.__sigma = math.sqrt(sigma2);
        self.__residual = residual;
        self.__rss = rss;
        self.__r2 = 1 - rss / tss if tss != 0 else 0;
        self.__aic = (rss + 2 * p * sigma2) / (n * sigma2) + math.log(2 * math.pi * sigma2);
        self.__bic = (rss + math.log(n) * p * sigma2) / (n * sigma2) + math.log(2 * math.pi * sigma2);
        self.__adjustedR2 = 1 - rss * (n - 1) / (n - p - 1) / tss if tss != 0 else 0;
        self.__c = C;

        self.__beta0Std = self.__sigma / math.sqrt(n);
        self.__beta0T = self.__beta0 / self.__beta0Std if self.__beta0Std != 0 else math.nan;
        self.__beta0P = self.__getP(self.__beta0T, n - p - 1) if self.__beta0Std != 0 else 0;
        self.__beta0Value = self.__beta0;

        diagonal = C[range(0, p), range(0, p)].T;
        self.__betaStd = self.__sigma * np.sqrt(diagonal);
        self.__betaT = np.divide(self.__beta, self.__betaStd);
        self.__betaP = self.__getP(self.__betaT, n - p - 1);
        self.__betaValue = self.__beta.copy();

        self.__allF = (tss - rss) / p / self.__sigma ** 2;
        self.__allP = 1 - f.cdf(self.__allF, p, n - p - 1);


    def predictValue(self, X):
        if X is None:
            raise ValueError();

        return self.__beta0Value + (X - self.__avgX) * self.__betaValue;


    def predictInterval(self, X, confidence = None, prediction = True):
        if X is None:
            raise ValueError();
        if confidence is not None and (confidence <= 0 or confidence >= 1):
            raise ValueError();

        centralizedX = X - self.__avgX;
        alpha = 1 - confidence if confidence is not None else MultipleLinearRegression.__DEFAULT_SIG_LEVEL;
        tValue = t.ppf(1 - alpha / 2, self.__n - self.__p - 1);
        interval = np.sqrt((1 if prediction else 0) + 1 / self.__n + np.multiply(centralizedX * self.__c, centralizedX).sum(1)) * self.__sigma * tValue;
        value = self.predictValue(X);

        return np.mat(np.hstack((value - interval, value, value + interval)));


    def calcRss(self, X, y):
        residual = y - self.predictValue(X);
        return (residual.T * residual)[0, 0];
