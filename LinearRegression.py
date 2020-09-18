import abc;
import math;
import multiprocessing;
import psutil;
import numpy as np;
from scipy.stats import t, f;

import DataHelper;


class LinearRegression:
    __DEFAULT_SIG_LEVEL = 0.05;


    @staticmethod
    def calcVIF(X):
        if X is None:
            raise ValueError("matrix X is None");

        return [1 / (1 - LinearRegression().fit(np.delete(X, i, 1), X[:, i]).r2) for i in range(0, X.shape[1])];


    @staticmethod
    def _optimalSubsetsCore(X, y, indices):
        return indices, LinearRegression().fit(X[:, indices], y);


    @staticmethod
    def optimalSubsets(X, y, m = None):
        if X is None or y is None:
            raise ValueError("matrix X or vector y is None");

        result = [];
        p = X.shape[1];

        if m is not None and (m < 1 or m > p):
            raise ValueError("m must be between 1 and column numbers of X");

        # number of models when m is null: 2^p
        if p <= 14 or m is not None:
            result.extend([min([LinearRegression._optimalSubsetsCore(X, y, indices) for indices in DataHelper.combinations(p, k)], key = lambda item: item[1].rss) for k in (range(1, p + 1) if m is None else range(m, m + 1))]);
        else:
            data, models = [], None;

            for k in range(1, p + 1):
                data.extend([(X, y, indices) for indices in DataHelper.combinations(p, k)]);
            data = list(map(tuple, np.array(data, np.object)[DataHelper.randomArrangement(len(data)), :].tolist()));

            with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
                models = pool.starmap(LinearRegression._optimalSubsetsCore, data);

            for k in range(1, p + 1):
                result.append(min([item for item in models if len(item[0]) == k], key = lambda item: item[1].rss));

        # result item format: (indices, model)
        return result;


    @staticmethod
    def forwardSelection(X, y):
        if X is None or y is None:
            raise ValueError("matrix X or vector y is None");

        result = [];
        p = X.shape[1];
        current, leftover = [], list(range(0, p));

        # number of models: p * (p + 1) / 2
        for k in range(1, p + 1):
            result.append(min([(current + [i], LinearRegression().fit(X[:, current + [i]], y)) for i in leftover], key = lambda item: item[1].rss));
            current = result[len(result) - 1][0];
            leftover.remove(current[len(current) - 1]);

        # result item format: (indices, model)
        return result;


    @staticmethod
    def backwardSelection(X, y):
        if X is None or y is None:
            raise ValueError("matrix X and vector y is None");

        result = [];
        p = X.shape[1];
        leftover = set(range(0, p));

        # number of models: p * (p + 1) / 2
        result.append((list(leftover), LinearRegression().fit(X, y)));

        for k in range(2, p + 1):
            result.append(min([(list(leftover - {i}), LinearRegression().fit(X[:, list(leftover - {i})], y)) for i in leftover], key = lambda item: item[1].rss));
            leftover = set(result[len(result) - 1][0]);

        # result item format: (indices, model)
        return result;


    @staticmethod
    def crossValidation(X, y, k):
        if X is None or y is None:
            raise ValueError("matrix X and vector y is None");

        mse = np.mat([list(map(lambda item: item[1].calcMse(testX[:, item[0]], testY), LinearRegression.optimalSubsets(trainX, trainY))) for trainX, trainY, testX, testY in DataHelper.foldOutSampling(X, y, k)]);

        return LinearRegression.optimalSubsets(X, y, np.argmin(mse.mean(0).A.flatten()) + 1)[0];


    def __init__(self):
        self.__basisFunctions = None;
        self.__n = None;
        self.__p = None;
        self.__beta = None;

        self.__sigma = None;
        self.__residual = None;
        self.__rss = None;
        self.__r2 = None;
        self.__cp = None;
        self.__aic = None;
        self.__bic = None;
        self.__adjustedR2 = None;
        self.__c = None;
        self.__allF = None;
        self.__allP = None;
        self.__betaStd = None;
        self.__betaT = None;
        self.__betaP = None;
        self.__betaValue = None;

        self.__sigLevel = None;


    def __repr__(self):
        return "y = {0}{1}".format(
            self.__beta[0, 0],
            "".join([" {0} {1} * x{2:.0f}".format("+" if item[0] >= 0 else "-", math.fabs(item[0]), item[1])
                     for item in
                     np.hstack((self.__betaValue, np.mat(range(1, self.__p)).T)).tolist()])
        );


    def __str__(self):
        return "y = β0{0}\r\n{1}\r\n{2}".format(
            "".join(
                [" + β{0:.0f} * x{0:.0f}".format(item) for item in list(range(1, self.__p))]),
            "\r\n".join(
                ["β{0:.0f} = {1}, std = {2}, t-value = {3}, p-value = {4}".format(*item)
                 for item in
                 np.hstack((np.mat(range(0, self.__p)).T, self.__beta, self.__betaStd, self.__betaT, self.__betaP)).tolist()]),
            "σ = {0}, R^2 = {1}, Cp = {2}, AIC = {3}, BIC = {4}, adjusted R^2 = {5}, F-value = {6}, F p-value = {7}".format(self.__sigma, self.__r2, self.__cp, self.__aic, self.__bic, self.__adjustedR2, self.__allF, self.__allP)
        );


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
    def rssDf(self):
        return self.__n - self.__p;


    @property
    def mse(self):
        return self.__rss / self.__n;


    @property
    def r2(self):
        return self.__r2;


    @property
    def cp(self):
        return self.__cp;


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
        if self.__betaP is None or self.__allP is None:
            return;

        if value is None:
            value = LinearRegression.__DEFAULT_SIG_LEVEL;

        self.__sigLevel = value;
        self.__betaValue[(self.__betaP >= value).A.flatten(), :] = 0;
        if self.__allP >= value:
            self.__betaValue[:, :] = 0;


    def __getX(self, X):
        dataSet = X;

        if self.__basisFunctions is not None:
            dataSet = np.hstack(tuple([self.__basisFunctions[j].getX(X[:, j]) if self.__basisFunctions[j] is not None else X[:, j] for j in range(X.shape[1])]));

        return np.hstack((np.mat(np.ones((dataSet.shape[0], 1))), dataSet));


    def __getP(self, value, degree):
        if isinstance(value, np.matrix):
            return np.mat(2 * (1 - t.cdf(np.abs(value.A.flatten()), degree))).T;
        else:
            return 2 * (1 - t.cdf(math.fabs(value), degree));


    def __predict(self, X):
        return X * self.__betaValue;


    def fit(self, X, y, baseFunctions = None, w = None):
        # w is the weight vector
        if X is None or y is None:
            raise ValueError("matrix X or vector y is None");
        if baseFunctions is not None and len(baseFunctions) != X.shape[1]:
            raise ValueError("the length of base functions must be equals to column numbers of X");
        if w is not None and w.shape[0] != X.shape[0]:
            raise ValueError("the length of weight vector must be equals to row numbers of X");

        self.__basisFunctions = baseFunctions;

        X = self.__getX(X);
        n, p = X.shape;
        W = np.diag(w) if w is not None else np.identity(n);
        C = (X.T * W * X).I;

        self.__n = n;
        self.__p = p;
        self.__beta = C * X.T * W * y;

        centralizedY = y - y.mean();
        residual = y - X * self.__beta;
        rss = (residual.T * residual)[0, 0];
        tss = (centralizedY.T * centralizedY)[0, 0];
        sigma2 = rss / (n - p);

        self.__sigma = math.sqrt(sigma2);
        self.__residual = residual;
        self.__rss = rss;
        self.__r2 = 1 - rss / tss if tss != 0 else 0;
        self.__cp = (rss + 2 * (p - 1) * sigma2) / n;
        # self.__aic = (rss + 2 * (p - 1) * sigma2) / (n * sigma2) + math.log(2 * math.pi * sigma2);
        # self.__bic = (rss + math.log(n) * (p - 1) * sigma2) / (n * sigma2) + math.log(2 * math.pi * sigma2);
        self.__aic =           2 * (p - 1) + n - p + n * math.log(2 * math.pi * sigma2);
        self.__bic = math.log(n) * (p - 1) + n - p + n * math.log(2 * math.pi * sigma2);
        self.__adjustedR2 = 1 - (rss / (n - p)) / (tss / (n - 1)) if tss != 0 else 0;
        self.__c = C;

        self.__betaStd = self.__sigma * np.sqrt(C.diagonal().T);
        self.__betaT = np.divide(self.__beta, self.__betaStd);
        self.__betaP = self.__getP(self.__betaT, n - p);
        self.__betaValue = self.__beta.copy();

        self.__allF = (tss - rss) / (p - 1) / self.__sigma ** 2;
        self.__allP = 1 - f.cdf(self.__allF, p - 1, n - p);

        return self;


    def predictValue(self, X):
        if X is None:
            raise ValueError("matrix X is None");

        return self.__predict(self.__getX(X));


    def predictInterval(self, X, confidence = None, prediction = True):
        if X is None:
            raise ValueError("matrix X is None");
        if confidence is not None and (confidence <= 0 or confidence >= 1):
            raise ValueError("the confidence must be between 0 and 1");

        X = self.__getX(X);
        alpha = 1 - confidence if confidence is not None else LinearRegression.__DEFAULT_SIG_LEVEL;
        tValue = t.ppf(1 - alpha / 2, self.__n - self.__p);
        interval = np.sqrt((1 if prediction else 0) + np.multiply(X * self.__c, X).sum(1)) * self.__sigma * tValue;
        value = self.__predict(X);

        return np.mat(np.hstack((value - interval, value, value + interval)));


    def calcRss(self, X, y):
        if X is None or y is None:
            raise ValueError("matrix X or vector y is None");

        residual = y - self.predictValue(X);
        return (residual.T * residual)[0, 0];


    def calcMse(self, X, y):
        if X is None or y is None:
            raise ValueError("matrix X or vector y is None");

        return self.calcRss(X, y) / X.shape[0];


class IBasisFunction(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def getX(self, x):
        pass;


'''
       d
f(x) = Σ βj * x^d
      j=0
degree of freedom = d + 1
'''


class PolynomialFunction(IBasisFunction):
    # df excludes intercept
    def __init__(self, df):
        if df < 1:
            raise ValueError("df must be at least 1");

        self.__d = df;

    def getX(self, x):
        if x is None:
            raise ValueError("vector x is None");

        return DataHelper.vectorPoly(x, self.__d);


class KnottedFunction(IBasisFunction, metaclass = abc.ABCMeta):
    def __init__(self, k):
        if k < 0:
            raise ValueError("number of knots must be at least 0");

        self._K = k;
        self._knots = None;


    @property
    def knots(self):
        return self._knots;


    def _findKnots(self, x):
        self._knots = [np.quantile(x, k / (self._K + 1), 0)[0] for k in range(1, self._K + 1)] if self._K > 0 else [];


    @abc.abstractmethod
    def _getX(self, x):
        pass;


    def getX(self, x):
        if x is None:
            raise ValueError("vector x is None");

        if self._knots is None:
            self._findKnots(x);

        return self._getX(x);


'''
            K
f(x) = β0 + Σβk * I(Ck <= x < Ck+1)
           k=1
degree of freedom = K + 1
'''


class StepFunction(KnottedFunction):
    # df excludes intercept
    def __init__(self, df):
        if df < 1:
            raise ValueError("df must be at least 1");

        super().__init__(df);


    def _getX(self, x):
        return np.hstack(tuple([(np.logical_and(x >= self._knots[i], x < self._knots[i + 1]) if i < len(self._knots) - 1 else x >= self._knots[i]) - 0 for i in range(0, self._K)]));


'''
      M-1               K
f(x) = Σ βj * x^(j-1) + Σ θk * (x-ξk)+^(M-1)
      j=0              k=1
f, f', f'', ... d^(M-2)f is continuous at  ξk, k = 1, 2, ..., K
degree of freedom = M + K
the default is cubic spline with M = 4.
'''


class RegressionSplineFunction(KnottedFunction):
    # df excludes intercept
    def __init__(self, df, m = 4):
        if m < 1:
            raise ValueError("m must be at least 1");
        if df < 1:
            raise ValueError("df must be at least 1");

        super().__init__(max(df + 1 - m, 0));

        self.__M = m;
        self.__df = df;


    def _getX(self, x):
        d = self.__M - 1;

        if self._K > 0:
            Y = np.hstack(tuple([DataHelper.truncatedPower(x, self._knots[k], d) for k in range(0, self._K)]));

            return np.hstack((DataHelper.vectorPoly(x, d), Y)) if d > 0 else Y;
        else:
            return DataHelper.vectorPoly(x, self.__df);


'''
              K-2
f = β0 + β1x + Σ θj * (ξK - ξj) * [d(j, x) - d(K-1, x)]
              j=1
d(j, x) = [(x - ξj)+^3 - (x - ξK)+^3] / (ξK - ξj)
f''(x) = 0, when x ∈ (-∞, ξ1] ∪ [ξK,  ∞)
degree of freedom = K
when K = 1 and 2, f(x) = β0 + β1x.
'''


class NatureCubicSplineFunction(KnottedFunction):
    # df excludes intercept
    def __init__(self, df):
        if df < 0:
            raise ValueError("df must be at least 0");

        super().__init__(df + 1);


    def __d(self, k, x):
        return (DataHelper.truncatedPower(x, self._knots[k], 3) - DataHelper.truncatedPower(x, self._knots[self._K - 1], 3)) / (self._knots[self._K - 1] - self._knots[k]);


    def _getX(self, x):
        if self._K > 2:
            dK_1 = self.__d(self._K - 2, x);

            return np.hstack(tuple([x] + [(self._knots[self._K - 1] - self.knots[k]) * (self.__d(k, x) - dK_1) for k in range(0, self._K - 2)]));
        else:
            return x;
