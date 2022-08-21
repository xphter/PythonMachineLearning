import sys;
import enum;
import math;
import functools;
from typing import List, Tuple;

import numpy as np

from ImportNumpy import *;
from Functions import *;
from KMeans import *;
from Random import *;
import scipy.stats as stats;


class GaussianMixtureInitType(enum.IntEnum):
    UseKMeans = 0x00,
    UseRandom = 0x01,


class GaussianMixtureCovarianceType(enum.IntEnum):
    Default = 0x00,

    EachDiagonal = 0x11,
    EachSpherical = 0x12,

    AllIdentity = 0x21,


class GaussianMixture:
    @staticmethod
    def optimalK(X : np.ndarray, maxK : int, tol : float = 1e-3, initType : GaussianMixtureInitType = GaussianMixtureInitType.UseKMeans, covType : GaussianMixtureCovarianceType = GaussianMixtureCovarianceType.Default, maxEpoch : int = 1, maxIteration : int  = 100, outDebugInfo : bool = False, verbose : bool = False):
        model, gmm = None, None;

        for k in range(1, maxK + 1):
            model = GaussianMixture();
            model.fit(X, K = k, tol = tol, initType = initType, covType = covType, maxEpoch = maxEpoch, maxIteration = maxIteration, outDebugInfo = outDebugInfo, verbose = verbose);

            if verbose:
                print(f"the BIC of GMM with {k} components is {model.bic}");

            if gmm is not None and model.bic > gmm.bic:
                break;

            gmm = model;

        return gmm;


    def __init__(self, weight : np.ndarray = None, mu : np.ndarray = None, sigma : np.ndarray = None):
        self._K = len(weight) if weight is not None else 0;
        self._weight = weight;
        self._mu = mu;
        self._sigma = sigma;
        self._aic = sys.maxsize;
        self._bic = sys.maxsize;
        self._pdfEpsilon = 1e-100;
        self._singularEpsilon = 1e-8;


    @property
    def K(self) -> int:
        return self._K;


    @property
    def weight(self) -> np.ndarray:
        return self._weight;


    @property
    def mu(self) -> np.ndarray:
        return self._mu;


    @property
    def sigma(self) -> np.ndarray:
        return self._sigma;


    @property
    def aic(self) -> float:
        return self._aic;


    @property
    def bic(self) -> float:
        return self._bic;


    def _initParams(self, X : np.ndarray, initType : GaussianMixtureInitType, covType : GaussianMixtureCovarianceType) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        D, epsilon = X.shape[-1], 1e-3;
        weight = softmax(epsilon + np.random.rand(self._K));

        if initType == GaussianMixtureInitType.UseKMeans:
            currentIndex, distance, center = KMeans().clustering(X, self._K);
            mu = center;
        elif initType == GaussianMixtureInitType.UseRandom:
            mu = np.random.rand(self._K, D);
        else:
            raise ValueError(f"invalid init type: {initType}");

        if covType == GaussianMixtureCovarianceType.Default:
            sigma = np.zeros((self._K, D, D));

            for k in range(self._K):
                sigma[k] = randomPDM(D);
        else:
            raise NotImplementedError("");

        return weight, mu, sigma;


    def _normalPdf(self, X : np.ndarray, mu : np.ndarray, sigma : np.ndarray):
        Z = X - mu;
        return np.exp(-0.5 * np.sum((Z @ np.linalg.inv(sigma)) * Z, axis = -1)) / ( math.pow(2 * math.pi, X.shape[-1] / 2.0) * math.sqrt(np.linalg.det(sigma)));


    def _iterStep(self, X : np.ndarray, weight0 : np.ndarray, mu0 : np.ndarray, sigma0 : np.ndarray, covType : GaussianMixtureCovarianceType) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N, D = X.shape;

        W = np.zeros((N, self._K));
        for k in range(self._K):
            try:
                W[:, k] = weight0[k] * stats.multivariate_normal.pdf(X, mu0[k], sigma0[k]);
            except np.linalg.LinAlgError:
                sigma0[k] += self._singularEpsilon * np.eye(D);
                W[:, k] = weight0[k] * stats.multivariate_normal.pdf(X, mu0[k], sigma0[k]);

        W[np.sum(W, axis = -1) == 0] = 1 / self._K;
        W /= np.sum(W, axis = -1, keepdims = True);
        Nk = np.sum(W, axis = 0);

        weight1 = Nk / N;
        denominator = np.copy(Nk);
        denominator[Nk == 0] = 1;
        mu1 = (W.T @ X) / denominator.reshape(self._K, 1);
        # mu1 = (W.T @ X) / Nk.reshape(self._K, 1);
        sigma1 = np.zeros((self._K, D, D));

        if covType == GaussianMixtureCovarianceType.Default:
            for k in range(self._K):
                Z = X - mu1[k];
                sigma1[k] = ((W[:, k] * Z.T) @ Z) / Nk[k] if Nk[k] != 0 else np.eye(D);
        else:
            raise NotImplementedError();

        return W, weight1, mu1, sigma1;


    def _Q(self, X : np.ndarray, W : np.ndarray, weight : np.ndarray, mu : np.ndarray, sigma : np.ndarray) -> float:
        Nk = np.sum(W, axis = 0);
        Q = float(np.dot(Nk, np.log(weight) - 0.5 * np.array([math.log(np.linalg.det(item)) for item in sigma])));

        for k in range(self._K):
            Z = X - mu[k];
            Q -= 0.5 * float(np.sum(((W[:, k].reshape(-1, 1) * Z) @ np.linalg.inv(sigma[k])) * Z));

        return Q;


    def _pdf(self, X : np.ndarray, weight : np.ndarray, mu : np.ndarray, sigma : np.ndarray, fixSingular : bool) -> np.ndarray:
        D, P = X.shape[-1], [];

        for k in range(len(weight)):
            try:
                P.append(stats.multivariate_normal.pdf(X, mu[k], sigma[k]));
            except np.linalg.LinAlgError as ex:
                if fixSingular:
                    sigma[k] += self._singularEpsilon * np.eye(D);
                    P.append(stats.multivariate_normal.pdf(X, mu[k], sigma[k]));
                else:
                    raise ex;

        p = np.sum(np.column_stack(tuple(P)) * weight, axis = -1);
        p[p == 0] = self._pdfEpsilon;
        return p;


    def _logLikelihood(self, X : np.ndarray, weight : np.ndarray, mu : np.ndarray, sigma : np.ndarray, fixSingular : bool) -> float:
        return float(np.sum(np.log(self._pdf(X, weight, mu, sigma, fixSingular))));


    def _fillOnce(self, X : np.ndarray, K : int, tol : float, initType : GaussianMixtureInitType, covType : GaussianMixtureCovarianceType, maxIteration : int, outDebugInfo : bool) ->  Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], List[float]]:
        qValues, pValues = [], [];
        weight0, mu0, sigma0 = self._initParams(X, initType, covType);

        for i in range(maxIteration):
            try:
                W, weight1, mu1, sigma1 = self._iterStep(X, weight0, mu0, sigma0, covType);
            except Exception as ex:
                raise ex;

            if outDebugInfo:
                Q0 = self._Q(X, W, weight0, mu0, sigma0);
                Q1 = self._Q(X, W, weight1, mu1, sigma1);

                assert Q1 >= Q0, f"EM error, Q0: {Q0}, Q1: {Q1}";

                qValues.append(Q1);
                pValues.append(self._logLikelihood(X, weight1, mu1, sigma1, fixSingular = True));

            if math.sqrt(float(np.sum(np.square(weight1 - weight0)) + np.sum(np.square(mu1 - mu0)) + np.sum(np.square(sigma1 - sigma0)))) < tol:
                weight0, mu0, sigma0 = weight1, mu1, sigma1;
                break;

            weight0, mu0, sigma0 = weight1, mu1, sigma1;

        return weight0, mu0, sigma0, qValues, pValues;


    def setParams(self, weight : np.ndarray, mu : np.ndarray, sigma : np.ndarray):
        self._K = len(weight);
        self._weight = weight;
        self._mu = mu;
        self._sigma = sigma;


    def fit(self, X : np.ndarray, K : int = 1, tol : float = 1e-3, initType : GaussianMixtureInitType = GaussianMixtureInitType.UseKMeans, covType : GaussianMixtureCovarianceType = GaussianMixtureCovarianceType.Default, maxEpoch : int = 1, maxIteration : int  = 100, outDebugInfo : bool = False, verbose : bool = False) -> List[Tuple[List[float], List[float]]]:
        if X is None:
            raise ValueError("X is None");
        if X.ndim != 2:
            raise ValueError("X is not a matrix");

        results = [];
        self._K, N, D = K, len(X), X.shape[-1];

        for i in range(maxEpoch):
            weight, mu, sigma, qValues, pValues = self._fillOnce(X, K, tol, initType, covType, maxIteration, outDebugInfo);
            results.append((weight, mu, sigma, self._logLikelihood(X, weight, mu, sigma, fixSingular = True), qValues, pValues));

            if verbose:
                print(f"the log-likelihood of {i} epoch with {K} components is {results[-1][3]}");

        self._weight, self._mu, self._sigma, logLikelihood = max(results, key = lambda item: item[3])[: 4];
        self._aic = 2           * K * (1 + D + D ** 2) - 2 * logLikelihood;
        self._bic = math.log(N) * K * (1 + D + D ** 2) - 2 * logLikelihood;

        return [(item[-2], item[-1]) for item in results];


    def pdf(self, X : np.ndarray) -> np.ndarray:
        return self._pdf(X, self._weight, self._mu, self._sigma, fixSingular = False);


    def logLikelihood(self, X : np.ndarray) -> float:
        return self._logLikelihood(X, self._weight, self._mu, self._sigma, fixSingular = False);


    def sample(self, *shape : int) -> np.ndarray:
        D = self._mu.shape[-1];
        N = functools.reduce(lambda x, y: x * y, shape);
        Z = np.random.multinomial(1, self._weight, N);
        X = np.zeros((N, D));

        for k in range(self._K):
            idx = np.flatnonzero(Z[:, k] == 1);
            X[idx] = np.random.multivariate_normal(self._mu[k], self._sigma[k], size = len(idx), check_valid = "raise");

        return X.reshape(shape + (D, ));
