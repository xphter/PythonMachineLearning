import enum;
import math;
import numpy as np;

from LinearRegression import *;


class WeibullDistributionFitMethod(enum.IntEnum):
    MLE = 0x10,
    OLS = 0x20,


class WeibullDistribution:
    def __init__(self, beta : float = None, eta : float = None):
        self._beta = eta;
        self._eta = eta;


    @property
    def beta(self) -> float:
        return self._beta;


    @property
    def eta(self) -> float:
        return self._eta;


    def _getLikelihood(self, x : np.ndarray, theta : np.ndarray) -> float:
        n = len(x);
        beta, eta = theta[0], theta[1];

        return n * (math.log(beta) - beta * math.log(eta)) + (beta - 1) * np.sum(np.log(x)) - np.sum(np.power(x / eta, beta));


    def _getGradient(self, x : np.ndarray, theta : np.ndarray) -> np.ndarray:
        n = len(x);
        beta, eta = theta[0], theta[1];
        dBeta = n * (1 / beta - math.log(eta)) + np.sum(np.log(x)) - np.sum(np.power(x / eta, beta) * np.log(x / eta));
        dEta = (np.sum(np.power(x / eta, beta)) - n) * beta / eta;

        return np.array([dBeta, dEta]).reshape(2, 1);


    def _getHessian(self, x : np.ndarray, theta : np.ndarray) -> np.ndarray:
        n = len(x);
        beta, eta = theta[0], theta[1];
        dBeta2 = -1 * (n / (beta ** 2) + np.sum(np.power(x / eta, beta) * np.square(np.log(x / eta))));
        dEta2 = (n - (1 + beta) * np.sum(np.power(x / eta, beta))) * beta / (eta ** 2);
        dBetaEta = (np.sum(np.power(x / eta, beta) * (1 + beta * np.log(x / eta))) - n) / eta;

        return np.array([dBeta2, dBetaEta, dBetaEta, dEta2]).reshape(2, 2);


    def _fitOLS(self, x : np.ndarray):
        x = np.sort(x, axis = None);
        n = len(x);

        lr = LinearRegression();
        lr.fit(np.mat(np.log(x).reshape(-1, 1)), np.mat(np.array([math.log(-math.log(1 - (k + 1) / (n + 1))) for k in range(n)]).reshape(-1, 1)));

        self._beta = lr.beta[1, 0];
        self._eta = math.exp(-lr.beta[0, 0] / lr.beta[1, 0]);


    def _fitMLE(self, x : np.ndarray, epsilon : float = 10e-6, verbose : bool = False):
        theta = 0.001 * np.random.rand(2);

        count = 0;
        g = self._getGradient(x, theta);

        while np.linalg.norm(g, ord = None) > epsilon:
            H = self._getHessian(x, theta);

            theta -= (np.linalg.inv(H) @ g).flatten();

            if np.any(theta < 0):
                theta = 0.001 * np.random.rand(2);

            g = self._getGradient(x, theta);
            l = self._getLikelihood(x, theta);

            count += 1;
            if verbose:
                print(f"iteration {count}, log-likelihood: {l}, gradient: {g.flatten()}");

        self._beta, self._eta = theta[0], theta[1];


    def fit(self, x : np.ndarray, method : WeibullDistributionFitMethod = WeibullDistributionFitMethod.MLE, epsilon : float = 10e-6, verbose : bool = False):
        if x is None or len(x) == 0:
            raise ValueError("x is none or empty");

        if method == WeibullDistributionFitMethod.OLS:
            self._fitOLS(x);
        elif method == WeibullDistributionFitMethod.MLE:
            self._fitMLE(x, epsilon, verbose);
        else:
            raise ValueError(f"invalid fit method: {method}");
