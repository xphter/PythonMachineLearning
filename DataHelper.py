import math;
import random;
import numpy as np;
import numpy.linalg as npl;
import numpy.matlib as npm;
from scipy.stats import norm;
from scipy.stats import chi2;
from scipy.special import comb;


def __groupByLabel(X):
    label = None;
    classes = {};

    for row in X:
        label = row[0, -1];

        if label not in classes:
            classes[label] = [row.A.flatten()];
        else:
            classes[label].append(row.A.flatten());

    return classes;


# get a random arrangement
def randomArrangement(n):
    array = list(range(n));

    for i in range(n-1, -1, -1):
        j = math.ceil(random.random() * (i + 1) % (i + 1));

        array[i], array[j] = array[j], array[i];

    return array;


# transfer to standard variable, mean is 0, std is 1
def normalizeFeatures(X):
    mu = np.mat(X.mean(0));
    Y = X - mu;

    sigma = np.mat(Y.std(0));
    sigma[sigma == 0] = 1;
    Y = Y / sigma;

    return Y, mu, sigma;


def performNormalize(X, mu, sigma):
    return (X - mu) / sigma;


# for common dataset
def calcMinkowskiDistance(X, v, p = 2):
    return np.power(np.power(np.abs(X - v), p).sum(1), 1/p);


def calcManhattanDistance(X, v):
    return np.abs(X - v).sum(1);


# output: L2^2
def calcEuclideanDistance(X, Y):
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y have different number of columns");

    if Y.shape[0] == 1:
        return np.square(X - Y).sum(1);
    else:
        return -2 * X * Y.T + np.square(X).sum(1) + np.square(Y).sum(1).T;


def calcChebyshevDistance(X, v):
    return np.abs(X - v).max(1);


# only for normal distribution
def calcMahalanobisDistance(X, v, inverseSigma):
    Y = X - v;

    return np.multiply(Y * inverseSigma, Y).sum(1);


# only for binary dataset
def calcJaccardCoefficient(X, v):
    return (X & v).sum(1) / (X | v).sum(1);


# for sparse dataset
def calcCosine(X, v):
    return np.divide(X * v.T, np.sqrt(np.power(X, 2).sum(1) * (v * v.T)[0, 0]));
    # return np.dot(X / np.mat(list(map(npl.norm, X))).T, v.T / npl.norm(v));


def __calcAcf(x, n, k, sigma2):
    # x must be centralized
    if k == 0:
        return 1;

    x1 = x[:n - k, :];
    x2 = x[k:, :];

    return (x1.T * x2)[0, 0] / sigma2;


def calcAcf(x, k):
    n = x.shape[0];
    centralizedX = x - x.mean();
    sigma2 = (centralizedX.T * centralizedX)[0, 0];

    return __calcAcf(centralizedX, n, k, sigma2);


def getExistingMedian(X, floor = False):
    uniqueList = [];
    medianList = [];
    length, uniqueVector = None, None;

    for j in range(0, X.shape[1]):
        uniqueList.append(np.unique(X[:, j].T.A.flatten()));

    for i in range(0, len(uniqueList)):
        uniqueVector = uniqueList[i];
        length = len(uniqueVector);

        if length % 2 == 0:
            medianList.append(uniqueVector[int((length / 2 - 1) + (0 if floor else 1))]);
        else:
            medianList.append(uniqueVector[int((length + 1) / 2 - 1)]);

    return np.mat(medianList);


def choiceProportional(proportions):
    total, index = 0, -1;
    r = proportions.sum() * np.random.random();

    if r == 0:
        return 0;

    while total < r:
        index += 1;
        total += proportions[index];

    return index;


def holdOutSampling(X, proportion):
    indices = None;
    trainSet, testSet = [], [];
    classes = __groupByLabel(X);

    for subSet in classes.values():
        indices = random.sample(list(range(0, len(subSet))), math.ceil(proportion * len(subSet)));

        trainSet.extend([subSet[index] for index in indices]);
        testSet.extend([subSet[index] for index in range(0, len(subSet)) if index not in indices]);

    return np.mat(trainSet), np.mat(testSet);


def bootstrapSampling(X):
    n = X.shape[0];
    return X[np.random.choice(n, n, True), :];


def statisticFrequency(v):
    result = {};

    for item in v.A.flatten():
        if item in result:
            result[item] += 1;
        else:
            result[item] = 1;

    return result;


def groupBy(X, featureIndex):
    result = {};
    featureValue = None;

    for row in X:
        featureValue = row[0, featureIndex];

        if featureValue in result:
            result[featureValue] = np.vstack((result[featureValue], row));
        else:
            result[featureValue] = row;

    return result;


# create a matrix with columns: x^n1, x^n2, ..., x^np
def vectorPoly(x, maxDegree, minDegree = 1):
    return np.mat([np.power(x, i).A.flatten() for i in range(minDegree, maxDegree + 1)]).T if maxDegree > 1 else x;


# Validation Set Approach
def validationSetSampling(X, y, size = None):
    if size is None:
        size = math.ceil(X.shape[0] / 2);

    indices = np.random.choice(X.shape[0], size, replace = False).tolist();

    return X[indices, :], y[indices, :], np.delete(X, indices, 0), np.delete(y, indices, 0)


# Leave One Out Cross Validation
def oneOutSampling(X, y):
    return [(np.delete(X, i, 0), np.delete(y, i, 0), X[i, :], y[i, :]) for i in range(0, X.shape[0])];


# k Fold Cross Validation
def foldOutSampling(X, y, k):
    n = X.shape[0];
    if n / k < 1:
        raise ValueError();

    size = math.ceil(n / k);
    arrangement = randomArrangement(n);

    result = [(np.delete(X, arrangement[i * size: (i + 1) * size], 0),
               np.delete(y, arrangement[i * size: (i + 1) * size], 0),
               X[arrangement[i * size: (i + 1) * size], :],
               y[arrangement[i * size: (i + 1) * size], :]) for i in range(0, k - 1)];
    result.append((np.delete(X, arrangement[(k - 1) * size:], 0),
                   np.delete(y, arrangement[(k - 1) * size:], 0),
                   X[arrangement[(k - 1) * size:], :],
                   y[arrangement[(k - 1) * size:], :]));

    return result;


# C(n, r) = A(n, r) / r! = n! / (n - r)! / r!
def combinations(n, r):
    if n <= 0 or r < 0 or r > n:
        raise ValueError();

    if r == 0:
        return [];
    if r == 1:
        return [[i] for i in range(0, n)];
    if r == n:
        return [list(range(0, n))];

    first = list(range(0, r));
    result, previous, current = [first], first, None;

    while True:
        indices = [i for i in first if previous[i] < n - r + i];
        if len(indices) == 0:
            break;

        j = max(indices);
        value = previous[j];
        current = previous[:];

        for i in range(j, r):
            value += 1;
            current[i] = value;

        previous = current;
        result.append(current);

    return result;


# return (x - value)^d when x > value, else 0.
def truncatedPower(x, value, d):
    if x is None:
        raise ValueError("vector x is None");

    return np.multiply((x > value) - 0, np.power(x - value, d));


# return p-value
def testNormalDistribution(x):
    if x is None:
        raise ValueError("vector x is None");

    n = x.shape[0];
    mu, std = x.mean(), x.std();
    skew = np.power(x - mu, 3).sum() / n / std ** 3;
    kurt = np.power(x - mu, 4).sum() / n / std ** 4 - 3;
    K = skew ** 2 * (n + 1) * (n + 3) / (6 * (n - 2));
    K += (kurt + 6 / (n + 1)) ** 2 * (n + 1) ** 2 * (n + 3) * (n + 5) / (24 * n * (n - 2) * (n - 3));

    return 1 - chi2.cdf(K, 2);


# return p-value
def testWhiteNoise(x, m):
    # x must be centralized
    n = x.shape[0];
    sigma2 = (x.T * x)[0, 0];

    if sigma2 == 0:
        return 1;

    rho = np.mat([__calcAcf(x, n, k, sigma2) for k in range(0, m + 1)]).T;

    Q = n * (n + 2) * ((1 / np.mat(range(n - 1, n - m - 1, -1))) * np.power(rho[1:, 0], 2))[0, 0];

    return 1 - chi2.cdf(Q, m);


# return p-value
def testRunsLeft(x):
    if x is None or x.shape[0] == 0:
        raise ValueError("x is none or empty");

    if not np.all(np.logical_or(x == 0, x == 1)):
        raise ValueError("x should only contains 0 or 1");

    n, n0, n1 = x.shape[0], (x == 0).sum(), (x == 1).sum();
    if n0 == 0 or n1 == 0:
        return 0;

    r, current = 1, x[0, 0];
    for i in range(1, x.shape[0]):
        if x[i, 0] != current:
            r += 1;
            current = x[i, 0];

    p = 0;

    if n0 <= 30 and n1 <= 30:
        p += sum([2 * comb(n0 - 1, k - 1) * comb(n1 - 1, k - 1) for k in range(1, math.floor(r / 2) + 1)]);
        p += sum([comb(n0 - 1, k - 1) * comb(n1 - 1, k) + comb(n0 - 1, k) * comb(n1 - 1, k - 1) for k in range(1, math.floor((r - 1) / 2) + 1)]);
        p /= comb(n, n0);
    else:
        mu = 2 * n0 * n1 / n + 1;
        sigma = math.sqrt(2 * n0 * n1 * (2 * n0 * n1 - n)/ n ** 2 / (n - 1));
        p = norm.cdf((r - mu) / sigma, 0, 1);

    return p;
