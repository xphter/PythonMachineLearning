import math;
import random;
import numpy as np;
import numpy.linalg as npl;
import numpy.matlib as npm;


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
def __getArrangement(n):
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


def calcEuclideanDistance(X, v):
    Y = X - v;

    return np.power(Y, 2).sum(1);


# only for normal distribution
def calcMahalanobisDistance(X, v, inverseSigma):
    Y = X - v;

    return np.multiply(Y * inverseSigma, Y).sum(1);


# only for binary dataset
def calcJaccardCoefficient(X, v):
    return (X & v).sum(1) / (X | v).sum(1);


# for sparse dataset
def calcCosine(X, v):
    return np.dot(X / np.mat(list(map(npl.norm, X))).T, v.T / npl.norm(v));


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
    result = [];
    classes = __groupByLabel(X);

    for subSet in classes.values():
        result.extend([random.choice(subSet) for i in range(0, len(subSet))]);

    return np.mat(result);


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


# Leave One Out Cross Validation
def oneOutSampling(X, y):
    result = [];
    n = X.shape[0];
    current = None;
    indices = list(range(0, n));

    for i in range(0, n):
        current = indices[0: i] + indices[i + 1:];

        result.append((X[current, :], y[current, :], X[i, :], y[i, :]));

    return result;


# k Fold Cross Validation
def foldOutSampling(X, y, k):
    n = X.shape[0];
    if n / k < 1:
        raise ValueError();

    indices = None;
    XFolds, yFolds = [], [];
    foldSize = math.ceil(n / k);
    arrangement = __getArrangement(n);
    for i in range(0, k):
        indices = arrangement[i * foldSize: (i + 1) * foldSize] if i < k - 1 else arrangement[i * foldSize:];
        XFolds.append(X[indices, :]);
        yFolds.append(y[indices, :]);

    result = [];

    for i in range(0, k):
        result.append((np.vstack(tuple(XFolds[0: i] + XFolds[i + 1:])),
                       np.vstack(tuple(yFolds[0: i] + yFolds[i + 1:])),
                       XFolds[i],
                       yFolds[i]));

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
