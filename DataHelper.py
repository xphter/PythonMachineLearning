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


# only for normal distribution
def calcMahalanobisDistance(X, v, inverseSigma):
    Y = X - v;

    return np.sqrt(np.multiply(Y * inverseSigma, Y).sum(1));


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


