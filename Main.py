#!/usr/bin/python3
import math;
import os;
import time;
import threading;
import json;
import sys;
import random;
import multiprocessing;
import pickle;

from scipy.stats import t, f;

from DecisionTree import *;
from Sampling import *;
from PCA import *;
from IsolationForest import *;
from scipy import stats;
import numpy as np;
import numpy.matlib as npm;
import numpy.linalg as npl;
import matplotlib.pyplot as plt;
import pandas as pd;

import DataHelper;
import IsolationForestTest;
import CutForestTest;
import KMeansTest;
import KMeans;
import KMediansTest;
import KMedians;
import LinearRegressionTest;
import UnaryLinearRegressionTest;
import MultipleLinearRegressionTest;
import Optimizer;
import LogisticRegressionTest;
import LDAClassifierTest;

import LinearRegression;
import LogisticRegression;
import LDAClassifier;
import Splines;


def acfTest(rho, n, k, alpha = 0.05):
    Q = n * (n + 2) * (np.power(rho[1:k + 1, 0], 2).T * (1 / np.mat(range(n - 1, n - k - 1, -1)).T))[0, 0];

    p = 1 - stats.chi2.cdf(Q, k);
    print("Χ^2 = {0}, df = {1}, p-value: {2}".format(Q, k, p));

    return p >= alpha;


def acf(x, k, n, sigma2):
    if k == 0:
        return 1;

    x1 = x[:n - k, :];
    x2 = x[k:, :];

    return (x1.T * x2)[0, 0] / sigma2;


def pacf(rho, k):
    if k == 0:
        return 1;
    if k == 1:
        return rho[1, 0];

    D = np.mat(np.zeros((k, k)));

    for i in range(0, k):
        D.A[i:, i] = rho.A[:k-i, 0];
    D += D.T - np.diagflat(D.diagonal());

    Dk = D.copy();
    Dk.A[:, k - 1] = rho.A[1:k + 1, 0];

    return np.linalg.det(Dk) / np.linalg.det(D);


def testResiduals(y, yHat):
    residuals = y - yHat;
    centralizedX = residuals - residuals.mean();
    n, sigma2, lag = residuals.shape[0], (centralizedX.T * centralizedX)[0, 0], 20;

    sacf = [acf(centralizedX, k, n, sigma2) for k in range(0, lag + 1)];
    rho = np.mat(sacf).T;

    for i in range(1, math.floor(lag / 3) + 1):
        acfTest(rho, n, 3 * i);
    print("");


def getNormal(r1, r2):
    return math.sqrt(-2 * math.log(1 - r1.random())) * math.cos(2 * math.pi * r2.random());


def table(actualValue, predictValue):
    tp = predictValue[(actualValue == 1).A.flatten(), :].sum();
    fp = predictValue[(actualValue == 0).A.flatten(), :].sum();
    tn = -(predictValue - 1)[(actualValue == 0).A.flatten(), :].sum();
    fn = -(predictValue - 1)[(actualValue == 1).A.flatten(), :].sum();

    accuracy = (tp + tn) / (tp + fp + tn + fn);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * tp / (2 * tp + fp + fn);

    print("accuracy: {0}, precision: {1}, recall: {2}, f1: {3}".format(accuracy, precision, recall, f1));


def plotAnomaly(position, points, j, daySize, data, marks):
    if len(points) == 0:
        return;

    plt.title("{0} - {1}".format(position, len(points)));
    plt.plot(list(range(0, daySize)), data[position * daySize: (position + 1) * daySize, j].A.flatten(), "-x");
    if np.all(marks[points, j]):
        plt.scatter([p - position * daySize for p in points], [data[p, j] for p in points], marker = "o", linewidths = 4, color = "red");
    elif not np.any(marks[points, j]):
        plt.scatter([p - position * daySize for p in points], [data[p, j] for p in points], marker = "o", linewidths = 4, color = "black");
    else:
        for p in points:
            plt.scatter(p - position * daySize, data[p, j], marker = "o", linewidths = 4, color = ("red" if marks[p, j] == 1 else "black"));


def showAnomaly(indices, j, size, data, marks):
    daySize = size * 24;
    position, current, points = 0, 0, [];
    for i in indices:
        position = math.floor(i / daySize);

        if len(points) > 0:
            if current == position:
                points.append(i);
            else:
                plt.figure(1, (12, 8));
                plt.get_current_fig_manager().window.showMaximized();
                plotAnomaly(current, points, j, daySize, data, marks);
                plt.show(block = True);
                plt.close();

                points.clear();
                current = position;
                points.append(i);
        else:
            current = position;
            points.append(i);


def showAnomaly2(indices1, indices2, j, size, data, marks):
    daySize = size * 24;

    for i in range(0, math.floor(data.shape[0] / daySize)):
        points1 = [k for k in indices1 if math.floor(k / daySize) == i];
        points2 = [k for k in indices2 if math.floor(k / daySize) == i];
        if len(points1) == 0 and len(points2) == 0:
            continue;

        plt.figure(1, (12, 8));
        plt.get_current_fig_manager().window.showMaximized();
        plt.subplot(211);
        plotAnomaly(i, points1, j, daySize, data, marks);
        plt.subplot(212);
        plotAnomaly(i, points2, j, daySize, data, marks);
        plt.show(block = True);
        plt.close();



def showDiff(y1, y2, size):
    x = list(range(0, size));

    for i in range(math.floor(len(y1) / size)):
        plt.figure(1, (12, 8));
        plt.get_current_fig_manager().window.showMaximized();
        plt.title(str(i + 1));
        plt.axhline(0);
        plt.scatter(x, y1[i * size: (i + 1) * size], color = "black");
        plt.scatter(x, y2[i * size: (i + 1) * size], color = "red");
        plt.show(block = True);
        plt.close();



def calcAmplitude(i, j, offset, size, h, M, data):
    X = np.mat(np.arange(size * h)).T;
    y = data[i + 1 + offset - size * h:i + 1 + offset, j];
    return data[i, j] - LinearRegression.LinearRegression().fit(X, y, [LinearRegression.RegressionSplineFunction(h + M - 2, M)]).predictValue(np.mat([size * h - 1 - offset]))[0, 0];


def detectAmplitude(j, f):
    print("amplitude {0} started".format(j));

    M = 3;
    h = 24;
    size, sY = int(3600 / f),  [];
    data = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/data.npy".format(f)));
    marks = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/marks.npy".format(f)));
    y1 = data[:size * (h + 1), j];
    X1 = np.mat(np.arange(y1.shape[0])).T;
    m1 = LinearRegression.LinearRegression().fit(X1, y1, [LinearRegression.RegressionSplineFunction(h + M - 1, M)]);
    sY1 = m1.predictValue(X1);
    X1 = X1[:-size, :];
    y1 = y1[:-size, :];
    sY1 = sY1[:-size, :];
    sY.extend(sY1.A.flatten().tolist());

    for i in range(1, math.floor((data.shape[0] - size) / (size * 24))):
        y2 = data[i * size * h - size:(i + 1) * size * h + size, j];
        X2 = np.mat(np.arange(y2.shape[0])).T;
        m2 = LinearRegression.LinearRegression().fit(X2, y2, [LinearRegression.RegressionSplineFunction(h + M, 3)]);
        sY2 = m2.predictValue(X2);
        X2 = X2[size:-size, :];
        y2 = y2[size:-size, :];
        sY2 = sY2[size:-size, :];
        sY.extend(sY2.A.flatten().tolist());

        # plt.figure(1, (12, 8));
        # plt.get_current_fig_manager().window.showMaximized();
        # plt.subplot(211);
        # plt.title(str(i - 1));
        # plt.plot(X1.A.flatten(), y1.A.flatten(), "-x");
        # plt.plot(X1.A.flatten(), sY1.A.flatten(), color = "red");
        # plt.subplot(212);
        # plt.title(str(i));
        # plt.plot(X2.A.flatten(), y2.A.flatten(), "-x");
        # plt.plot(X2.A.flatten(), sY2.A.flatten(), color = "red");
        # plt.show(block = True);
        # plt.close();

        X1, y1, sY1 = X2, y2, sY2;

    amplitude = data[:len(sY), j] - np.mat(sY).T;
    amplitudeMean, amplitudeStd = amplitude.mean(), amplitude.std();
    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.showMaximized();
    # plt.hist(amplitude.A.flatten(), bins = 1000);
    # plt.show(block = True);
    # plt.close();

    indices1 = np.argwhere(amplitude < (amplitudeMean - 3 * amplitudeStd))[:, 0].flatten().tolist() + np.argwhere(amplitude > (amplitudeMean + 3 * amplitudeStd))[:, 0].flatten().tolist();
    indices1.sort();
    # showAnomaly(indices1, j, size, data, marks);

    h = 24;
    startIndex, offset, values = size * 24, int(12 * 60 / f), None;
    if not os.path.isfile("{0}/amplitude_{1}_values.npy".format(f, j)):
        with multiprocessing.Pool() as pool:
            values = pool.starmap(calcAmplitude, [(i, j, offset, size, h, M, data) for i in range(startIndex, data.shape[0] - offset)]);
        np.save("{0}/amplitude_{1}_values.npy".format(f, j), np.mat(values).T);

    values = np.mat(np.load("{0}/amplitude_{1}_values.npy".format(f, j)));
    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.showMaximized();
    # plt.hist(values.A.flatten(), bins = 1000);
    # plt.show(block = True);
    # plt.close();

    indices2 = (np.argwhere(values < (amplitudeMean - 3 * amplitudeStd))[:, 0].flatten() + startIndex).tolist() + (np.argwhere(values > (amplitudeMean + 3 * amplitudeStd))[:, 0].flatten() + startIndex).tolist();
    indices2.sort();
    # showAnomaly(indices2, j, size, data, marks);

    forest = None;
    if not os.path.isfile("{0}/amplitude_{1}_forest.npy".format(f, j)):
        forest = IsolationForest(200, 2 ** 9, CurvesThresholdFinder(0.65, 0.68, 0.73, False));
        forest.fill(amplitude);
        print("forest fill completed");
        forest.train(amplitude);
        print("forest train completed");

        with open("{0}/amplitude_{1}_forest.npy".format(f, j), "wb") as file:
            pickle.dump(forest, file, protocol = pickle.DEFAULT_PROTOCOL);
    else:
        with open("{0}/amplitude_{1}_forest.npy".format(f, j), "rb") as file:
            forest = pickle.load(file);

    scores = None;
    if not os.path.isfile("{0}/amplitude_{1}_scores.npy".format(f, j)):
        with multiprocessing.Pool() as pool:
            scores = pool.map(forest.getAnomalyScore, [np.mat([v]) for v in values.A.flatten().tolist()]);
        np.save("{0}/amplitude_{1}_scores.npy".format(f, j), np.mat(scores).T);

    scores = np.mat(np.load("{0}/amplitude_{1}_scores.npy".format(f, j)));
    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.showMaximized();
    # plt.hist(scores.A.flatten(), bins = 1000);
    # plt.show(block = True);
    # plt.close();

    indices3 = (np.argwhere(scores >= forest.threshold)[:, 0].flatten() + startIndex).tolist();
    indices3.sort();
    # showAnomaly(indices3, j, size, data, marks);

    indices4 = (np.argwhere(values < (amplitudeMean - 3 * amplitudeStd))[:, 0].flatten()).tolist() + (np.argwhere(values > (amplitudeMean + 3 * amplitudeStd))[:, 0].flatten()).tolist();
    indices4 = [i + startIndex for i in indices4 if values[i, 0] < amplitudeMean - 6 * amplitudeStd or values[i, 0] > amplitudeMean + 6 * amplitudeStd or scores[i] >= forest.threshold];
    indices4.sort();
    # showAnomaly(indices4, j, size, data, marks);

    print("amplitude {0} completed".format(j));


def getSpeedM2Internal(beta, X):
    return beta[1, 0] + X * beta[2:, 0];


def getSpeedM2(beta, knots, x):
    return getSpeedM2Internal(beta, np.hstack(tuple([(x > k) - 0 for k in knots])));


def calcSpeedM2(i, j, offset, size, h, data, X, T):
    y = data[i + 1 + offset - size * h:i + 1 + offset, j];
    m = LinearRegression.LinearRegression().fit(X, y);
    return getSpeedM2Internal(m.beta, T)[0, 0];


def getSpeedM3Internal(beta, x, X):
    return beta[1, 0] + 2 * beta[2, 0] * x + 2 * X * beta[3:, 0];


def getSpeedM3(beta, knots, x):
    return getSpeedM3Internal(beta, x, np.multiply(np.hstack(tuple([x - k for k in knots])), np.hstack(tuple([(x > k) - 0 for k in knots]))));


def calcSpeedM3(i, j, offset, size, h, data, X, x, T):
    y = data[i + 1 + offset - size * h:i + 1 + offset, j];
    m = LinearRegression.LinearRegression().fit(X, y);
    return getSpeedM3Internal(m.beta, x, T)[0, 0];


def calcDelta(i, j, data):
    return data[i, j] - data[i - 1, j];


def detectSpeed(j, f):
    print("speed {0} started".format(j));

    M = 3;
    h = 12;
    timespan = 12;
    size, speed = int(3600 / f), [];
    data = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/data.npy".format(f)));
    marks = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/marks.npy".format(f)));
    y1 = data[:size * (h + 1), j];
    X1 = np.mat(np.arange(y1.shape[0])).T;
    f1 = LinearRegression.RegressionSplineFunction(int((h + 1) * 60 / timespan) + M - 2, M);
    m1 = LinearRegression.LinearRegression().fit(X1, y1, [f1]);
    sY1 = m1.predictValue(X1);
    X1 = X1[:-size, :];
    y1 = y1[:-size, :];
    sY1 = sY1[:-size, :];
    speed.extend(getSpeedM3(m1.beta, f1.knots, X1).A.flatten().tolist());

    for i in range(1, math.floor((data.shape[0] - size) / (size * h))):
        y2 = data[i * size * h - size:(i + 1) * size * h + size, j];
        X2 = np.mat(np.arange(y2.shape[0])).T;
        f2 = LinearRegression.RegressionSplineFunction(int((h + 2) * 60 / timespan) + M - 2, M);
        m2 = LinearRegression.LinearRegression().fit(X2, y2, [f2]);
        sY2 = m2.predictValue(X2);
        X2 = X2[size:-size, :];
        y2 = y2[size:-size, :];
        sY2 = sY2[size:-size, :];
        speed.extend(getSpeedM3(m2.beta, f2.knots, X2).A.flatten().tolist());

        # plt.figure(1, (12, 8));
        # plt.get_current_fig_manager().window.showMaximized();
        # plt.subplot(211);
        # plt.title(str(i - 1));
        # plt.plot(X1.A.flatten(), y1.A.flatten(), "-x");
        # plt.plot(X1.A.flatten(), sY1.A.flatten(), color = "red");
        # plt.subplot(212);
        # plt.title(str(i));
        # plt.plot(X2.A.flatten(), y2.A.flatten(), "-x");
        # plt.plot(X2.A.flatten(), sY2.A.flatten(), color = "red");
        # plt.show(block = True);
        # plt.close();

        X1, y1, sY1 = X2, y2, sY2;
    print("speed history completed.");

    speed = np.mat(speed).T;
    speedMean, speedStd = speed.mean(), speed.std();
    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.showMaximized();
    plt.hist(speed.A.flatten(), bins = 1000);
    plt.show(block = True);
    plt.close();

    indices1 = np.argwhere(speed < (speedMean - 6 * speedStd))[:, 0].flatten().tolist() + np.argwhere(speed > (speedMean + 6 * speedStd))[:, 0].flatten().tolist();
    indices1.sort();
    # showAnomaly(indices1, j, size, data, marks);

    h = 1;
    startIndex, offset, values = size * h, int(12 * 60 / f), None;
    if not os.path.isfile("{0}/speed_{1}_values.npy".format(f, j)):
        ftn = LinearRegression.RegressionSplineFunction(int(h * 60 / timespan) + M - 2, M);
        X = ftn.getX(np.mat(np.arange(size * h)).T);
        x = np.mat([size * h - 1 - offset]);

        with multiprocessing.Pool() as pool:
            if M == 3:
                T = np.multiply(np.hstack(tuple([x - k for k in ftn.knots])), np.hstack(tuple([(x > k) - 0 for k in ftn.knots])));

                # values = [calcSpeedM3(i, j, offset, size, h, data, X, x, T) for i in range(startIndex, size * 24 * 10)];
                # showDiff(speed[startIndex: startIndex + len(values)].A.flatten().tolist(), values, size * 6);

                values = pool.starmap(calcSpeedM3, [(i, j, offset, size, h, data, X, x, T) for i in range(startIndex, data.shape[0] - offset)]);
            else:
                T = np.hstack(tuple([(x > k) - 0 for k in ftn.knots]));

                # values = [calcSpeedM2(i, j, offset, size, h, data, X, T) for i in range(startIndex, size * 24 * 10)];
                # showDiff(speed[startIndex: startIndex + len(values)].A.flatten().tolist(), values, size * 6);

                values = pool.starmap(calcSpeedM2, [(i, j, offset, size, h, data, X, T) for i in range(startIndex, data.shape[0] - offset)]);
        np.save("{0}/speed_{1}_values.npy".format(f, j), np.mat(values).T);
        print("realtime speed completed.");

    values = np.mat(np.load("{0}/speed_{1}_values.npy".format(f, j)));
    valuesMean, valuesStd = values.mean(), values.std();
    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.showMaximized();
    plt.hist(values.A.flatten(), bins = 1000);
    plt.show(block = True);
    plt.close();

    indices2 = (np.argwhere(values < (speedMean - 6 * speedStd))[:, 0].flatten() + startIndex).tolist() + (np.argwhere(values > (speedMean + 6 * speedStd))[:, 0].flatten() + startIndex).tolist();
    indices2.sort();
    # showAnomaly(indices2, j, size, data, marks);

    forest = None;
    if not os.path.isfile("{0}/speed_{1}_forest.npy".format(f, j)):
        forest = IsolationForest(200, 2 ** 9, CurvesThresholdFinder(0.65, 0.68, 0.73, False));
        forest.fill(speed);
        print("forest fill completed");
        forest.train(speed);
        print("forest train completed");

        with open("{0}/speed_{1}_forest.npy".format(f, j), "wb") as file:
            pickle.dump(forest, file, protocol = pickle.DEFAULT_PROTOCOL);
    else:
        with open("{0}/speed_{1}_forest.npy".format(f, j), "rb") as file:
            forest = pickle.load(file);

    scores = None;
    if not os.path.isfile("{0}/speed_{1}_scores.npy".format(f, j)):
        with multiprocessing.Pool() as pool:
            scores = pool.map(forest.getAnomalyScore, [np.mat([v]) for v in values.A.flatten().tolist()]);
        np.save("{0}/speed_{1}_scores.npy".format(f, j), np.mat(scores).T);
        print("realtime score completed.");

    scores = np.mat(np.load("{0}/speed_{1}_scores.npy".format(f, j)));
    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.showMaximized();
    # plt.hist(scores.A.flatten(), bins = 1000);
    # plt.show(block = True);
    # plt.close();

    indices3 = (np.argwhere(scores >= forest.threshold)[:, 0].flatten() + startIndex).tolist();
    indices3.sort();
    # showAnomaly(indices3, j, size, data, marks);

    indices4 = (np.argwhere(values < (speedMean - 3 * speedStd))[:, 0].flatten()).tolist() + (np.argwhere(values > (speedMean + 3 * speedStd))[:, 0].flatten()).tolist();
    indices4 = [i + startIndex for i in indices4 if values[i, 0] < speedMean - 6 * speedStd or values[i, 0] > speedMean + 6 * speedStd or scores[i] >= forest.threshold];
    indices4.sort();
    # showAnomaly(indices4, j, size, data, marks);

    deltaValues = np.diff(data[:, j], 1, 0);
    deltaMean, deltaStd = deltaValues.mean(), deltaValues.std();
    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.showMaximized();
    # plt.hist(deltaValues.A.flatten(), bins = 1000);
    # plt.show(block = True);
    # plt.close();

    # deltaScores = None;
    # if not os.path.isfile("{0}/speed_{1}_delta_scores.npy".format(f, j)):
    #     with multiprocessing.Pool() as pool:
    #         deltaScores = pool.map(forest.getAnomalyScore, [np.mat([v]) for v in deltaValues.A.flatten().tolist()]);
    #     np.save("{0}/speed_{1}_delta_scores.npy".format(f, j), np.mat(deltaScores).T);
    #
    # deltaScores = np.mat(np.load("{0}/speed_{1}_delta_scores.npy".format(f, j)));

    indices5 = [i + 1 for i in range(0, deltaValues.shape[0]) if deltaValues[i, 0] < deltaMean - 6 * deltaStd or deltaValues[i, 0] > deltaMean + 6 * deltaStd];
    indices5.sort();
    showAnomaly2(indices4, indices5, j, size, data, marks);

    print("speed {0} completed".format(j));



if __name__ == '__main__':
    # Hitters
    # data = np.mat(np.loadtxt("/home/xphter/Downloads/统计学习导论/Data/Hitters.csv", delimiter = ",", dtype = np.object))[1:, 1:];
    # r = (data[:, 13] == '"N"').A.flatten();
    # data[r, 13] = 1;
    # data[~r, 13] = 0;
    # r = (data[:, 14] == '"W"').A.flatten();
    # data[r, 14] = 1;
    # data[~r, 14] = 0;
    # r = (data[:, -1] == '"N"').A.flatten();
    # data[r, -1] = 1;
    # data[~r, -1] = 0;
    # r = (data[:, -2] != 'NA').A.flatten();
    # y = np.mat(data[r, -2], np.float);
    # X = np.mat(np.delete(data[r, :], -2, 1), np.float);

    # Credit
    # data = np.mat(np.loadtxt("/home/xphter/Downloads/统计学习导论/Data/Credit.csv", delimiter = ",", dtype = np.object))[1:, 1:];
    # r = (data[:, 6] == 'Male').A.flatten();
    # data[r, 6] = 1;
    # data[~r, 6] = 0;
    # r = (data[:, 7] == 'Yes').A.flatten();
    # data[r, 7] = 1;
    # data[~r, 7] = 0;
    # r = (data[:, 8] == 'Yes').A.flatten();
    # data[r, 8] = 1;
    # data[~r, 8] = 0;
    # data = np.insert(data, 10, (data[:, 9] == 'Asian').A.flatten() - 0, 1);
    # r = (data[:, 9] == 'Caucasian').A.flatten();
    # data[r, 9] = 1;
    # data[~r, 9] = 0;
    # y = np.mat(data[:, -1], np.float);
    # X = np.mat(data[:, :-1], np.float);

    # Wage
    # data = np.mat(np.loadtxt("/home/xphter/Downloads/统计学习导论/Data/Wage.csv", delimiter = ",", dtype = np.object))[1:, 1:];
    # y = np.mat(data[:, -1], np.float);
    # X = np.mat(data[:, 1], np.float);

    # detectAmplitude(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 60);
    detectSpeed(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 60);

    # data = np.mat(np.loadtxt("/home/xphter/Downloads/统计学习导论/Data/Wage.csv", delimiter = ",", dtype = np.object))[1:, 1:];
    # x = np.mat(data[:, 1], dtype = np.float);
    # y = np.mat(data[:, -1], dtype = np.float);
    #
    # df = 15;
    # ages = np.mat(np.arange(x.min(), x.max() + 0.1, 0.1)).T;
    #
    # # knots = [25, 40, 60];
    # spline1 = Splines.UnaryRegressionSpline(df);
    # spline1.fit(x, y);
    # print(spline1);
    # print();
    # wages1 = spline1.predictValue(ages);
    #
    # knots = list(set(x.A.flatten().tolist()));
    # spline2 = Splines.NatureCubicSpline(df);
    # spline2.fit(x, y);
    # print(spline2);
    # print();
    # wages2 = spline2.predictValue(ages);
    #
    # spline3 = Splines.UnaryPolynomialRegression(df);
    # spline3.fit(x, y);
    # print(spline3);
    # print();
    # wages3 = spline3.predictValue(ages);
    #
    # plt.figure(1, (12, 8));
    #
    # ax1 = plt.subplot(111);
    # ax1.scatter(x.A.flatten(), y.A.flatten());
    # ax1.plot(ages.A.flatten(), wages1.A.flatten(), color = "yellow");
    # # for i in spline1.knots:
    # #     ax1.axvline(i, color = "yellow");
    # ax1.plot(ages.A.flatten(), wages2.A.flatten(), color = "red");
    # # for i in spline2.knots:
    # #     ax1.axvline(i, color = "red");
    # ax1.plot(ages.A.flatten(), wages3.A.flatten(), color = "black");
    # # for i in spline2.knots:
    # #     ax1.axvline(i, color = "red");
    #
    # plt.show(block = True);
    # plt.close();


    # LDAClassifierTest.testLDAClassifier();
    # LogisticRegressionTest.testLogisticRegression();
    # LinearRegressionTest.testLinearRegression();
    # UnaryLinearRegressionTest.testUnaryLinearRegression();
    # MultipleLinearRegressionTest.testMultipleLinearRegression();
    # MultipleLinearRegressionTest.testOptimalSubset();
    # MultipleLinearRegressionTest.testForwardSelection();
    # MultipleLinearRegressionTest.testBackwardSelection();
    # IsolationForestTest.testIsolationForest();

    # data = np.mat(np.loadtxt("/media/WindowsE/Data/PARS/JNLH/Models/653483c8783c4b3ea04060ac02054912/isolation_scores.txt")).T * 100;
    # data = np.mat(np.loadtxt("scores.txt")).T * 100;
    # indices, distances, center = KMeans.KMeans(lambda X, k: np.mat([X.min(), X.mean(), X.max()]).T).clustering(data, 3,1);
    # print(center);
    #
    # threshold = center[2, 0];
    # if threshold > 70:
    #     threshold = max(65, data[(indices == 2).A.flatten(), :].min(0)[0, 0]);
    # print(threshold);
    #
    # i = None;
    # finder = ThresholdFinder();
    #
    # for j in range(0, data.shape[0]):
    #     if data[j, 0] >= threshold and i is None:
    #         i = j;
    #
    #     if data[j, 0] < threshold and i is not None:
    #         if j - i > 20:
    #             x, y = finder.fit(data[i:j, 0]);
    #
    #             plt.figure(1, (16, 10));
    #             plt.plot(list(range(0, j - i)), data[i:j, 0].A.flatten().tolist(), marker = "x");
    #             if x is not None and y is not None:
    #                 plt.plot(x, y, color = "r");
    #             plt.show();
    #
    #         i = None;
    #
    # print(finder.threshold);


# plt.figure(1, (12, 8));
# plt.scatter(X.A.flatten(), Y.A.flatten());
# plt.plot([X.min(), X.max()], [beta0 + beta1 * X.min(), beta0 + beta1 * X.max()]);
# plt.show();




# lr2 = LogisticRegression.LogisticRegression();
# lr2.train(dataSet, 0.000001, 0.0000001, None, 0);


# data = np.mat(np.load(r"D:\BrightWay\PARS\Code\Backend\Brightway.PARS.IntelligentModel.AppHost\bin\Debug\Models\1d389eb979a647e3ad22f95448bfad74\dataSet.npy"));
# np.savetxt(r"D:\BrightWay\PARS\Code\Backend\Brightway.PARS.IntelligentModel.AppHost\bin\Debug\Models\1d389eb979a647e3ad22f95448bfad74\dataSet.txt", data, delimiter = ",");


# data1 = np.mat(np.loadtxt(r"E:\Data\PARS\JNLH\Models\c9735c8842ad4ad0a2740b20c5c4e92f\isolation_scores.txt"));
# data2 = np.mat(np.loadtxt(r"E:\Data\PARS\JNLH\Models\c9735c8842ad4ad0a2740b20c5c4e92f\cut_scores.txt"));


# def getPageNumber(ordinal):
#     pageNumber, totalCount = 0, 0;
#     numbers = np.mat(np.loadtxt(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\testSet\numbers.txt", delimiter = ",")).T;
#
#     for i in range(0, numbers.shape[0]):
#         if (i == numbers.shape[0] - 1) or (ordinal > totalCount) and (ordinal <= totalCount + numbers[i + 1, 0]):
#             pageNumber = i + 1;
#             break;
#
#         totalCount += numbers[i, 0];
#
#     return pageNumber, ordinal - totalCount;


# print(getPageNumber(263541));
# print(getPageNumber(283836));
# print(getPageNumber(284888));

# IsolationForestTest.testIsolationForest();
# CutForestTest.testCutForest();

# X = np.mat(np.loadtxt(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\2019\all.txt", delimiter = ","));
# Y = X[:, 1:];
# np.savetxt(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\2019\all_75.txt", Y, delimiter = ",");
# np.save(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\2019\all_75.npy", Y);

# matrix, dataSet = None, None;
# folderPath = r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\2019";
#
# for fileName in os.listdir(folderPath):
#     dataSet = np.mat(np.loadtxt(folderPath + "\\" + fileName, delimiter = ","));
#
#     if matrix is None:
#         matrix = dataSet;
#     else:
#         matrix = np.vstack((matrix, dataSet));
#
# np.savetxt(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\2019\all.txt", matrix, delimiter = ",");



# X = np.mat(np.loadtxt(r"data\datingTestSet.txt", delimiter = "\t"));
# Y = X[:, -1];
# X = X[:, :-1:];
# X, mu, sigma = DataHelper.normalizeFeatures(X);
# X = np.hstack((X, Y));
#
# ax = plt.figure().add_subplot(111);
# colors = np.empty(Y.shape[0], np.str);
# colors[(Y == 0).A.flatten()] = "r";
# colors[(Y == 1).A.flatten()] = "g";
# colors[(Y == 2).A.flatten()] = "b";
# ax.scatter(X[:, 0].A.flatten(), X[:, 1].A.flatten(), c = colors);
# plt.show();


# data = np.loadtxt(r"E:\8\machinelearninginaction\Ch08\ex0.txt", delimiter = "\t");
#
# X = npm.mat(data[:, [0, 1]]);
# Y = npm.mat(data[:, -1]).T;
# Theta = (X.T * X).I * X.T * Y;
#
# Y2 = X * Theta;
#
# f = plt.figure();
# ax = f.add_subplot(111);
# ax.scatter(X[:, 1].flatten().A[0], Y.flatten().A[0]);
# ax.plot(X[:, 1].flatten().A[0], Y2.flatten().A[0]);
# plt.show();


# with open("data/iris/iris.data", "r") as stream:
#     for line in stream.readlines():
#         if len(line.strip()) == 0:
#             continue;
#
#         matrix.append([float(item) for item in line.strip().replace("Iris-setosa", "1").replace("Iris-versicolor", "2").replace("Iris-virginica","3").split(",")]);

    #matrix = [[float(item) for item in line.strip().replace("Iris-setosa", "1").replace("Iris-versicolor", "2").replace("Iris-virginica", "3").split(",")] for line in stream.readlines() if len(line.strip(())) > 0];

#matrix = [row[1:] + row[:1] for row in matrix];

# trainSet, testSet = holdOutSampling(matrix, 0.7);
#
# forest = [];
# types = [1] * 4;
#
# for i in range(0, 20):
#     treeSet = bootstrapSampling(matrix);
#     features = chooseFeature(4, 4);
#     forest.append(trainCART(treeSet, types, features));

# for tree in forest:
#     showTree(tree);

#tree = trainCART(trainSet, types);
# showTree(tree);
# pruningTree(tree, 2);
#showTree(tree);

# errorCount = 0;
#
# for row in matrix:
#     if classifyForest(forest, row) != row[-1]:
#         errorCount += 1;
#
# print("{0}, {1}%".format(errorCount, errorCount * 100 / len(testSet)));







