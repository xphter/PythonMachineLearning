#!/usr/bin/python3
import math;
import os;
import time;
import datetime;
import threading;
import json;
import sys;
import random;
import multiprocessing;
import psutil;
import pickle;
import PIL;
import scipy;


from scipy.stats import t, f;

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
import DecisionTree;
import PyTorchTest;

import DeviceConfig;
DeviceConfig.EnableGPU = True;

import NNTest;


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
                # plt.get_current_fig_manager().window.maximize();
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
        # plt.get_current_fig_manager().window.maximize();
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
        plt.get_current_fig_manager().window.maximize();
        plt.title(str(i + 1));
        plt.axhline(0);
        plt.scatter(x, y1[i * size: (i + 1) * size], color = "black");
        plt.scatter(x, y2[i * size: (i + 1) * size], color = "red");
        plt.show(block = True);
        plt.close();


def calcAmplitude(i, j, offset, size, h, M, data):
    X = np.mat(np.arange(size * h)).T;
    y = data[i + 1 + offset - size * h: i + 1 + offset, j];
    return data[i, j] - LinearRegression.LinearRegression().fit(X, y, [LinearRegression.RegressionSplineFunction(h + M - 2, M)]).predictValue(np.mat([size * h - 1 - offset]))[0, 0];


def detectAmplitude(j, f):
    print("amplitude {0} started".format(j));

    M = 3;
    h = 24;
    size, sY = int(3600 / f),  [];
    data = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/2020-08-01/data.npy".format(f)));
    marks = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/2020-08-01/marks.npy".format(f)));
    # y1 = data[:size * (h + 1), j];
    # X1 = np.mat(np.arange(y1.shape[0])).T;
    # m1 = LinearRegression.LinearRegression().fit(X1, y1, [LinearRegression.RegressionSplineFunction((h + 1) + M - 2, M)]);
    # sY1 = m1.predictValue(X1);
    # X1 = X1[:-size, :];
    # y1 = y1[:-size, :];
    # sY1 = sY1[:-size, :];
    # sY.extend(sY1.A.flatten().tolist());

    if not os.path.isfile(f"{f}/amplitude_{j}_amplitude.npy"):
        totalCount = math.floor((data.shape[0] - 0) / (size * h));

        for i in range(0, totalCount):
            y2 = data[i * size * h - 0:(i + 1) * size * h + 0, j];
            X2 = np.mat(np.arange(y2.shape[0])).T;
            m2 = LinearRegression.LinearRegression().fit(X2, y2, [LinearRegression.RegressionSplineFunction(h + M - 2, M)]);
            sY2 = m2.predictValue(X2);
            X2 = X2[:, :];
            y2 = y2[:, :];
            sY2 = sY2[:, :];
            sY.extend(sY2.A.flatten().tolist());

            # plt.figure(1, (12, 8));
            # # plt.get_current_fig_manager().window.showMaximized();
            # plt.subplot(111);
            # plt.title(f"{i}, {m2.r2}");
            # plt.plot(X2.A.flatten(), y2.A.flatten(), "-xk");
            # plt.plot(X2.A.flatten(), sY2.A.flatten(), "-or");
            # for x in f2.knots:
            #     plt.axvline(x, color = "b");
            # # plt.scatter(f1.knots, [y1.mean()] * len(f1.knots), marker="*", color = "b");
            # # plt.subplot(212);
            # # plt.title(str(i));
            # # plt.plot(X2.A.flatten(), y2.A.flatten(), "-x");
            # # plt.plot(X2.A.flatten(), sY2.A.flatten(), color = "red");
            # plt.show(block = True);
            # plt.savefig(f"/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/amplitude_images_history_YCH_FI6221.PV/{i}.png");
            # print(f"{i}/{totalCount} saved.");
            # plt.close();

            # X1, y1, sY1, f1 = X2, y2, sY2, f2;

        print("amplitude history completed.");
        amplitude = data[: len(sY), j].A.flatten() - np.array(sY);
        np.save(f"{f}/amplitude_{j}_amplitude.npy", amplitude);
    else:
        amplitude = np.load(f"{f}/amplitude_{j}_amplitude.npy");

    amplitudeMean, amplitudeStd = amplitude.mean(), amplitude.std();
    print(DataHelper.testNormalDistribution(amplitude));
    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.showMaximized();
    # plt.hist(amplitude, bins = 1000);
    # plt.show(block = True);
    # plt.close();

    indices1 = np.argwhere(amplitude < (amplitudeMean - 6 * amplitudeStd))[:, 0].flatten().tolist() + np.argwhere(amplitude > (amplitudeMean + 6 * amplitudeStd))[:, 0].flatten().tolist();
    indices1.sort();
    showAnomaly(indices1, j, size, data, marks);

    h, m = 24, 12; # 24 hours, 12 minutes
    startIndex, offset, values = size * h, int(m * 60 / f), None;
    if not os.path.isfile(f"{f}/amplitude_{j}_values.npy"):
        with multiprocessing.Pool(psutil.cpu_count(False) - 2) as pool:
            values = pool.starmap(calcAmplitude, [(i, j, offset, size, h, M, data) for i in range(startIndex, data.shape[0] - offset)]);
        np.save("{0}/amplitude_{1}_values.npy".format(f, j), np.array(values));
    else:
        values = np.load(f"{f}/amplitude_{j}_values.npy");

    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.showMaximized();
    # plt.hist(values, bins = 1000);
    # plt.show(block = True);
    # plt.close();

    indices2 = (np.argwhere(values < (amplitudeMean - 6 * amplitudeStd))[:, 0] + startIndex).tolist() + (np.argwhere(values > (amplitudeMean + 6 * amplitudeStd))[:, 0] + startIndex).tolist();
    indices2.sort();
    showAnomaly(indices2, j, size, data, marks);

    # forest = None;
    # if not os.path.isfile("{0}/amplitude_{1}_forest.npy".format(f, j)):
    #     forest = IsolationForest(200, 2 ** 9, CurvesThresholdFinder(0.65, 0.68, 0.73, False));
    #     forest.fill(amplitude);
    #     print("forest fill completed");
    #     forest.train(amplitude);
    #     print("forest train completed");
    #
    #     with open("{0}/amplitude_{1}_forest.npy".format(f, j), "wb") as file:
    #         pickle.dump(forest, file, protocol = pickle.DEFAULT_PROTOCOL);
    # else:
    #     with open("{0}/amplitude_{1}_forest.npy".format(f, j), "rb") as file:
    #         forest = pickle.load(file);
    #
    # scores = None;
    # if not os.path.isfile("{0}/amplitude_{1}_scores.npy".format(f, j)):
    #     with multiprocessing.Pool(psutil.cpu_count(False) - 2) as pool:
    #         scores = pool.map(forest.getAnomalyScore, [np.mat([v]) for v in values.A.flatten().tolist()]);
    #     np.save("{0}/amplitude_{1}_scores.npy".format(f, j), np.mat(scores).T);
    #
    # scores = np.mat(np.load("{0}/amplitude_{1}_scores.npy".format(f, j)));
    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.maximize();
    # plt.hist(scores.A.flatten(), bins = 1000);
    # plt.show(block = True);
    # plt.close();

    # indices3 = (np.argwhere(scores >= forest.threshold)[:, 0].flatten() + startIndex).tolist();
    # indices3.sort();
    # showAnomaly(indices3, j, size, data, marks);

    # indices4 = (np.argwhere(values < (amplitudeMean - 3 * amplitudeStd))[:, 0].flatten()).tolist() + (np.argwhere(values > (amplitudeMean + 3 * amplitudeStd))[:, 0].flatten()).tolist();
    # indices4 = [i + startIndex for i in indices4 if values[i, 0] < amplitudeMean - 6 * amplitudeStd or values[i, 0] > amplitudeMean + 6 * amplitudeStd or scores[i] >= forest.threshold];
    # indices4.sort();
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


def findKnots1(y, gap, offset):
    l, m, M = len(y), y.argmin().item(), y.argmax().item();

    if m == M:
        return [(l - 1) / 2];

    if m > M:
        m, M = M, m;

    knots = [];
    if m >= gap:
        knots.append(offset + m);
    if l - 1 - M >= gap:
        knots.append(offset + M);

    if m > gap:
        if m < 3 * gap:
            knots.append(offset + (m - 1) / 2);
        else:
            knots.extend(findKnots1(y[:m], gap, offset));

    if M - 1 - m > gap:
        if M - 1 - m < 3 * gap:
            knots.append(offset + (m + M) / 2);
        else:
            knots.extend(findKnots1(y[m + 1:M], gap, offset + m + 1));

    if l - 1 - M > gap:
        if l - 1 - M < 3 * gap:
            knots.append(offset + (M + l) / 2);
        else:
            knots.extend(findKnots1(y[M + 1:], gap, offset + M + 1));

    knots.sort();

    return knots;


def findKnots2(y, gap = 2):
    knots = [];
    i, l = 1, len(y);
    maxIndex = l - 1;

    while i <= maxIndex - 1:
        value = y[i];

        if y[i - 1] < value and y[i + 1] < value or y[i - 1] > value and y[i + 1] > value:
            knots.append(i);
            i += gap;
        elif y[i + 1] == value:
            j = i + 1;

            while j + 1 <= maxIndex and y[j + 1] == value:
                j += 1;

            knots.append(i);
            if j - i >= gap and j < maxIndex:
                knots.append(j);
                i = j + gap;
            else:
                i += gap;
        else:
            i += 1;

    if len(knots) == 0:
        x = np.arange(l);
        C0 = np.vstack((x, y));
        alpha = -math.atan((y[l - 1] - y[0]) / (l - 1));
        if alpha > 0:
            print(alpha);
        A = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]]);
        C1 = A @ C0;
        y1 = C1[1];

        i = 1;
        while i <= maxIndex - 1:
            value = y1[i];

            if y1[i - 1] < value and y1[i + 1] < value or y1[i - 1] > value and y1[i + 1] > value:
                knots.append(i);
                i += gap;
            else:
                i += 1;

    return knots if len(knots) > 0 else [l / 2];


def findTurnPoints(y, gap = 2):
    knots = [];
    l = len(y);
    maxIndex = l - 1;

    if l == 1:
        return knots;

    if y[l - 1] != y[0]:
        C0 = np.vstack((np.arange(l), y));
        alpha = -math.atan((y[l - 1] - y[0]) / (l - 1.0));
        A = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]]);
        C1 = A @ C0;
        y1 = C1[1];
    else:
        y1 = y;

    i = 1;
    while i <= maxIndex - 1:
        value = y1[i];

        if y1[i - 1] < value and y1[i + 1] < value or y1[i - 1] > value and y1[i + 1] > value:
            knots.append(i);
            i += gap;
        else:
            i += 1;

    return knots;


def findKnots3(y, gap = 1):
    knots = [];
    l = len(y);
    maxIndex = l - 1;

    i = 1;
    while i <= maxIndex - 1:
        value = y[i];

        if y[i - 1] < value and y[i + 1] < value or y[i - 1] > value and y[i + 1] > value:
            knots.append(i);
            i += gap;
        elif y[i + 1] == value:
            j = i + 1;

            while j + 1 <= maxIndex and y[j + 1] == value:
                j += 1;

            knots.append(i);
            if j - i >= gap and j < maxIndex:
                knots.append(j);
                i = j + gap;
            else:
                i += gap;
        else:
            i += 1;

    offset = 0;
    for k in knots + [l]:
        nodes = findTurnPoints(y[offset: k + 1]);
        if len(nodes) > 0:
            knots.extend([offset + n for n in nodes]);

        offset = k;

    if len(knots) > 0:
        knots.sort();

        return knots;
    else:
        return [l / 2];


def detectSpeed(j, f):
    print("speed {0} started".format(j));

    M = 2;
    h = 1;
    timespan = 6;
    size, speed = int(3600 / f), [];
    # data = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/Realtime_30/__JNRTDB_YCH_LIC6205.PV.npy")).T;
    data = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/2020-08-01/data.npy".format(f)));
    marks = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/2020-08-01/marks.npy".format(f)));
    # y1 = data[:size * (h + 0), j];
    # X1 = np.mat(np.arange(y1.shape[0])).T;
    # knots = findKnots2(y1.A.flatten());
    # f1 = LinearRegression.RegressionSplineFunction(int((h + 0) * 60 / timespan) + M - 2, M, knots);
    # m1 = LinearRegression.LinearRegression().fit(X1, y1, [f1]);
    # sY1 = m1.predictValue(X1);
    # X1 = X1[:, :];
    # y1 = y1[:, :];
    # sY1 = sY1[:, :];
    # speed.extend(getSpeedM2(m1.beta, f1.knots, X1).A.flatten().tolist());

    if not os.path.isfile(f"{f}/speed_{j}_speed.npy"):
        totalCount = math.floor((data.shape[0] - 0) / (size * h));

        for i in range(0, totalCount):
            y2 = data[i * size * h - 0:(i + 1) * size * h + 0, j];
            X2 = np.mat(np.arange(y2.shape[0])).T;
            knots = findKnots3(y2.A.flatten());
            f2 = LinearRegression.RegressionSplineFunction(int((h + 0) * 60 / timespan) + M - 2, M, knots);
            m2 = LinearRegression.LinearRegression().fit(X2, y2, [f2]);
            sY2 = m2.predictValue(X2);
            X2 = X2[:, :];
            y2 = y2[:, :];
            sY2 = sY2[:, :];
            speed.extend(getSpeedM2(m2.beta, f2.knots, X2).A.flatten().tolist());

            # plt.figure(1, (12, 8));
            # # plt.get_current_fig_manager().window.showMaximized();
            # plt.subplot(111);
            # plt.title(f"{i}, {m2.r2}");
            # plt.plot(X2.A.flatten(), y2.A.flatten(), "-xk");
            # plt.plot(X2.A.flatten(), sY2.A.flatten(), "-or");
            # for x in f2.knots:
            #     plt.axvline(x, color = "b");
            # # plt.scatter(f1.knots, [y1.mean()] * len(f1.knots), marker="*", color = "b");
            # # plt.subplot(212);
            # # plt.title(str(i));
            # # plt.plot(X2.A.flatten(), y2.A.flatten(), "-x");
            # # plt.plot(X2.A.flatten(), sY2.A.flatten(), color = "red");
            # plt.show(block = True);
            # plt.savefig(f"/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/speed_images_history_YCH_LIC6206.PV/{i}.png");
            # print(f"{i}/{totalCount} saved.");
            # plt.close();

            # X1, y1, sY1, f1 = X2, y2, sY2, f2;

        print("speed history completed.");
        speed = np.array(speed);
        np.save(f"{f}/speed_{j}_speed.npy", speed);
    else:
        speed = np.load(f"{f}/speed_{j}_speed.npy");

    speedMean, speedStd = speed.mean(), speed.std();
    print(np.logical_or((speed - speedMean) / speedStd < -6, (speed - speedMean) / speedStd > 6).sum());

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.showMaximized();
    plt.hist(speed, bins = 1000);
    for x in [speedMean, speedMean - 6 * speedStd, speedMean + 6 * speedStd]:
        plt.axvline(x, color = "b");
    plt.show(block = True);
    plt.close();

    deltaValues = np.diff(data[:, j], 1, 0);
    deltaMean, deltaStd = deltaValues.mean(), deltaValues.std();
    print(np.logical_or((deltaValues - deltaMean) / deltaStd < -6, (deltaValues - deltaMean) / deltaStd > 6).sum());

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.showMaximized();
    plt.hist(deltaValues.A.flatten(), bins = 1000);
    for x in [deltaMean, deltaMean - 6 * deltaStd, deltaMean + 6 * deltaStd]:
        plt.axvline(x, color = "b");
    plt.show(block = True);
    plt.close();

    indices1 = np.argwhere(speed < (speedMean - 6 * speedStd))[:, 0].flatten().tolist() + np.argwhere(speed > (speedMean + 6 * speedStd))[:, 0].flatten().tolist();
    indices1.sort();
    # showAnomaly(indices1, j, size, data, marks);

    # h = 1;
    # startIndex, offset, values = size * h, int(12 * 60 / f), None;
    # if not os.path.isfile("{0}/speed_{1}_values.npy".format(f, j)):
    #     ftn = LinearRegression.RegressionSplineFunction(int(h * 60 / timespan) + M - 2, M);
    #     X = ftn.getX(np.mat(np.arange(size * h)).T);
    #     x = np.mat([size * h - 1 - offset]);
    #
    #     with multiprocessing.Pool(psutil.cpu_count(False) - 2) as pool:
    #         if M == 3:
    #             T = np.multiply(np.hstack(tuple([x - k for k in ftn.knots])), np.hstack(tuple([(x > k) - 0 for k in ftn.knots])));
    #
    #             # values = [calcSpeedM3(i, j, offset, size, h, data, X, x, T) for i in range(startIndex, size * 24 * 10)];
    #             # showDiff(speed[startIndex: startIndex + len(values)].A.flatten().tolist(), values, size * 6);
    #
    #             values = pool.starmap(calcSpeedM3, [(i, j, offset, size, h, data, X, x, T) for i in range(startIndex, data.shape[0] - offset)]);
    #         else:
    #             T = np.hstack(tuple([(x > k) - 0 for k in ftn.knots]));
    #
    #             # values = [calcSpeedM2(i, j, offset, size, h, data, X, T) for i in range(startIndex, size * 24 * 10)];
    #             # showDiff(speed[startIndex: startIndex + len(values)].A.flatten().tolist(), values, size * 6);
    #
    #             values = pool.starmap(calcSpeedM2, [(i, j, offset, size, h, data, X, T) for i in range(startIndex, data.shape[0] - offset)]);
    #     np.save("{0}/speed_{1}_values.npy".format(f, j), np.mat(values).T);
    #     print("realtime speed completed.");
    #
    # values = np.load(f"{f}/speed_{j}_values.npy");
    # valuesMean, valuesStd = values.mean(), values.std();
    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.showMaximized();
    # plt.hist(values, bins = 1000);
    # plt.show(block = True);
    # plt.close();

    # indices2 = (np.argwhere(values < (speedMean - 6 * speedStd))[:, 0].flatten() + startIndex).tolist() + (np.argwhere(values > (speedMean + 6 * speedStd))[:, 0].flatten() + startIndex).tolist();
    # indices2.sort();
    # showAnomaly(indices2, j, size, data, marks);

    forest = None;
    if not os.path.isfile("{0}/speed_{1}_forest.npy".format(f, j)):
        dataSet = np.mat(speed).T;
        forest = IsolationForest(200, 2 ** 9, CurvesThresholdFinder(0.65, 0.68, 0.73, False));
        forest.fill(dataSet);
        print("forest fill completed");
        forest.train(dataSet);
        print("forest train completed");

        with open("{0}/speed_{1}_forest.npy".format(f, j), "wb") as file:
            pickle.dump(forest, file, protocol = pickle.DEFAULT_PROTOCOL);
    else:
        with open("{0}/speed_{1}_forest.npy".format(f, j), "rb") as file:
            forest = pickle.load(file);

    # scores = None;
    # if not os.path.isfile("{0}/speed_{1}_scores.npy".format(f, j)):
    #     with multiprocessing.Pool(psutil.cpu_count(False) - 2) as pool:
    #         scores = pool.map(forest.getAnomalyScore, [np.mat([v]) for v in values.A.flatten().tolist()]);
    #     np.save("{0}/speed_{1}_scores.npy".format(f, j), np.mat(scores).T);
    #     print("realtime score completed.");
    #
    # scores = np.mat(np.load("{0}/speed_{1}_scores.npy".format(f, j)));
    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.showMaximized();
    # plt.hist(scores.A.flatten(), bins = 1000);
    # plt.show(block = True);
    # plt.close();

    scores = np.array(forest.scores);
    indices3 = np.argwhere(scores >= forest.threshold)[:, 0].flatten().tolist();
    indices3.sort();
    # showAnomaly(indices3, j, size, data, marks);

    # indices4 = (np.argwhere(values < (speedMean - 3 * speedStd))[:, 0].flatten()).tolist() + (np.argwhere(values > (speedMean + 3 * speedStd))[:, 0].flatten()).tolist();
    # indices4 = [i + startIndex for i in indices4 if values[i, 0] < speedMean - 6 * speedStd or values[i, 0] > speedMean + 6 * speedStd or scores[i] >= forest.threshold];
    # indices4.sort();
    # showAnomaly(indices4, j, size, data, marks);

    # deltaScores = None;
    # if not os.path.isfile("{0}/speed_{1}_delta_scores.npy".format(f, j)):
    #     with multiprocessing.Pool(psutil.cpu_count(False) - 2) as pool:
    #         deltaScores = pool.map(forest.getAnomalyScore, [np.mat([v]) for v in deltaValues.A.flatten().tolist()]);
    #     np.save("{0}/speed_{1}_delta_scores.npy".format(f, j), np.mat(deltaScores).T);
    #
    # deltaScores = np.mat(np.load("{0}/speed_{1}_delta_scores.npy".format(f, j)));

    # indices5 = [i + 1 for i in range(0, deltaValues.shape[0]) if deltaValues[i, 0] < deltaMean - 6 * deltaStd or deltaValues[i, 0] > deltaMean + 6 * deltaStd];
    indices5 = np.argwhere(deltaValues < (deltaMean - 6 * deltaStd))[:, 0].flatten().tolist() + np.argwhere(deltaValues > (deltaMean + 6 * deltaStd))[:, 0].flatten().tolist();
    indices5 = [i + 1 for i in indices5];
    indices5.sort();
    # showAnomaly(indices5, j, size, data, marks);
    # showAnomaly2(indices4, indices5, j, size, data, marks);

    print("speed {0} completed".format(j));


def detectSpeedFNN(j, f):
    print("speed {0} started".format(j));

    h = 1;
    iterationNumber = 1000;
    size, speed = int(3600 / f), [];
    data = np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/2020-08-01/data.npy".format(f));
    marks = np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/2020-08-01/marks.npy".format(f));

    for i in range(0, math.floor(data.shape[0] / (size * h))):
        optimizer = NNTest.Adam(0.01);
        network = NNTest.MultiLayerNet(1, [1000], 1, lastLayerType=NNTest.IdentityWithLossLayer, useBatchNormalization=True, useDropout=False, dropoutRatio=0.2);

        trainLossData = [];
        T = data[i * size * h:(i + 1) * size * h, j].reshape(-1, 1);
        X = np.arange(T.shape[0]).reshape(-1, 1);

        for k in range(iterationNumber):
            gradients, lossValue = network.gradient(X, T, True);

            optimizer.update(network.params, gradients);
            trainLossData.append(lossValue);
            print("loss value: {0}".format(lossValue));

        Y, dX = network.predictWithdX(X);

        # h = 1e-4;
        # Y1 = network.predict(X - h, False);
        # Y2 = network.predict(X + h, False);
        # dX2 = (Y2 - Y1) / (2 * h);
        # print(np.abs(dX - dX2).sum() / dX.shape[0]);

        Y, dX = Y.flatten(), dX.flatten();

        plt.figure(1, (12, 8));
        plt.get_current_fig_manager().window.maximize();
        plt.title(str(i));

        plt.plot(X.flatten(), T.flatten(), "-ok");
        plt.plot(X.flatten(), Y.flatten(), "-or");
        # for i in range(X.shape[0]):
        #     plt.plot([X[i, 0] - 0.3, X[i, 0] + 0.3], [Y[i] - 0.3 * dX[i], Y[i] + 0.3 * dX[i]], "-b");

        plt.show(block=True);
        plt.close();


    # for i in range(1, math.floor((data.shape[0] - size) / (size * h))):
    #     y2 = data[i * size * h - size:(i + 1) * size * h + size, j];
    #     X2 = np.mat(np.arange(y2.shape[0])).T;
    #     f2 = LinearRegression.RegressionSplineFunction(int((h + 2) * 60 / timespan) + M - 2, M);
    #     m2 = LinearRegression.LinearRegression().fit(X2, y2, [f2]);
    #     sY2 = m2.predictValue(X2);
    #     X2 = X2[size:-size, :];
    #     y2 = y2[size:-size, :];
    #     sY2 = sY2[size:-size, :];
    #     speed.extend(getSpeedM3(m2.beta, f2.knots, X2).A.flatten().tolist());
    #
    #     # plt.figure(1, (12, 8));
    #     # plt.get_current_fig_manager().window.maximize();
    #     # plt.subplot(211);
    #     # plt.title(str(i - 1));
    #     # plt.plot(X1.A.flatten(), y1.A.flatten(), "-x");
    #     # plt.plot(X1.A.flatten(), sY1.A.flatten(), color = "red");
    #     # plt.subplot(212);
    #     # plt.title(str(i));
    #     # plt.plot(X2.A.flatten(), y2.A.flatten(), "-x");
    #     # plt.plot(X2.A.flatten(), sY2.A.flatten(), color = "red");
    #     # plt.show(block = True);
    #     # plt.close();
    #
    #     X1, y1, sY1 = X2, y2, sY2;
    print("speed history completed.");

    print("speed {0} completed".format(j));


def detectChange(j, f):
    print("change {0} started".format(j));

    M = 2;
    h = 12;
    timespan = 6;
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
    if M == 3:
        speed.extend(getSpeedM3(m1.beta, f1.knots, X1).A.flatten().tolist());
    else:
        speed.extend(getSpeedM2(m1.beta, f1.knots, X1).A.flatten().tolist());

    for i in range(1, math.floor((data.shape[0] - size) / (size * h))):
        y2 = data[i * size * h - size:(i + 1) * size * h + size, j];
        X2 = np.mat(np.arange(y2.shape[0])).T;
        f2 = LinearRegression.RegressionSplineFunction(int((h + 2) * 60 / timespan) + M - 2, M);
        m2 = LinearRegression.LinearRegression().fit(X2, y2, [f2]);
        sY2 = m2.predictValue(X2);
        X2 = X2[size:-size, :];
        y2 = y2[size:-size, :];
        sY2 = sY2[size:-size, :];
        if M == 3:
            speed.extend(getSpeedM3(m2.beta, f2.knots, X2).A.flatten().tolist());
        else:
            speed.extend(getSpeedM2(m2.beta, f2.knots, X2).A.flatten().tolist());

        plt.figure(1, (12, 8));
        plt.get_current_fig_manager().window.maximize();
        plt.subplot(211);
        plt.title(str(i - 1));
        plt.plot(X1.A.flatten(), y1.A.flatten(), "-x");
        plt.plot(X1.A.flatten(), sY1.A.flatten(), color = "red");
        plt.subplot(212);
        plt.title(str(i));
        plt.plot(X2.A.flatten(), y2.A.flatten(), "-x");
        plt.plot(X2.A.flatten(), sY2.A.flatten(), color = "red");
        plt.show(block = True);
        plt.close();

        X1, y1, sY1 = X2, y2, sY2;
    print("change history completed.");

    speed = np.mat(speed).T;
    speedMean, speedStd = speed.mean(), speed.std();
    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();
    plt.hist(speed.A.flatten(), bins = 1000);
    plt.show(block = True);
    plt.close();


def isConstant(y, periods, alpha):
    if y.var() == 0:
        return True;

    p1 = [DataHelper.testWhiteNoise(y - y.mean(), m) for m in periods];
    if np.any(np.mat(p1) <= alpha):
        return False;

    p2 = LinearRegression.LinearRegression().fit(np.mat(range(0, y.shape[0])).T, y).betaP;
    if p2[1, 0] <= alpha:
        return False;

    p3 = DataHelper.testRunsLeft((y > np.quantile(y, 0.5)) - 0);
    if p3 <= alpha:
        return False;

    print("{0}, {1}, {2}".format(p1, p2.T, p3));
    return True;


def detectConstant(j, f):
    print("constant {0} started".format(j));

    timespan = 15;
    size = int(timespan * 60 / f);
    periods, alpha = [3, 6, 12], 0.1;
    data = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/data.npy".format(f)));
    marks = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/{0}/marks.npy".format(f)));
    y = data[:, j];
    n, i, offset = y.shape[0], 0, int(1 * 60 / f);

    while i + size <= n:
        k = 0;

        while i + size + k <= n and isConstant(y[i: i + size + k, :], periods, alpha):
            k += 1;

        if k > 0:
            length = size + k - 1;
            start, end = max(i - offset, 0), min(i + length + offset, n);

            plt.figure(1, (12, 8));
            plt.get_current_fig_manager().window.maximize();
            plt.title("{0}({1} - {2}, {3})".format(length, i, i + length - 1, j));
            plt.plot(list(range(start, end)), y[start: end, :].A.flatten(), "-x");
            plt.scatter(list(range(i, i + length)), y[i: i + length, :].A.flatten(), color="red");
            plt.show(block=True);
            plt.close();

            i += length;
        else:
            i += 1;

    print("constant {0} completed".format(j));


def testSpeed():
    startIndex, endIndex = 93, 118;
    data = [];
    y = np.mat(data).T;

    h, timespan, M = 1, 6, 2;
    X = np.mat(np.arange(y.shape[0])).T;
    f = LinearRegression.RegressionSplineFunction(int(h * 60 / timespan) + M - 2, M);
    m = LinearRegression.LinearRegression().fit(X, y, [f]);
    yHeat = m.predictValue(X);

    # speed1 = getSpeedM3(m.beta, f.knots, X[startIndex: endIndex, :]);
    # print(speed1.A.flatten().tolist());

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();
    plt.subplot(211);
    plt.plot(X.A.flatten(), y.A.flatten(), "-xb");
    plt.plot(X.A.flatten(), yHeat.A.flatten(), "-r");
    plt.plot(X[startIndex: endIndex, :].A.flatten(), y[startIndex: endIndex, :].A.flatten(), "or");
    plt.subplot(212);
    plt.plot(X[startIndex: endIndex, :].A.flatten(), y[startIndex: endIndex, :].A.flatten(), "-xb");
    plt.plot(X[startIndex: endIndex, :].A.flatten(), yHeat[startIndex: endIndex, :].A.flatten(), "-r");
    plt.show(block=True);
    plt.close();


def testAmplitude():
    startIndex, endIndex = 2855, 2880;
    data = [];
    y = np.mat(data).T;

    h, M = 24, 3;
    X = np.mat(np.arange(y.shape[0])).T;
    # m = LinearRegression.LinearRegression().fit(X, y, [LinearRegression.RegressionSplineFunction(h + M - 2, M)]);
    m = LinearRegression.LinearRegression().fit(X, y, [LinearRegression.RegressionSplineFunction(int(h * 60 / 60) + M - 2, M)]);
    yHeat = m.predictValue(X);

    amplitude = y[startIndex: endIndex, :] - yHeat[startIndex: endIndex, :];
    print(amplitude.A.flatten().tolist());

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();
    plt.subplot(211);
    plt.plot(X.A.flatten(), y.A.flatten(), "-xb");
    plt.plot(X.A.flatten(), yHeat.A.flatten(), "-r");
    plt.subplot(212);
    plt.plot(X[startIndex: endIndex, :].A.flatten(), y[startIndex: endIndex, :].A.flatten(), "-xb");
    plt.plot(X[startIndex: endIndex, :].A.flatten(), yHeat[startIndex: endIndex, :].A.flatten(), "-r");
    plt.show(block=True);
    plt.close();


def testChange():
    startIndex, endIndex = 95, 120;
    data = [];
    y = np.mat(data).T;

    h, timespan, M = 1, 6, 2;
    X = np.mat(np.arange(y.shape[0])).T;
    f = LinearRegression.RegressionSplineFunction(int(h * 60 / timespan) + M - 2, M);
    m = LinearRegression.LinearRegression().fit(X, y, [f]);
    yHeat = m.predictValue(X);

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();
    plt.subplot(211);
    plt.plot(X.A.flatten(), y.A.flatten(), "-xb");
    plt.plot(X.A.flatten(), yHeat.A.flatten(), "-r");
    plt.plot(X[startIndex: endIndex, :].A.flatten(), y[startIndex: endIndex, :].A.flatten(), "or");
    plt.subplot(212);
    plt.plot(X[startIndex: endIndex, :].A.flatten(), y[startIndex: endIndex, :].A.flatten(), "-xb");
    plt.plot(X[startIndex: endIndex, :].A.flatten(), yHeat[startIndex: endIndex, :].A.flatten(), "-r");
    plt.show(block=True);
    plt.close();


def testThreshold():
    startIndex, endIndex = 5857, 5882;
    data = [];
    y = np.mat(data).T;
    X = np.mat(np.arange(y.shape[0])).T;

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();
    plt.subplot(211);
    plt.plot(X.A.flatten(), y.A.flatten(), "-xb");
    plt.plot(X[startIndex: endIndex, :].A.flatten(), y[startIndex: endIndex, :].A.flatten(), "or");
    plt.subplot(212);
    plt.plot(X[startIndex - 5 * 120: endIndex, :].A.flatten(), y[startIndex - 5 * 120: endIndex, :].A.flatten(), "-xb");
    plt.plot(X[startIndex: endIndex, :].A.flatten(), y[startIndex: endIndex, :].A.flatten(), "or");
    plt.show(block=True);
    plt.close();


def testConstant():
    startIndex, endIndex = 5851, 5881;
    data = [];
    y = np.mat(data).T;
    X = np.mat(np.arange(y.shape[0])).T;

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();
    plt.get_current_fig_manager().window.maximize();
    plt.subplot(211);
    plt.plot(X.A.flatten(), y.A.flatten(), "-xb");
    plt.plot(X[startIndex: endIndex, :].A.flatten(), y[startIndex: endIndex, :].A.flatten(), "or");
    plt.subplot(212);
    plt.plot(X[startIndex - 1 * 120: endIndex, :].A.flatten(), y[startIndex - 1 * 120: endIndex, :].A.flatten(), "-xb");
    plt.plot(X[startIndex: endIndex, :].A.flatten(), y[startIndex: endIndex, :].A.flatten(), "or");
    plt.show(block=True);
    plt.close();


if __name__ == '__main__':
    # for root, folders, files in os.walk("/media/WindowsE/Data/PARS/JNLH/AllYiCuiHua/ISYS_history_20210422_20210629"):
    #     for path in files:
    #         if not path.endswith(".npy"):
    #             continue;
    #
    #         X = np.load(os.path.join(root, path));
    #         if X.shape != (293760, 2):
    #             print(path);

    # testSpeed();
    # testAmplitude();
    # testThreshold();
    # testConstant();
    # testChange();

    # detectBreakout(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 60);
    # detectAmplitude(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 60);
    # detectSpeed(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 60);
    # detectSpeedFNN(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 60);
    # detectChange(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 60);
    # detectConstant(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 60);
    # DataHelper.testRunsLeft(np.mat([1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1]).T);

    startTime = time.time();

    try:
        NNTest.test();
        # PyTorchTest.test();
    finally:
        print(f"elapsed time: {time.time() - startTime}");

    # Hitters
    # data = np.mat(np.loadtxt("/media/WindowsD/WorkSpace/统计学习导论/Data/Hitters.csv", delimiter = ",", dtype = np.object))[1:, 1:];
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
    # y = np.log(np.mat(data[r, -2], np.float));
    # X = np.mat(np.delete(data[r, :], -2, 1), np.float);

    # Credit
    # data = np.mat(np.loadtxt("/media/WindowsD/WorkSpace/统计学习导论/Data/Credit.csv", delimiter = ",", dtype = np.object))[1:, 1:];
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
    # data = np.mat(np.loadtxt("/media/WindowsD/WorkSpace/统计学习导论/Data/Wage.csv", delimiter = ",", dtype = np.object))[1:, 1:];
    # y = np.mat(data[:, -1], np.float);
    # X = np.mat(data[:, 1], np.float);

    # Heart
    # data = np.mat(np.loadtxt("/media/WindowsD/WorkSpace/统计学习导论/Data/Heart.csv", delimiter = ",", dtype = np.object))[1:, 1:];
    # r1 = (data[:, 2] == '"typical"').A.flatten();
    # r2 = (data[:, 2] == '"nontypical"').A.flatten();
    # r3 = (data[:, 2] == '"asymptomatic"').A.flatten();
    # data[r1, 2] = 1;
    # data[r2, 2] = 2;
    # data[r3, 2] = 3;
    # data[~np.logical_or(np.logical_or(r1, r2), r3), 2] = 0;
    # r1 = (data[:, -2] == '"fixed"').A.flatten();
    # r2 = (data[:, -2] == '"reversable"').A.flatten();
    # data[r1, -2] = 1;
    # data[r2, -2] = 2;
    # data[~np.logical_or(r1, r2), -2] = 0;
    # r = (data[:, -1] == '"Yes"').A.flatten();
    # data[r, -1] = 1;
    # data[~r, -1] = 0;
    # r = np.logical_and((data[:, -2] != 'NA').A.flatten(), (data[:, -3] != 'NA').A.flatten());
    # y = np.mat(data[r, -1], np.float);
    # X = np.mat(np.delete(data[r, :], -1, 1), np.float);

    # Loan
    # data = np.mat(np.loadtxt("/media/WindowsD/WorkSpace/统计学习导论/Data/Loan.csv", delimiter = ","));
    # y = data[:, -1];
    # X = data[:, [0, 1, 2, 3]];

    # Carseats
    # data = np.mat(np.loadtxt("/media/WindowsD/WorkSpace/统计学习导论/Data/Carseats.csv", delimiter = ",", dtype = np.object))[1:, 1:];
    # r1 = (data[:, 6] == '"Medium"').A.flatten();
    # r2 = (data[:, 6] == '"Good"').A.flatten();
    # data[r1, 6] = 1;
    # data[r2, 6] = 2;
    # data[~np.logical_or(r1, r2), 6] = 0;
    # r = (data[:, 9] == '"Yes"').A.flatten();
    # data[r, 9] = 1;
    # data[~r, 9] = 0;
    # r = (data[:, 10] == '"Yes"').A.flatten();
    # data[r, 10] = 1;
    # data[~r, 10] = 0;
    # y = np.mat(np.array(data[:, 0], np.float) > 8, dtype = np.float);
    # X = np.mat(data[:, 1:], np.float);

    # Boston
    # data = np.mat(np.loadtxt("/media/WindowsD/WorkSpace/统计学习导论/Data/Boston.csv", delimiter=",", dtype=np.object))[1:,1:];
    # y = np.mat(data[:, -1], np.float);
    # X = np.mat(data[:, :-1], np.float);
    #
    # D = np.hstack((X, y));

    # Hitters
    # tree = DecisionTree.DecisionTree(DecisionTree.RNode, DecisionTree.RssDataSplitter());
    # tree.fit(D, [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, False]);
    # errors = [];
    # forest = DecisionTree.RandomForest(DecisionTree.RNode, DecisionTree.RssDataSplitter());
    # for i in range(200):
    #     errors.append(forest.fit(D, [True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, False], i + 1).OOBError);
    # print(errors);

    # Heart
    # tree = DecisionTree.DecisionTree(DecisionTree.CNode, DecisionTree.GiniIndexSplitter());
    # tree.fit(D, [True, False, False, True, True, True, True, True, True, True, True, True, False]);
    # errors = [];
    # forest = DecisionTree.RandomForest(DecisionTree.CNode, DecisionTree.GiniIndexSplitter());
    # for i in range(400):
    #     errors.append(forest.fit(D, [True, False, False, True, True, True, True, True, True, True, True, True, False], i + 1).OOBError);
    # print(errors);

    # Loan
    # tree = DecisionTree.DecisionTree(DecisionTree.CNode, DecisionTree.InformationGainSplitter());
    # tree.fit(D, [False] * 4);
    # forest = DecisionTree.RandomForest(DecisionTree.CNode, DecisionTree.InformationGainSplitter());
    # forest.fit(D, [False] * 4, 10000);

    # Carseats
    # tree = DecisionTree.DecisionTree(DecisionTree.CNode, DecisionTree.GiniIndexSplitter());
    # tree.fit(D, [True, True, True, True, True, False, True, True, False, False]);

    # Boston
    # tree = DecisionTree.DecisionTree(DecisionTree.RNode, DecisionTree.RssDataSplitter());
    # tree.fit(D, [True] * X.shape[1]);
    # forest = DecisionTree.RandomForest(DecisionTree.RNode, DecisionTree.RssDataSplitter());
    # forest.fit(D, [True] * X.shape[1], 500, variableCount = X.shape[1]);

    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.maximize();
    # plt.plot(list(range(1, len(errors) + 1)), errors, "-or");
    # plt.show(block=True);
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
    # KMeansTest.testGapStatistic();

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
