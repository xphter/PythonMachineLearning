import os;
import time;
import threading;
import numpy as np;
import matplotlib.pyplot as plt;

import PCA;
import DataHelper;
import IsolationForest;


def testIsolationForest():
    trainSet = np.mat(np.load("data/IsolationForest.npy"));
    # trainSet = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/Models/58e3466ab1b24e6591ba8c4ed28a11cf/data.npy"));
    # trainSet = np.mat(np.load("Z:/e/Data/PARS/JNLH/Models/58e3466ab1b24e6591ba8c4ed28a11cf/data.npy"));
    trainSet = trainSet[0: 10000 + 1, :];
    print("load completed");
    print("");

    index = 9;
    thresholds = [];
    for i in range(0, 1):
        print("/********** forest {0} **********\\".format(i + 1));
        finder = IsolationForest.CurvesThresholdFinder(0.63, 0.68, 0.73, False);
        forest = IsolationForest.IsolationForest(treeCount = 200, subsamplingSize = 2 ** index, finder = finder);

        forest.fill(trainSet);
        print("fill completed");

        forest.train(trainSet);
        forest.threshold = forest.threshold;
        print("train completed");
        print("");

        thresholds.append(forest.threshold);

    print("thresholds: {0}, mean: {1}".format(thresholds, sum(thresholds) / len(thresholds)));


if __name__ == '__main__':
    testIsolationForest();
