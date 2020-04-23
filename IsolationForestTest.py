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
    # trainSet = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/Models/d77d645a6e204fe4b5fd606db3749ba9/data.npy"));
    # trainSet = np.mat(np.load("Z:/e/Data/PARS/JNLH/Models/d77d645a6e204fe4b5fd606db3749ba9/data.npy"));
    print("load completed");
    print("");

    index = 9;
    thresholds = [];
    for i in range(0, 10):
        print("/********** forest {0} **********\\".format(i + 1));
        finder = IsolationForest.CurvesThresholdFinder(0.63, 0.68, 0.73, False);
        forest = IsolationForest.IsolationForest(treeCount = 200, subSamplingSize = 2 ** index, thresholdFinder = finder);

        forest.fill(trainSet);
        print("fill completed");

        scores = forest.train(trainSet);
        print("train completed");
        print("");

        thresholds.append(forest.threshold);

    print("thresholds: {0}, mean: {1}".format(thresholds, sum(thresholds) / len(thresholds)));
