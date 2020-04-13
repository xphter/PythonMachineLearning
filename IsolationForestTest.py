import os;
import time;
import threading;
import numpy as np;
import matplotlib.pyplot as plt;

import PCA;
import DataHelper;
import IsolationForest;


def testIsolationForest():
    trainSet = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/Models/d77d645a6e204fe4b5fd606db3749ba9/data.npy"));
    # trainSet = np.mat(np.load("Z:/e/Data/PARS/JNLH/Models/43c310835c23453e9af7830f61a8c167/data.npy"));
    print("load completed");

    index = 9;
    finder = IsolationForest.CurvesThresholdFinder(0.63, 0.68, 0.7, 0.8, 0.73, True);
    forest = IsolationForest.IsolationForest(treeCount = 200, subSamplingSize = 2 ** index, thresholdFinder = finder);

    forest.fill(trainSet);
    print("fill completed");

    scores = forest.train(trainSet);
    print("train completed");

    np.savetxt("scores.txt", scores, delimiter = ",");
    plt.figure(1, (16, 10));
    plt.hist(scores, 1000);
    plt.show();
