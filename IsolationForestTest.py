import os;
import time;
import threading;
import numpy as np;

import PCA;
import DataHelper;
import IsolationForest;


g_stop = False;


def __onFillProcessChange(count, total):
    global g_stop;
    print("fill process: %d / %d" % (count, total));
    return g_stop;


def __onTrainProcessChange(count, total):
    global g_stop;
    stop = g_stop;
    print("train process: %d / %d" % (count, total));
    g_stop = stop;
    return stop;


def testIsolationForest():
    global g_stop;

    data1 = np.mat(np.load(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\trainSet\train_all_75.npy"));
    data2, mu, sigma, data3, p = PCA.projectData(data1, 0.999);
    trainSet = data3;

    testSet = np.mat(np.load(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\testSet\test_all_75.npy"));
    testSet = PCA.performProject(testSet, mu, sigma, p);

    index = 9;
    forest = IsolationForest.IsolationForest(treeCount = 200, subSamplingSize = 2 ** index);
    forest.fillProcessChanged.addListener(__onFillProcessChange);
    forest.trainProcessChanged.addListener(__onTrainProcessChange);
    forest.fill(trainSet, heightLimit = index);
    print("fill completed");

    scores = [];

    def train():
        global g_stop;
        nonlocal scores;

        scores = forest.train(testSet);
        g_stop = True;

    worker = threading.Thread(target = train);
    worker.start();

    while not g_stop:
        time.sleep(1);

    worker.join();
    print("train completed");

    # for i in range(0, testSet.shape[0]):
    #     if i % 10000 == 0:
    #         print(i);
    #
    #     scores.append(forest.getAnomalyScore(testSet[i, :]));

    np.savetxt("scores.txt", scores, delimiter = ",");
