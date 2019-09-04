import os;
import time;
import threading;
import numpy as np;

import PCA;
import DataHelper;
import CutForest;


g_stop = False;


def __onFillProgressChange(count, total):
    global g_stop;
    print("fill process: %d / %d" % (count, total));
    return g_stop;


def __onTrainProgressChange(count, total):
    global g_stop;
    stop = g_stop;
    print("train process: %d / %d" % (count, total));
    g_stop = stop;
    return stop;


def testCutForest():
    global g_stop;

    data1 = np.mat(np.load(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\trainSet\train_all_75.npy"));
    # data2, mu, sigma, data3, p = PCA.projectData(data1, 0.999);
    # minValues, maxValues = data1.min(0), data1.max(0);
    # trainSet = (data1 - minValues) / (maxValues - minValues);
    trainSet = data1;

    testSet = np.mat(np.load(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\testSet\test_all_75.npy"));
    # testSet = (testSet - mu) / sigma;
    # testSet = PCA.performProject(testSet, mu, sigma, p);
    # testSet = (testSet - minValues) / (maxValues - minValues);

    index = 9;
    forest = CutForest.CutForest(treeCount = 200, subSamplingSize = 2 ** index);
    forest.fillProgressChanged.addListener(__onFillProgressChange);
    forest.trainProgressChanged.addListener(__onTrainProgressChange);
    forest.fill(trainSet);
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
    #     if i % 100 == 0:
    #         print(i);
    #         np.savetxt("scores.txt", np.array(scores), delimiter = ",");
    #
    #     scores.append(forest.getAnomalyScore(testSet[i, :]));

    np.savetxt("scores.txt", scores, delimiter = ",");
