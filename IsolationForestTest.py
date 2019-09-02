import os;
import numpy as np;

import PCA;
import DataHelper;
import IsolationForest;


def testIsolationForest():
    data1 = np.mat(np.load(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\trainSet\train_all_75.npy"));
    data2, mu, sigma, data3, p = PCA.projectData(data1, 0.999);

    trainSet = data3;
    testSet = np.mat(np.load(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\testSet\test_all_75.npy"));
    testSet = PCA.performProject(testSet, mu, sigma, p);

    index = 9;
    forest = IsolationForest.IsolationForest(treeCount = 200, subSamplingSize = 2 ** index);
    forest.fill(trainSet, heightLimit = index);

    print("train completed");

    scores = [];
    for i in range(0, testSet.shape[0]):
        if i % 10000 == 0:
            print(i);

        scores.append(forest.getAnomalyScore(testSet[i, :]));

    np.savetxt("scores.txt", np.array(scores), delimiter = ",");
