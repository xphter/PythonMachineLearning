import os;
import numpy as np;
from matplotlib import pyplot as plt

import KMeans;
import DataHelper;


def testKMeans():
    data = np.loadtxt("data/kmeans1.txt", delimiter = "\t");
    index, distance, center = KMeans.KMeans.optimalClustering(data, 3, 1000);
    print("the final center is:\r\n{0}".format(center));

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.showMaximized();

    colors = ["g", "b", "y", "k", "r", "m", "c"];
    markers = ["*", "+", "D", "s", "h", "v", "d"];

    for i in range(0, len(center)):
        plt.scatter(data[index == i, 0], data[index == i, 1], color = colors[i], marker = markers[i]);
        plt.scatter(center[i, 0], center[i, 1], color = ["r"], marker = "x");

    plt.show(block=True);


def testElbowMethod():
    count = 1;
    # X = np.mat(np.loadtxt("data/kmeans1.txt", delimiter = "\t"));
    # X = np.load("/media/WindowsE/Data/PARS/JNLH/YuanLiaoDaiShui/trainSet/train_all_75.npy")[:, 57].reshape(-1, 1);
    X = np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/30/2020-08-01/data.npy")[:, 1].reshape(-1, 1);

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.showMaximized();
    plt.hist(X, bins = 1000);
    plt.show(block=True);

    selector = KMeans.ElbowMethod(True);
    index, distance, center = KMeans.KMeans.optimalK(X, 5, count, selector);

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.showMaximized();
    plt.hist(X, bins=1000);
    for i in range(len(center)):
        plt.axvline(center[i, 0], color = "r");
    plt.show(block=True);

    print("the optimal k is {0}".format(len(center)));
    print("the centers is:\r\n{0}".format(center));


def testGapStatistic():
    count = 10;
    # X = np.array(np.loadtxt("data/kmeans1.txt", delimiter = "\t"));
    # X = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/YuanLiaoDaiShui/trainSet/train_all_75.npy"))[:, 68];#57
    # X = np.mat(np.load("/media/WindowsD/WorkSpace/data.npy"))[:, 0];
    X = np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/30/2020-08-01/data.npy")[:, 1].reshape(-1, 1);

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.showMaximized();
    plt.hist(X, bins = 1000);
    plt.show(block=True);

    selector1 = KMeans.GapStatistic(20, True);
    selector2 = KMeans.ElbowMethod(True);
    index, distance, center = KMeans.KMeans.optimalK(X, 5, count, KMeans.CombinationOptimalKSelector([selector1, selector2]));

    # plt.figure(1, (12, 8));
    # plt.get_current_fig_manager().window.showMaximized();
    #
    # colors = ["g", "b", "y", "k", "r", "m", "c"];
    # markers = ["*", "+", "D", "s", "h", "v", "d"];
    #
    # for i in range(0, len(center)):
    #     plt.scatter(X[index == i, 0], X[index == i, 1], color = colors[i], marker = markers[i]);
    #     plt.scatter(center[i, 0], center[i, 1], color = ["r"], marker = "x");
    #
    # plt.show(block=True);

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.showMaximized();
    plt.hist(X, bins=1000);
    for i in range(len(center)):
        plt.axvline(center[i, 0], color = "r");
    plt.show(block=True);
