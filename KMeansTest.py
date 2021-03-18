import os;
import numpy as np;
from matplotlib import pyplot as plt

import KMeans;
import DataHelper;


def testKMeans():
    data = np.mat(np.loadtxt("data/kmeans1.txt", delimiter = "\t"));
    index, distance, center = KMeans.KMeans.optimalClustering(data, 3, 20000);
    print("the final center is:\r\n{0}".format(center));

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();

    colors = ["g", "b", "y", "k", "r", "m", "c"];
    markers = ["*", "+", "D", "s", "h", "v", "d"];

    for i in range(0, center.shape[0]):
        plt.scatter(data[(index == i).A.flatten(), 0].A.flatten(), data[(index == i).A.flatten(), 1].A.flatten(), color = colors[i], marker = markers[i]);
        plt.scatter(center[i, 0], center[i, 1], color = ["r"], marker = "x");

    plt.show(block=True);


def ElbowMethod():
    count = 1;
    # X = np.mat(np.loadtxt("data/kmeans1.txt", delimiter = "\t"));
    X = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/YuanLiaoDaiShui/trainSet/train_all_75.npy"))[:, 57];

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();
    plt.hist(X, bins = 1000);
    plt.show(block=True);

    selector = KMeans.ElbowMethod(True);
    index, distance, center = KMeans.KMeans.optimalK(X, 10, count, selector);

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();
    plt.hist(X, bins=1000);
    for i in range(center.shape[0]):
        plt.axvline(center[i, 0], color = "r");
    plt.show(block=True);

    print("the optimal k is {0}".format(center.shape[0]));
    print("the centers is:\r\n{0}".format(center));


def GapStatistic():
    count = 1;
    # X = np.mat(np.loadtxt("data/kmeans1.txt", delimiter = "\t"));
    # X = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/YuanLiaoDaiShui/trainSet/train_all_75.npy"))[:, 68];#57
    # X = np.mat(np.load("/media/WindowsD/WorkSpace/data.npy"))[:, 0];
    X = np.mat(np.load("/media/WindowsE/Data/PARS/JNLH/ReasonAnalysis/60/data.npy"))[:, 1];

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();
    plt.hist(X, bins = 1000);
    plt.show(block=True);

    selector = KMeans.GapStatistic(20, True);
    index, distance, center = KMeans.KMeans.optimalK(X, 10, count, selector);

    plt.figure(1, (12, 8));
    plt.get_current_fig_manager().window.maximize();
    plt.hist(X, bins=1000);
    for i in range(center.shape[0]):
        plt.axvline(center[i, 0], color = "r");
    plt.show(block=True);
