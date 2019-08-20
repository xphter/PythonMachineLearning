import os;
import numpy as np;
from matplotlib import pyplot as plt

import KMedians;
import DataHelper;


def testKMedians():
    data = np.mat(np.loadtxt(r"data\kmeans2.txt", delimiter = "\t"));
    index, distance, center = KMedians.KMedians().clustering(data, 3);
    print(center);

    fig = plt.figure();
    ax = plt.subplot(111);

    colors = ["g", "b", "y", "k"];
    markers = ["*", "+", "D", "s"];

    for i in range(0, center.shape[0]):
        ax.scatter(data[(index == i).A.flatten(), 0].A.flatten(), data[(index == i).A.flatten(), 1].A.flatten(), color = colors[i], marker = markers[i]);
        ax.scatter(center[i, 0], center[i, 1], color = ["r"], marker = "x");

    plt.show();
