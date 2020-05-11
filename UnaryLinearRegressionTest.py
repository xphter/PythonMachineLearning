import numpy as np;
import matplotlib.pyplot as plt;

from UnaryLinearRegression import *;


def __run(x, y):
    lr = UnaryLinearRegression();
    lr.fit(x, y);
    lr.sigLevel = None;
    return lr;


def __show(x, y, line):
    plt.figure(1, (16, 10));
    plt.scatter(x.A.flatten().tolist(), y.A.flatten().tolist(), marker = "x");
    plt.plot([x.min(), x.max()], [line.predictValue(x.min()), line.predictValue(x.max())], color = "r");
    plt.show();


def testUnaryLinearRegression():
    data = np.mat(np.loadtxt("data/advertising.txt", delimiter = ","));

    y = data[:, -1];

    x = data[:, 1];
    tv = __run(x, y);
    tv.predictInterval(x[0, 0]);
    tv.predictInterval(x);
    __show(x, y, tv);
    print("tv: ");
    print(tv);
    print("");

    x = data[:, 2];
    radio = __run(x, y);
    radio.predictInterval(x[0, 0]);
    radio.predictInterval(x);
    __show(x, y, radio);
    print("radio: ");
    print(radio);
    print("");

    x = data[:, 3];
    newspaper = __run(x, y);
    newspaper.predictInterval(x[0, 0]);
    newspaper.predictInterval(x);
    __show(x, y, newspaper);
    print("newspaper: ");
    print(newspaper);
    print("");
