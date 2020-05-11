import numpy as np;
import matplotlib.pyplot as plt;
import mpl_toolkits.mplot3d as p3d;

from UnaryLinearRegression import *;
from MultipleLinearRegression import *;


def __run(X, y):
    lr = MultipleLinearRegression();
    lr.fit(X, y);
    lr.sigLevel = None;
    return lr;


def __show(X, y, line):
    predictY = line.predictValue(X);

    plt.figure(1, (16, 10));
    plt.scatter(predictY.A.flatten(), (y - predictY).A.flatten(), marker = "x");
    plt.show();


def __vif(X):
    p = X.shape[1];
    y, X1 = None, None;

    for i in range(0, p):
        y = X[:, i];

        if i == 0:
            X1 = X[:, 1:];
        elif i == p - 1:
            X1 = X[:, :-1];
        else:
            X1 = np.hstack((X[:, :i], X[:, i + 1:]));

        lr = MultipleLinearRegression();
        lr.fit(X1, y);
        print("VIF of {0} attribute: {1}".format(i + 1, 1 / (1 - lr.r2)));


def testMultipleLinearRegression():
    data = np.mat(np.loadtxt("data/advertising.txt", delimiter = ","));

    y = data[:, -1];

    X = data[:, 1:-1];
    model = __run(X, y);
    model.predictValue(X[0, :]);
    model.predictInterval(X);
    __show(X, y, model);
    print("original: ");
    print(model);
    __vif(X);
    print("");

    X = np.hstack((X[:, :-1], np.multiply(X[:, 0], X[:, 1])));
    model = __run(X, y);
    model.predictValue(X[0, :]);
    model.predictInterval(X);
    __show(X, y, model);
    print("interaction: ");
    print(model);
    __vif(X);
    print("");

