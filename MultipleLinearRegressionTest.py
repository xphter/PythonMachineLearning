import numpy as np;
import matplotlib.pyplot as plt;
import mpl_toolkits.mplot3d as p3d;

import DataHelper;
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
    plt.show(block = True);


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


def __loadData():
    data = np.mat(np.loadtxt("data/hitters.csv", delimiter = ",", dtype = np.object));

    names = np.mat(data[0, 1:], dtype = np.str).A.flatten().tolist();
    names.remove(names[-2]);
    names = np.array(names);

    r = (data[:, -1] == '"N"').A.flatten();
    data[r, -1] = 1;
    data[~r, -1] = 0;
    r = (data[:, -6] == '"W"').A.flatten();
    data[r, -6] = 1;
    data[~r, -6] = 0;
    r = (data[:, -7] == '"N"').A.flatten();
    data[r, -7] = 1;
    data[~r, -7] = 0;

    r = (data[:, -2] != 'NA').A.flatten();
    X = np.mat(np.hstack((data[r, 1:-2], data[r, -1]))[1:, :], dtype = np.float);
    y = np.mat(data[r, -2][1:, :], dtype = np.float);

    return names, X, y;


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


def testOptimalSubset():
    names, X, y = __loadData();
    p = X.shape[1];
    models = [];

    for i in range(0, p):
        layerModels = [];

        for j in DataHelper.combinations(p, i + 1):
            model = MultipleLinearRegression();
            model.fit(X[:, j], y);
            layerModels.append((j, model));

        models.append(layerModels[np.argmin([item[1].rss for item in layerModels])]);

    for indices, model in models:
        print("variables: {0}, rss: {1}, r^2: {2}, aic: {3}, bic: {4}, adjusted-r^2: {5}".format(
            ", ".join(names[indices]), model.rss, model.r2, model.aic, model.bic, model.adjustedR2));
    print("");

    print("selected model: {0}\r\n".format(np.argmin([item[1].bic for item in models])));

    print("\r\n\r\n".join([item[1].__str__() for item in models]));


def testForwardSelection():
    names, X, y = __loadData();
    n, p = X.shape;
    models = [];

    selectedFeatures = [];
    remainFeatures = list(range(0, p));
    previousX = np.mat(np.zeros((n, 0)));

    for i in range(0, p):
        layerModels = [];

        for j in range(0, len(remainFeatures)):
            model = MultipleLinearRegression();
            model.fit(np.hstack((previousX, X[:, remainFeatures[j]])), y);
            layerModels.append((remainFeatures[j], model));

        index = np.argmin([item[1].rss for item in layerModels]);
        selectedFeatures = selectedFeatures + [layerModels[index][0]];
        remainFeatures.remove(layerModels[index][0]);

        previousX = np.hstack((previousX, X[:, layerModels[index][0]]));
        models.append((selectedFeatures, layerModels[index][1]));

    for j, model in models:
        print("variables: {0}, rss: {1}, r^2: {2}, aic: {3}, bic: {4}, adjusted-r^2: {5}".format(
            ", ".join(names[j]), model.rss, model.r2, model.aic, model.bic, model.adjustedR2));
    print("");

    print("selected model: {0}\r\n".format(np.argmin([item[1].bic for item in models])));

    print("\r\n\r\n".join([item[1].__str__() for item in models]));


def testBackwardSelection():
    names, X, y = __loadData();
    n, p = X.shape;
    models = [];

    model = MultipleLinearRegression();
    model.fit(X, y);
    models.append((list(range(0, p)), model));

    remainFeatures = list(range(0, p));

    for i in range(0, p - 1):
        layerModels = [];

        for j in range(0, len(remainFeatures)):
            model = MultipleLinearRegression();
            model.fit(X[:, remainFeatures[0:j] + remainFeatures[j+1:]], y);
            layerModels.append((remainFeatures[j], model));

        index = np.argmin([item[1].rss for item in layerModels]);
        remainFeatures.remove(layerModels[index][0]);

        models.append((remainFeatures[:], layerModels[index][1]));

    for j, model in models:
        print("variables: {0}, rss: {1}, r^2: {2}, aic: {3}, bic: {4}, adjusted-r^2: {5}".format(
            ", ".join(names[j]), model.rss, model.r2, model.aic, model.bic, model.adjustedR2));
    print("");

    print("selected model: {0}\r\n".format(np.argmin([item[1].bic for item in models])));

    print("\r\n\r\n".join([item[1].__str__() for item in models]));
