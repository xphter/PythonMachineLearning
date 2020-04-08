import numpy as np;
import matplotlib.pyplot as plt;
import mpl_toolkits.mplot3d as p3d;

from MultipleLinearRegression import *;


def testMultipleLinearRegression():
    data = np.mat(np.loadtxt("data/advertising.txt", delimiter = ","));

    y = data[:, -1];
    X = data[:, 1:-1];

    model = MultipleLinearRegression();
    model.fit(X, y);
    model.predictValue(X[0, :]);
    model.predictInterval(X);
    print(model);

