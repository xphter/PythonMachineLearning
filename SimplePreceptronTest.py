import numpy as np;
from SimplePreceptron import *;


def testSimplePreceptron():
    X = np.mat(np.loadtxt(r"data\iris\iris.txt", delimiter = ","));
    X[(X[:, -1] != 0).A.flatten(), -1] = -1;
    X[(X[:, -1] == 0).A.flatten(), -1] = 1;

    sp = SimplePerceptron(1);
    sp.train(X);
    sp.predict(X[:, :-1]);

