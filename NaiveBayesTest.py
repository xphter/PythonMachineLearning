import numpy as np;
from NaiveBayes import *;


def testNaiveBayes():
    X = np.mat(np.loadtxt(r"data\iris\iris.txt", delimiter = ","));
    numbers = np.mat([0] * 4);

    nb = NaiveBayes(1);
    nb.train(X, numbers);
    result = nb.predict(X);

    print(X[(X[:, -1] != result).A.flatten(), :].shape[0] / X.shape[0]);