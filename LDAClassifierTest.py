import os;
import time;
import threading;
import numpy as np;
import matplotlib.pyplot as plt;

import LDAClassifier;


def __testCore(trainingData, testData, classifier):
    classifier.train(trainingData[:, :-1], trainingData[:, -1]);

    actualValue = testData[:, -1];
    predictValue, posterior = classifier.predict(testData[:, :-1]);

    tp = predictValue[(actualValue == 1).A.flatten(), :].sum();
    fp = predictValue[(actualValue == 0).A.flatten(), :].sum();
    tn = -(predictValue - 1)[(actualValue == 0).A.flatten(), :].sum();
    fn = -(predictValue - 1)[(actualValue == 1).A.flatten(), :].sum();

    accuracy = (tp + tn) / (tp + fp + tn + fn);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * tp / (2 * tp + fp + fn);

    print("accuracy: {0}, precision: {1}, recall: {2}, f1: {3}".format(accuracy, precision, recall, f1));


def testLDAClassifier():
    testData = np.mat(np.loadtxt("data/horseColicTest.txt"));
    trainingData = np.mat(np.loadtxt("data/horseColicTraining.txt"));

    __testCore(trainingData, testData, LDAClassifier.LDAClassifier());
    __testCore(trainingData, testData, LDAClassifier.QDAClassifier());
