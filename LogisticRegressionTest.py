import os;
import numpy as np;

import Optimizer;
import LogisticRegression;


def __testCore(trainingData, testData, optimizer):
    lr = LogisticRegression.LogisticRegression(optimizer);
    costValue = lr.train(trainingData[:, :-1], trainingData[:, -1]);
    lr.sigLevel = None;
    print("theta: {0}, value of cost: {1}".format(lr.theta, costValue));

    actualValue = testData[:, -1];
    predictValue = (lr.predictValue(testData[:, :-1]) > 0.5) - 0;

    tp = predictValue[(actualValue == 1).A.flatten(), :].sum();
    fp = predictValue[(actualValue == 0).A.flatten(), :].sum();
    tn = -(predictValue - 1)[(actualValue == 0).A.flatten(), :].sum();
    fn = -(predictValue - 1)[(actualValue == 1).A.flatten(), :].sum();

    accuracy = (tp + tn) / (tp + fp + tn + fn);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * tp / (2 * tp + fp + fn);

    print("accuracy: {0}, precision: {1}, recall: {2}, f1: {3}".format(accuracy, precision, recall, f1));


def testLogisticRegression():
    epsilon = 0.000001;
    testData = np.mat(np.loadtxt("data/horseColicTest.txt"));
    trainingData = np.mat(np.loadtxt("data/horseColicTraining.txt"));

    __testCore(trainingData, testData, Optimizer.GradientDescent(epsilon));
    __testCore(trainingData, testData, Optimizer.NewtonMethod(epsilon));
