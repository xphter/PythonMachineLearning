import os;
import numpy as np;

import LogisticRegression;


def testLogisticRegression():
    testData = np.mat(np.loadtxt("data/horseColicTest.txt"));
    trainingData = np.mat(np.loadtxt("data/horseColicTraining.txt"));

    lr = LogisticRegression.LogisticRegression();
    lr.train(trainingData, 0.001, 0.00000001);

    actualValue = testData[:, -1];
    predictValue = lr.predict(testData[:, :-1]);

    tp = predictValue[(actualValue == 1).A.flatten(), :].sum();
    fp = predictValue[(actualValue == 0).A.flatten(), :].sum();
    tn = -(predictValue - 1)[(actualValue == 0).A.flatten(), :].sum();
    fn = -(predictValue - 1)[(actualValue == 1).A.flatten(), :].sum();

    accuracy = (tp + tn) / (tp + fp + tn + fn);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * tp / (2 * tp + fp + fn);

    print("accuracy: {0}, precision: {0}, recall: {1}, f1: {2}".format(accuracy, precision, recall, f1));