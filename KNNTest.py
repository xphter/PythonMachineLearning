import os;
import numpy as np;

import KNN;
import DataHelper;


def testKNN():
    digit = None;
    X_train, X_test = [], [];

    folderPath = r"data\digits\trainingDigits";
    for fileName in os.listdir(folderPath):
        with open(folderPath + "\\" + fileName, "rt") as file:
            digit = list(np.array([[int(d) for d in line if d != "\r" and d != "\n"] for line in file.readlines()]).flatten());
            digit.append(int(fileName[0]));
            X_train.append(digit);

    folderPath = r"data\digits\testDigits";
    for fileName in os.listdir(folderPath):
        with open(folderPath + "\\" + fileName, "rt") as file:
            digit = list(np.array([[int(d) for d in line if d != "\r" and d != "\n"] for line in file.readlines()]).flatten());
            digit.append(int(fileName[0]));
            X_test.append(digit);

    X_train = np.mat(X_train);
    X_test = np.mat(X_test);

    knn = KNN.KNN(X_train, X_train.shape[1] - 1,
                  distanceCalculator = lambda X, v: 1 - DataHelper.calcJaccardCoefficient(X, v),
                  isNormalizeFeatures = False);

    errorCount = 0;

    for row in X_test:
        if knn.getMostFrequentlyClass(knn.findKNN(row, 3)[0]) != row[0, -1]:
            errorCount += 1;

    print(errorCount);
    print(errorCount * 100 / X_test.shape[0]);