#!/usr/bin/python3
import os;
from DecisionTree import *;
from Sampling import *;
from PCA import *;
from IsolationForest import *;
import numpy as np;
import numpy.matlib as npm;
import numpy.linalg as npl;
import matplotlib.pyplot as plt;

import IsolationForestTest;
import CutForestTest;


data = np.mat(np.loadtxt(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\数据分析\砍伐森林\75_200_512.txt"));
q1, q2, q3 = np.quantile(data, [0.25, 0.5, 0.75]);
iqr = 1.5 * (q3 - q1);

result1 = data[:, (data >= q3 + iqr).A.flatten()];
result2 = data[:, (data >= data.mean() + 3 * data.std()).A.flatten()];


def getPageNumber(ordinal):
    pageNumber, totalCount = 0, 0;
    numbers = np.mat(np.loadtxt(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\testSet\numbers.txt", delimiter = ",")).T;

    for i in range(0, numbers.shape[0]):
        if (i == numbers.shape[0] - 1) or (ordinal > totalCount) and (ordinal <= totalCount + numbers[i + 1, 0]):
            pageNumber = i + 1;
            break;

        totalCount += numbers[i, 0];

    return pageNumber, ordinal - totalCount;


# print(getPageNumber(263541));
# print(getPageNumber(283836));
# print(getPageNumber(284888));

# IsolationForestTest.testIsolationForest();
# CutForestTest.testCutForest();

# X = np.mat(np.loadtxt(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\2019\all.txt", delimiter = ","));
# Y = X[:, 1:];
# np.savetxt(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\2019\all_75.txt", Y, delimiter = ",");
# np.save(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\2019\all_75.npy", Y);

# matrix, dataSet = None, None;
# folderPath = r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\2019";
#
# for fileName in os.listdir(folderPath):
#     dataSet = np.mat(np.loadtxt(folderPath + "\\" + fileName, delimiter = ","));
#
#     if matrix is None:
#         matrix = dataSet;
#     else:
#         matrix = np.vstack((matrix, dataSet));
#
# np.savetxt(r"E:\Data\PARS\JNLH\YuanLiaoDaiShui\2019\all.txt", matrix, delimiter = ",");



# X = np.mat(np.loadtxt(r"data\datingTestSet.txt", delimiter = "\t"));
# Y = X[:, -1];
# X = X[:, :-1:];
# X, mu, sigma = DataHelper.normalizeFeatures(X);
# X = np.hstack((X, Y));
#
# ax = plt.figure().add_subplot(111);
# colors = np.empty(Y.shape[0], np.str);
# colors[(Y == 0).A.flatten()] = "r";
# colors[(Y == 1).A.flatten()] = "g";
# colors[(Y == 2).A.flatten()] = "b";
# ax.scatter(X[:, 0].A.flatten(), X[:, 1].A.flatten(), c = colors);
# plt.show();


# data = np.loadtxt(r"E:\8\machinelearninginaction\Ch08\ex0.txt", delimiter = "\t");
#
# X = npm.mat(data[:, [0, 1]]);
# Y = npm.mat(data[:, -1]).T;
# Theta = (X.T * X).I * X.T * Y;
#
# Y2 = X * Theta;
#
# f = plt.figure();
# ax = f.add_subplot(111);
# ax.scatter(X[:, 1].flatten().A[0], Y.flatten().A[0]);
# ax.plot(X[:, 1].flatten().A[0], Y2.flatten().A[0]);
# plt.show();


# with open("data/iris/iris.data", "r") as stream:
#     for line in stream.readlines():
#         if len(line.strip()) == 0:
#             continue;
#
#         matrix.append([float(item) for item in line.strip().replace("Iris-setosa", "1").replace("Iris-versicolor", "2").replace("Iris-virginica","3").split(",")]);

    #matrix = [[float(item) for item in line.strip().replace("Iris-setosa", "1").replace("Iris-versicolor", "2").replace("Iris-virginica", "3").split(",")] for line in stream.readlines() if len(line.strip(())) > 0];

#matrix = [row[1:] + row[:1] for row in matrix];

# trainSet, testSet = holdOutSampling(matrix, 0.7);
#
# forest = [];
# types = [1] * 4;
#
# for i in range(0, 20):
#     treeSet = bootstrapSampling(matrix);
#     features = chooseFeature(4, 4);
#     forest.append(trainCART(treeSet, types, features));

# for tree in forest:
#     showTree(tree);

#tree = trainCART(trainSet, types);
# showTree(tree);
# pruningTree(tree, 2);
#showTree(tree);

# errorCount = 0;
#
# for row in matrix:
#     if classifyForest(forest, row) != row[-1]:
#         errorCount += 1;
#
# print("{0}, {1}%".format(errorCount, errorCount * 100 / len(testSet)));







