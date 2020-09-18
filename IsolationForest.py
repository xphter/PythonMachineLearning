import os;
import abc;
import math;
import multiprocessing;
import psutil;
import numpy as np;
import matplotlib.pyplot as plt;

from Errors import *;
from KMeans import *;
from UnaryLinearRegression import *;


class _Node:
    def __init__(self, samplesCount, featureIndex = None, featureValue = None, leftChild = None, rightChild = None):
        self.samplesCount = samplesCount;
        self.featureIndex = featureIndex;
        self.featureValue = featureValue;
        self.leftChild = leftChild;
        self.rightChild = rightChild;


    def __repr__(self):
        return self.__str__();


    def __str__(self):
        if self.isLeaf():
            return "Leaf node, {0} samples".format(self.samplesCount);
        else:
            return "Internal node, {0} samples, feature {1} of value {2}".format(self.samplesCount, self.featureIndex, self.featureValue);


    def isLeaf(self):
        return self.featureIndex is None;


    def getLeafCount(self):
        if self.isLeaf():
            return 1;

        return (self.leftChild.getLeafCount() if self.leftChild is not None else 0) +\
               (self.rightChild.getLeafCount() if self.rightChild is not None else 0);


class IsolationForest:
    def __init__(self, treeCount = 100, subSamplingSize = 256, thresholdFinder = None):
        if thresholdFinder is None or not isinstance(thresholdFinder, IThresholdFinder):
            raise ValueError();

        self.__treeCount = treeCount;
        self.__subSamplingSize = subSamplingSize;
        self.__treesList = [];

        self.__scores = None;
        self.__threshold = None;
        self.__thresholdFinder = thresholdFinder;


    @property
    def scores(self):
        return self.__scores;


    @property
    def threshold(self):
        return self.__threshold;


    @threshold.setter
    def threshold(self, value):
        if value <= 0.5 or value >= 1:
            raise ValueError();

        self.__threshold = value;


    def __calcHarmonicNumber(self, i):
        return np.log(i) + np.euler_gamma;


    def __calcAveragePathLength(self, psi):
        if psi < 2:
            return 0;

        if psi == 2:
            return 1;

        return 2 * self.__calcHarmonicNumber(psi - 1) - 2 * (psi - 1) / psi;


    def __getPathLength(self, instance, node, currentLength, lengthLimit):
        if node.isLeaf() or currentLength >= lengthLimit:
            return currentLength + self.__calcAveragePathLength(node.samplesCount);

        if instance[0, node.featureIndex] < node.featureValue:
            return self.__getPathLength(instance, node.leftChild, currentLength + 1, lengthLimit);
        else:
            return self.__getPathLength(instance, node.rightChild, currentLength + 1, lengthLimit);


    def __getAnomalyScore(self, instance, lengthLimit):
        length = 0;

        for tree in self.__treesList:
            length += self.__getPathLength(instance, tree, 0, lengthLimit);
        length /= self.__treeCount;

        return 1 / (2 ** (length / self.__calcAveragePathLength(self.__subSamplingSize)));


    def __hasSameFeatureValues(self, dataSet, featureIndex):
        if dataSet.shape[0] == 0:
            return True;

        result = True;
        value = dataSet[0, featureIndex];

        for rowIndex in range(0, dataSet.shape[0]):
            if dataSet[rowIndex, featureIndex] != value:
                result = False;
                break;

        return result;


    def __choiceFeatureIndex(self, features):
        if len(features) == 1:
            return features[0];

        return features[np.random.randint(0, len(features))];


    def __choiceFeatureValue(self, dataSet, featureIndex):
        values = dataSet[:, featureIndex];
        minValue, maxValue = values.min(), values.max();

        return minValue + (maxValue - minValue) * np.random.random();


    def __createNode(self, dataSet, features, currentHeight):
        samplesCount = dataSet.shape[0];

        if samplesCount == 0:
            return None;

        if samplesCount == 1:
            return _Node(samplesCount);

        for index in [item for item in features if self.__hasSameFeatureValues(dataSet, item)]:
            features.remove(index);

        if len(features) == 0:
            return _Node(samplesCount);

        featureIndex = self.__choiceFeatureIndex(features);
        featureValue = self.__choiceFeatureValue(dataSet, featureIndex);

        return _Node(samplesCount, featureIndex, featureValue,
                     self.__createNode(dataSet[(dataSet[:, featureIndex] < featureValue).A.flatten(), :], features[:], currentHeight + 1),
                     self.__createNode(dataSet[(dataSet[:, featureIndex] >= featureValue).A.flatten(), :], features[:], currentHeight + 1));


    def _createTree(self, subSet):
        return self.__createNode(subSet, list(range(0, subSet.shape[1])), 0);


    def fill(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        self.__scores = None;
        self.__threshold = None;

        n = dataSet.shape[0];
        with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
            self.__treesList = pool.map(self._createTree, [dataSet[np.random.choice(n, self.__subSamplingSize, False), :] for i in range(0, self.__treeCount)]);


    def getAnomalyScore(self, instance):
        if instance is None:
            raise ValueError();

        return self.__getAnomalyScore(instance, self.__subSamplingSize - 1);


    def train(self, dataSet):
        if self.__threshold is not None and self.__scores is not None:
            return False;

        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();
        if len(self.__treesList) != self.__treeCount:
            raise InvalidOperationError();

        with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
            self.__scores = pool.map(self.getAnomalyScore, [item for item in dataSet]);

        self.__threshold = self.__thresholdFinder.find(self.__scores);

        return True;


class IThresholdFinder(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def find(self, scores):
        pass;


class ProportionThresholdFinder(IThresholdFinder):
    def __init__(self, proportion):
        self.__proportion = max(0, max(1, proportion));


    def find(self, scores):
        scores.sort(reverse = True);
        return np.quantile(scores, self.__proportion);


class CurvesThresholdFinder(IThresholdFinder):
    MIN_SAMPLES_NUMBER = 10;
    MIN_PARALLEL_NUMBER = 10000;

    def __init__(self, minCheckValue, maxCheckValue, defaultThreshold, showPlot = False):
        self._minCheckValue = minCheckValue;
        self._maxCheckValue = maxCheckValue;
        self._defaultThreshold = defaultThreshold;
        self.__showPlot = showPlot;

        self._values = [];
        self._curves = None;
        self._leftLines = None;
        self._rightLines = None;


    def _leftOnly(self, y):
        maxValue = y.max();
        value, residual = None, None;

        for i in range(CurvesThresholdFinder.MIN_SAMPLES_NUMBER, y.shape[0]):
            line = UnaryLinearRegression();
            line.fit(np.mat(np.arange(i)).T, y[:i, 0]);
            line.sigLevel = None;
            if line.slop <= 0:
                continue;

            value = line.predictValue(i - 1);
            if value > maxValue:
                continue;

            residual = y[i:, 0] - value;

            self._leftLines[i] = (line, value);
            self._curves.append([line, i - 1, None, None, value, line.rss + (residual.T * residual)[0, 0]]);


    def _rightOnly(self, y):
        n, maxValue = y.shape[0], y.max();
        value, residual = None, None;

        for j in range(n - CurvesThresholdFinder.MIN_SAMPLES_NUMBER, 0, -1):
            line = UnaryLinearRegression();
            line.fit(np.mat(np.arange(j, n)).T, y[j:, 0]);
            line.sigLevel = None;
            if line.slop >= 0:
                continue;

            value = line.predictValue(j);
            if value > maxValue:
                continue;

            residual = y[:j, 0] - value;

            self._rightLines[j] = (line, value);
            self._curves.append([None, None, line, j, value, line.rss + (residual.T * residual)[0, 0]]);


    def _processItem(self, i, j, y, maxValue):
        leftLine, leftValue = self._leftLines[i] if i in self._leftLines else (None, None);
        if leftLine is None:
            return None;

        rightLine, rightValue = self._rightLines[j] if j in self._rightLines else (None, None);
        if rightLine is None:
            return None;

        value, residual = None, None;
        endIndex, startIndex = None, None;

        if leftValue < rightValue:
            value = rightValue;
            startIndex = j;
            endIndex = math.floor(leftLine.inverse(rightValue));
        elif rightValue < leftValue:
            value = leftValue;
            endIndex = i - 1;
            startIndex = math.ceil(rightLine.inverse(leftValue));
        else:
            endIndex = i - 1;
            startIndex = j;
            value = leftValue;

        if endIndex >= startIndex - 1 or value > maxValue:
            return None;

        residual = y[endIndex + 1:startIndex, 0] - value;
        leftRss = (leftLine.calcRss(np.mat(np.arange(i, endIndex + 1)).T, y[i:endIndex + 1, 0]) if endIndex > i - 1 else 0) + leftLine.rss;
        rightRss = (rightLine.calcRss(np.mat(np.arange(startIndex, j)).T, y[startIndex:j, 0]) if startIndex < j else 0) + rightLine.rss;

        return [leftLine, endIndex, rightLine, startIndex, value, leftRss + rightRss + (residual.T * residual)[0, 0]];


    def _bothSides(self, y):
        points = [];
        n, maxValue = y.shape[0], y.max();

        for i in range(CurvesThresholdFinder.MIN_SAMPLES_NUMBER, n - CurvesThresholdFinder.MIN_SAMPLES_NUMBER - 1):
            for j in range(n - CurvesThresholdFinder.MIN_SAMPLES_NUMBER, i, -1):
                points.append((i, j, y, maxValue));

        curves = None;
        if len(points) >= CurvesThresholdFinder.MIN_PARALLEL_NUMBER:
            with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
                curves = pool.starmap(self._processItem, points);
        else:
            curves = list(map(lambda obj: self._processItem(*obj), points));

        for item in curves:
            if item is None:
                continue;

            self._curves.append(item);


    def _fit(self, y):
        if y is None:
            raise ValueError();

        n = y.shape[0];
        if n < CurvesThresholdFinder.MIN_SAMPLES_NUMBER * 2 + 1:
            return None, None;

        self._curves = [];
        self._leftLines = {};
        self._rightLines = {};

        self._leftOnly(y);
        self._rightOnly(y);
        self._bothSides(y);

        if len(self._curves) == 0:
            value = y.mean();
            self._values.append(value);
            return [0, n - 1], [value, value];

        self._curves = np.mat(self._curves);
        leftLine, endIndex, rightLine, startIndex, value, rss = tuple(self._curves[self._curves[:, -1].argmin(0)[0, 0], :].A.flatten().tolist());
        self._values.append(value);

        if leftLine is not None and rightLine is not None:
            return [0, endIndex, endIndex + 1, startIndex - 1, startIndex, n - 1],\
                   [leftLine.predictValue(0), leftLine.predictValue(endIndex), value, value, rightLine.predictValue(startIndex), rightLine.predictValue(n - 1)];
        elif leftLine is not None:
            return [0, endIndex, n - 1], [leftLine.predictValue(0), leftLine.predictValue(endIndex), leftLine.predictValue(endIndex)];
        elif rightLine is not None:
            return [0, startIndex, n - 1], [rightLine.predictValue(startIndex), rightLine.predictValue(startIndex), rightLine.predictValue(n - 1)];
        else:
            return None, None;


    def find(self, scores):
        scale = 100;
        data = np.mat(scores).T * scale;
        indices, distances, center = KMeans(lambda X, k: np.mat([X.min(), X.mean(), X.max()]).T).clustering(data, 3, 1);
        print("anomaly score centers:{0}".format(center.T));

        checkValue = center[2, 0];
        minCheckValue = self._minCheckValue * scale;
        maxCheckValue = self._maxCheckValue * scale;
        defaultThreshold = self._defaultThreshold * scale;
        minValue = data[(indices == 2).A.flatten(), :].min(0)[0, 0];
        maxValue = data[(indices == 2).A.flatten(), :].max(0)[0, 0];

        if maxValue <= defaultThreshold:
            return defaultThreshold / scale;

        if checkValue >= defaultThreshold:
            checkValue = (minValue + checkValue) / 2;
        elif checkValue <= minCheckValue:
            checkValue = (checkValue + maxValue) / 2;
        if checkValue < minCheckValue:
            checkValue = minCheckValue;
        elif checkValue > maxCheckValue:
            checkValue = maxCheckValue;
        print("threshold check value: {0}".format(checkValue));

        i = None;
        for j in range(0, data.shape[0]):
            if data[j, 0] >= checkValue and i is None:
                i = j;

            if data[j, 0] < checkValue and i is not None:
                if j - i > CurvesThresholdFinder.MIN_SAMPLES_NUMBER * 2:
                    x, y = self._fit(data[i:j, 0]);

                    if self.__showPlot:
                        plt.figure(1, (16, 10));
                        plt.plot(list(range(0, j - i)), data[i:j, 0].A.flatten().tolist(), color = "b", marker = "x");
                        if x is not None and y is not None:
                            plt.plot(x, y, color = "r");
                        plt.show();

                i = None;
        print("threshold all values: {0}".format(self._values));

        threshold = (np.mean(self._values) if len(self._values) > 0 else defaultThreshold) / scale;
        print("threshold found: {0}".format(threshold));

        return threshold;
