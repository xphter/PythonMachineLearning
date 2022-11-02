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
from typing import Union, List, Tuple, Callable, Any, Optional;


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
            return f"Leaf node, {self.samplesCount} samples";
        else:
            return f"Internal node, {self.samplesCount} samples, feature {self.featureIndex} of value {self.featureValue}";


    def isLeaf(self):
        return self.featureIndex is None;


    def getLeafCount(self):
        if self.isLeaf():
            return 1;

        return (self.leftChild.getLeafCount() if self.leftChild is not None else 0) +\
               (self.rightChild.getLeafCount() if self.rightChild is not None else 0);


class IThresholdFinder(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def find(self, scores : List[float]):
        pass;


class IsolationForest:
    def __init__(self, treeCount : int = 100, subsamplingSize : int = 256, finder : IThresholdFinder = None):
        if finder is not None and not isinstance(finder, IThresholdFinder):
            raise ValueError();

        self._treeCount = treeCount;
        self._subsamplingSize = subsamplingSize;
        self._treesList = [];

        self._scores = None;
        self._threshold = None;
        self._finder = finder;


    @property
    def scores(self):
        return self._scores;


    @property
    def threshold(self):
        return self._threshold;


    @threshold.setter
    def threshold(self, value):
        if value <= 0.5 or value >= 1:
            raise ValueError();

        self._threshold = value;


    @property
    def params(self) -> List:
        return [self._subsamplingSize, self._treesList, self._scores, self._threshold];


    @params.setter
    def params(self, value : List):
        self._subsamplingSize, self._treesList, self._scores, self._threshold = tuple(value);
        self._treeCount = len(self._treesList);


    def _calcHarmonicNumber(self, i):
        return np.log(i) + np.euler_gamma;


    def _calcAveragePathLength(self, psi):
        if psi < 2:
            return 0;

        if psi == 2:
            return 1;

        return 2 * self._calcHarmonicNumber(psi - 1) - 2 * (psi - 1) / psi;


    def _getPathLength(self, instance, node, currentLength, lengthLimit):
        if node.isLeaf() or currentLength >= lengthLimit:
            return currentLength + self._calcAveragePathLength(node.samplesCount);

        if instance[0, node.featureIndex] < node.featureValue:
            return self._getPathLength(instance, node.leftChild, currentLength + 1, lengthLimit);
        else:
            return self._getPathLength(instance, node.rightChild, currentLength + 1, lengthLimit);


    def _getAnomalyScore(self, instance, lengthLimit):
        length = 0;

        for tree in self._treesList:
            length += self._getPathLength(instance, tree, 0, lengthLimit);
        length /= self._treeCount;

        return 1 / (2 ** (length / self._calcAveragePathLength(self._subsamplingSize)));


    def _hasSameFeatureValues(self, dataSet, featureIndex):
        if dataSet.shape[0] == 0:
            return True;

        return bool(np.all(dataSet[:, featureIndex] == dataSet[0, featureIndex]));


    def _choiceFeatureIndex(self, features):
        if len(features) == 1:
            return features[0];

        return features[np.random.randint(0, len(features))];


    def _choiceFeatureValue(self, dataSet, featureIndex):
        values = dataSet[:, featureIndex];
        minValue, maxValue = values.min(), values.max();

        return minValue + (maxValue - minValue) * np.random.random();


    def _createNode(self, dataSet, features, currentHeight):
        samplesCount = dataSet.shape[0];

        if samplesCount == 0:
            return None;

        if samplesCount == 1:
            return _Node(samplesCount);

        for index in [item for item in features if self._hasSameFeatureValues(dataSet, item)]:
            features.remove(index);

        if len(features) == 0:
            return _Node(samplesCount);

        featureIndex = self._choiceFeatureIndex(features);
        featureValue = self._choiceFeatureValue(dataSet, featureIndex);

        return _Node(samplesCount, featureIndex, featureValue,
                     self._createNode(dataSet[(dataSet[:, featureIndex] < featureValue).A.flatten(), :], features[:], currentHeight + 1),
                     self._createNode(dataSet[(dataSet[:, featureIndex] >= featureValue).A.flatten(), :], features[:], currentHeight + 1));


    def _createTree(self, subSet):
        return self._createNode(subSet, list(range(0, subSet.shape[1])), 0);


    def fill(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        self._scores = None;
        self._threshold = None;

        n = dataSet.shape[0];
        with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
            self._treesList = pool.map(self._createTree, [dataSet[np.random.choice(n, self._subsamplingSize, False), :] for i in range(0, self._treeCount)]);


    def getAnomalyScore(self, instance):
        if instance is None:
            raise ValueError();

        return self._getAnomalyScore(instance, self._subsamplingSize - 1);


    def train(self, dataSet):
        if self._threshold is not None and self._scores is not None:
            return False;

        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();
        if len(self._treesList) != self._treeCount:
            raise InvalidOperationError();

        with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
            self._scores = pool.map(self.getAnomalyScore, [item for item in dataSet]);

        if self._finder is not None:
            self._threshold = self._finder.find(self._scores);

        return True;


class ProportionThresholdFinder(IThresholdFinder):
    def __init__(self, proportion):
        self.__proportion = max(0, min(1, proportion));


    def find(self, scores : List[float]):
        scores.sort(reverse = True);
        return float(np.quantile(scores, self.__proportion));


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


    def _reset(self):
        self._curves = None;
        self._leftLines = None;
        self._rightLines = None;


    def _leftOnly(self, y):
        maxValue = y.max();
        value, residual = None, None;

        for i in range(max(2, CurvesThresholdFinder.MIN_SAMPLES_NUMBER), y.shape[0]):
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

        for j in range(n - max(2, CurvesThresholdFinder.MIN_SAMPLES_NUMBER), 0, -1):
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


    def find(self, scores : List[float]):
        scale = 100;
        data = np.array(scores).reshape(-1, 1) * scale;
        indices, distances, center = KMeans(lambda X, k: np.array([X.min(), X.mean(), X.max()]).reshape(-1, 1)).clustering(data, 3, 1);
        print("anomaly score centers:{0}".format(center.T));

        checkValue = center[2, 0];
        minCheckValue = self._minCheckValue * scale;
        maxCheckValue = self._maxCheckValue * scale;
        defaultThreshold = self._defaultThreshold * scale;
        minValue = float(np.amin(data[indices == 2, :], axis = 0));
        maxValue = float(np.amax(data[indices == 2, :], axis = 0));

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
        data = np.mat(data);  # compatible for old code
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

        self._reset();

        return threshold;
