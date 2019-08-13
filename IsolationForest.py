import math;
import numpy as np;


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
    def __init__(self, treeCount = 100, subSamplingSize = 256):
        self.__treeCount = treeCount;
        self.__subSamplingSize = subSamplingSize;
        self.__treesList = [];


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

        if instance[0, node.featureIndex] <= node.featureValue:
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

        return features[np.random.random_integers(0, len(features) - 1)];


    def __choiceFeatureValue(self, dataSet, featureIndex):
        values = dataSet[:, featureIndex];
        minValue, maxValue = values.min(), values.max();

        return minValue + (maxValue - minValue) * np.random.random();


    def __createNode(self, dataSet, features, currentHeight, heightLimit):
        samplesCount = dataSet.shape[0];

        if samplesCount == 0:
            return None;

        if samplesCount == 1 or currentHeight >= heightLimit:
            return _Node(samplesCount);

        for index in [item for item in features if self.__hasSameFeatureValues(dataSet, item)]:
            features.remove(index);

        if len(features) == 0:
            return _Node(samplesCount);

        featureIndex = self.__choiceFeatureIndex(features);
        featureValue = self.__choiceFeatureValue(dataSet, featureIndex);

        return _Node(samplesCount, featureIndex, featureValue,
                     self.__createNode(dataSet[(dataSet[:, featureIndex] <= featureValue).A.flatten(), :], features[:], currentHeight + 1, heightLimit),
                     self.__createNode(dataSet[(dataSet[:, featureIndex] > featureValue).A.flatten(), :], features[:], currentHeight + 1, heightLimit));


    def __createTree(self, dataSet, subSamplingSize, heightLimit):
        indices = np.random.random_integers(0, dataSet.shape[0] - 1, subSamplingSize);
        subSet = dataSet[indices, :];

        return self.__createNode(subSet, list(range(0, dataSet.shape[1])), 0, heightLimit);


    def train(self, dataSet, heightLimit = None):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        if heightLimit is None:
            heightLimit = self.__subSamplingSize;

        for i in range(0, self.__treeCount):
            self.__treesList.append(self.__createTree(dataSet, self.__subSamplingSize, heightLimit));


    def getAnomalyScore(self, instance):
        if instance is None:
            raise ValueError();

        return self.__getAnomalyScore(instance, self.__subSamplingSize);
