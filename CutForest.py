import math;
import numpy as np;

from Event import *;
from Errors import *;
from KMeans import *;


class _Node:
    def __init__(self, dataSet, box = None, featureIndex = None, featureValue = None, leftChild = None, rightChild = None):
        self.dataSet = dataSet;
        self.box = box;
        self.featureIndex = featureIndex;
        self.featureValue = featureValue;

        self.parentNode = None;
        self.leftChild = leftChild;
        if leftChild is not None:
            leftChild.parentNode = self;

        self.rightChild = rightChild;
        if rightChild is not None:
            rightChild.parentNode = self;

        self.__leftCount = None;
        self.__siblingNode = None;
        self.__hasSibling = False;


    def __repr__(self):
        return self.__str__();


    def __str__(self):
        if self.isLeaf():
            return "Leaf node, {0} samples".format(self.dataSet.shape[0]);
        else:
            return "Internal node, {0} samples, feature {1} of value {2}".format(self.dataSet.shape[0], self.featureIndex, self.featureValue);


    def __calcLeafCount(self):
        if self.isLeaf():
            return 1;

        return (self.leftChild.getLeafCount() if self.leftChild is not None else 0) + \
               (self.rightChild.getLeafCount() if self.rightChild is not None else 0);


    def __calcSiblingNode(self):
        if self.parentNode is None:
            return None;

        return self.parentNode.rightChild if self == self.parentNode.leftChild else self.parentNode.leftChild;


    def isLeaf(self):
        return self.featureIndex is None;


    def getLeafCount(self):
        if self.__leftCount is None:
            self.__leftCount = self.__calcLeafCount();

        return self.__leftCount;


    def clearLeafCount(self):
        self.__leftCount = None;


    def getSiblingNode(self):
        if not self.__hasSibling:
            self.__siblingNode = self.__calcSiblingNode();
            self.__hasSibling = True;

        return self.__siblingNode;


    def clearSiblingNode(self):
        self.__hasSibling = False;


class CutForest:
    def __init__(self, treeCount = 100, subSamplingSize = 256, trainProcessChangedFrequency = 10):
        self.__treeCount = treeCount;
        self.__subSamplingSize = subSamplingSize;
        self.__treesList = [];
        self.__trainProcessChangedFrequency = trainProcessChangedFrequency;

        self.threshold = None;
        self.fillProgressChanged = Event("fillProcessChanged");
        self.trainProgressChanged = Event("trainProcessChanged");


    def __calcBox(self, dataSet):
        return np.vstack((dataSet.min(0), dataSet.max(0)));


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


    def __choiceFeatureIndex(self, features, box):
        if len(features) == 1:
            return features[0];

        allRange = (box[1, features] - box[0, features]).sum();

        return np.random.choice(features, 1, replace = False, p = [(box[1, i] - box[0, i]) / allRange for i in features])[0];


    def __choiceFeatureValue(self, featureIndex, box):
        minValue, maxValue = box[0, featureIndex], box[1, featureIndex];

        return minValue + (maxValue - minValue) * np.random.random();


    def __createNode(self, dataSet, features):
        if dataSet.shape[0] == 0:
            return None;

        box = self.__calcBox(dataSet);

        if dataSet.shape[0] == 1:
            return _Node(dataSet, box);

        for index in [item for item in features if self.__hasSameFeatureValues(dataSet, item)]:
            features.remove(index);

        if len(features) == 0:
            return _Node(dataSet, box);

        featureIndex = self.__choiceFeatureIndex(features, box);
        featureValue = self.__choiceFeatureValue(featureIndex, box);

        return _Node(dataSet, box, featureIndex, featureValue,
                     self.__createNode(dataSet[(dataSet[:, featureIndex] <= featureValue).A.flatten(), :], features[:]),
                     self.__createNode(dataSet[(dataSet[:, featureIndex] > featureValue).A.flatten(), :], features[:]));


    def __createTree(self, dataSet, subSamplingSize):
        indices = np.random.randint(0, dataSet.shape[0], subSamplingSize);
        subSet = dataSet[indices, :];

        return self.__createNode(subSet, list(range(0, dataSet.shape[1])));


    def __findCutFeature(self, node, instance):
        box = np.vstack((np.vstack((node.box[0, :], instance)).min(0), np.vstack((node.box[1, :], instance)).max(0)));
        scope = box[1, :] - box[0, :];

        allScope = scope.sum();

        if allScope == 0:
            return -1, None;

        total, featureIndex, featureValue = 0, -1, None;
        r = allScope * np.random.random();
        for i in range(0, scope.shape[1]):
            if scope[0, i] == 0:
                continue;

            total += scope[0, i];
            if total >= r:
                featureIndex = i;
                featureValue = box[0, i] + total - r;
                break;

        return featureIndex, featureValue;


    def __findCutNode(self, node, instance):
        featureIndex, featureValue = self.__findCutFeature(node, instance);

        if (featureIndex < 0) or\
                (node.box[1, featureIndex] <= featureValue) and (instance[0, featureIndex] > featureValue) or\
                (instance[0, featureIndex] <= featureValue) and (node.box[0, featureIndex] > featureValue):
            return node, featureIndex, featureValue;

        return self.__findCutNode(node.leftChild if instance[0, node.featureIndex] <= node.featureValue else node.rightChild, instance);


    def __calcAnomalyScore(self, node, instance):
        displacement, offset = [], 0;
        targetNode, featureIndex, featureValue = self.__findCutNode(node, instance);
        currentNode, siblingNode = targetNode, None;

        if featureIndex >= 0:
            offset = 1;
            displacement.append(targetNode.getLeafCount() / offset);
            siblingNode = targetNode.getSiblingNode();

            if siblingNode is not None:
                displacement.append(siblingNode.getLeafCount() / (targetNode.getLeafCount() + offset));

            currentNode = targetNode.parentNode;

        while currentNode is not None:
            siblingNode = currentNode.getSiblingNode();

            if siblingNode is not None:
                displacement.append(siblingNode.getLeafCount() / (currentNode.getLeafCount() + offset));

            currentNode = currentNode.parentNode;

        return max(displacement);


    def __onFillProgressChanged(self, count):
        return any(self.fillProgressChanged.trigger(count, self.__treeCount));


    def __onTrainProgressChanged(self, count, total):
        return any(self.trainProgressChanged.trigger(count, total));


    def fill(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        for i in range(0, self.__treeCount):
            self.__treesList.append(self.__createTree(dataSet, self.__subSamplingSize));

            if self.__onFillProgressChanged(i + 1):
                return False;

        return True;


    def getAnomalyScore(self, instance):
        if instance is None or not isinstance(instance, np.matrix):
            raise ValueError();

        score = 0;

        for i in range(0, self.__treeCount):
            score += self.__calcAnomalyScore(self.__treesList[i], instance);

        return score / self.__treeCount;


    def train(self, dataSet, scores = None):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();
        if len(self.__treesList) != self.__treeCount:
            raise InvalidOperationError();

        if scores is None:
            scores = [];

        requestStop = False;
        totalCount = dataSet.shape[0];

        for i in range(0, totalCount):
            scores.append(self.getAnomalyScore(dataSet[i, :]));

            if (i + 1) % self.__trainProcessChangedFrequency == 0 and self.__onTrainProgressChanged(i + 1, totalCount):
                requestStop = True;
                break;

        if not requestStop:
            indices, distances, center = KMeans(lambda X, k: np.mat([X.min(), X.max()]).T).clustering(np.mat(scores).T, 2, 1);
            self.threshold = center[1, 0];

        return scores;
