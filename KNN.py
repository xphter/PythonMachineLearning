import math;
import numpy as np;

import DataHelper;
import KDTree;


class KNN:
    def __init__(self, dataSet, featureCount = None, distanceCalculator = None, isNormalizeFeatures = True):
        if featureCount is not None and featureCount > dataSet.shape[1]:
            raise ValueError();

        if featureCount is None:
            featureCount = dataSet.shape[1];

        if distanceCalculator is None:
            distanceCalculator = (lambda X, v: DataHelper.calcMinkowskiDistance(X, v));

        if isNormalizeFeatures:
            if featureCount < dataSet.shape[1]:
                self.__dataSet = dataSet[:, 0:featureCount];
                self.__dataSet, self.__mu, self.__sigma = DataHelper.normalizeFeatures(self.__dataSet);
                self.__dataSet = np.hstack((self.__dataSet, dataSet[:, featureCount:]));
            else:
                self.__dataSet, self.__mu, self.__sigma = DataHelper.normalizeFeatures(dataSet);
        else:
            self.__dataSet, self.__mu, self.__sigma = dataSet, 0, 1;

        self.__featureCount = featureCount;
        self.__distanceCalculator = distanceCalculator;
        self.__isNormalizeFeatures = isNormalizeFeatures;
        self.__tree = KDTree.KDTree(self.__dataSet, featureCount);


    def __findNearestNode(self, sample, currentNode):
        currentValue = currentNode.featureValue;
        sampleValue = sample[0, currentNode.featureIndex];

        if sampleValue < currentValue and currentNode.leftChild is not None:
            return self.__findNearestNode(sample, currentNode.leftChild);
        elif sampleValue > currentValue and currentNode.rightChild is not None:
            return self.__findNearestNode(sample, currentNode.rightChild);
        else:
            return currentNode;


    def __calcDistance(self, dataSet, sample):
        if self.__featureCount < dataSet.shape[1]:
            dataSet = dataSet[:, 0:self.__featureCount];
            sample = sample[:, 0:self.__featureCount];

        return self.__distanceCalculator(dataSet, sample);


    def __joinNeighbors(self, sample, k, neighbors, distances, currentNode):
        neighbors.extend(currentNode.dataSet.A);
        distances.extend(self.__calcDistance(currentNode.dataSet, sample).A.flatten());

        if len(neighbors) > k:
            distanceIndices = np.argsort(distances)[k:];

            for i in distanceIndices:
                neighbors[i] = None;

            index = 0;
            while index < len(neighbors):
                if neighbors[index] is None:
                    neighbors.pop(index);
                    distances.pop(index);
                else:
                    index += 1;


    def __findKNN(self, sample, k, neighbors, distances, nodeStack, currentNode):
        nodeStack.append(currentNode);

        sampleValue = sample[0, currentNode.featureIndex];
        maxDistance = max(distances) if len(distances) > 0 else 0;
        isCross = len(neighbors) < k or math.fabs(sampleValue - currentNode.featureValue) <= maxDistance;

        if isCross:
            self.__joinNeighbors(sample, k, neighbors, distances, currentNode);

        # whether never visit this subtree
        isBacktracking = (currentNode.leftChild not in nodeStack) and (currentNode.rightChild not in nodeStack);

        if currentNode.leftChild is not None and currentNode.leftChild not in nodeStack:
            if len(neighbors) < k or isBacktracking or not(currentNode.rightChild in nodeStack and not isCross):
                self.__findKNN(sample, k, neighbors, distances, nodeStack, currentNode.leftChild);

        if currentNode.rightChild is not None and currentNode.rightChild not in nodeStack:
            if len(neighbors) < k or isBacktracking or not(currentNode.leftChild in nodeStack and not isCross):
                self.__findKNN(sample, k, neighbors, distances, nodeStack, currentNode.rightChild);

        if currentNode.parentNode is not None and currentNode.parentNode not in nodeStack:
            self.__findKNN(sample, k, neighbors, distances, nodeStack, currentNode.parentNode);


    def findKNN(self, sample, k):
        if sample is None:
            raise ValueError();

        neighbors, distances = [], [];

        if self.__isNormalizeFeatures:
            if self.__featureCount < sample.shape[1]:
                sample = np.hstack(
                    ((sample[:, 0:self.__featureCount] - self.__mu) / self.__sigma, sample[:, self.__featureCount:]));
            else:
                sample = (sample - self.__mu) / self.__sigma;

        self.__findKNN(sample, k, neighbors, distances, [], self.__findNearestNode(sample, self.__tree.getRootNode()));

        return np.mat(neighbors), np.mat(distances);


    def getMostFrequentlyClass(self, neighbors):
        if neighbors is None or neighbors.shape[0] == 0:
            raise ValueError();

        if neighbors.shape[0] == 1:
            return neighbors[0, -1];

        values, value = {}, None;

        for row in neighbors:
            value = row[0, -1];

            if value in values.keys():
                values[value] += 1;
            else:
                values[value] = 1;

        if len(values) == 1:
            return value;

        keys = list(values.keys());
        keys.sort(key = lambda item: values[item], reverse = True);
        return keys[0];
