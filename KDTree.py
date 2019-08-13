import numpy as np;
import numpy.matlib as npm;

import DataHelper;


class KDTree:
    def __init__(self, dataSet, featureCount = None):
        if dataSet is None or not isinstance(dataSet, np.matrix) or dataSet.shape[0] == 0:
            raise ValueError();

        if featureCount is not None and featureCount > dataSet.shape[1]:
            raise ValueError();

        if featureCount is None:
            featureCount = dataSet.shape[1];

        self.__rootNode, self.__layerCount = self.__createNodeRecursive(0, dataSet, featureCount, None);
        # self.__rootNode, self.__layerCount = self.__createNodeLoop(dataSet, featureCount);


    def __createNodeRecursive(self, depth, dataSet, featureCount, parentNode):
        layerCount = depth + 1;
        featureIndex = depth % featureCount;
        median = DataHelper.getExistingMedian(dataSet[:, featureIndex])[0, 0];

        leftDataset = dataSet[(dataSet[:, featureIndex] < median).A.flatten(), :];
        currentDataset = dataSet[(dataSet[:, featureIndex] == median).A.flatten(), :];
        rightDataset = dataSet[(dataSet[:, featureIndex] > median).A.flatten(), :];

        currentNode = KDTree.Node(depth, currentDataset, featureIndex, median, parentNode);
        currentNode.leftChild, leftLayerCount = self.__createNodeRecursive(depth + 1, leftDataset, featureCount, currentNode) if leftDataset.shape[0] > 0 else (None, 0);
        currentNode.rightChild, rightLayerCount = self.__createNodeRecursive(depth + 1, rightDataset, featureCount, currentNode) if rightDataset.shape[0] > 0 else (None, 0);

        return currentNode, max(layerCount, leftLayerCount, rightLayerCount);


    def __createNodeLoop(self, dataSet, featureCount):
        nodeStack, setStack = [], [KDTree.__DataSetWrapper(0, dataSet)];

        while len(setStack) > 0:
            currentSet = setStack.pop();
            if currentSet.dataSet is None:
                nodeStack.append(KDTree.__NodeWrapper(currentSet.depth, None));
                continue;

            featureIndex = currentSet.depth % featureCount;
            median = DataHelper.getExistingMedian(currentSet.dataSet[:, featureIndex])[0, 0];

            leftData = currentSet.dataSet[(currentSet.dataSet[:, featureIndex] < median).A.flatten(), :];
            currentData = currentSet.dataSet[(currentSet.dataSet[:, featureIndex] == median).A.flatten(), :];
            rightData = currentSet.dataSet[(currentSet.dataSet[:, featureIndex] > median).A.flatten(), :];

            nodeStack.append(KDTree.__NodeWrapper(currentSet.depth, KDTree.Node(currentSet.depth, currentData, featureIndex, median)));
            setStack.append(KDTree.__DataSetWrapper(currentSet.depth + 1, leftData if leftData.shape[0] > 0 else None));
            setStack.append(KDTree.__DataSetWrapper(currentSet.depth + 1, rightData if rightData.shape[0] > 0 else None));

        parent, current, layerCount = None, None, 0;

        while len(nodeStack) > 0:
            current = nodeStack.pop();

            if current.depth > layerCount:
                layerCount = current.depth;

            i = len(nodeStack) - 1;
            while i >= 0:
                if nodeStack[i].depth < current.depth:
                    nodeStack[i].setChild(current.node);
                    break;

                i -= 1;


        return current.node, layerCount;


    def getRootNode(self):
        return self.__rootNode;


    def getLayerCount(self):
        return self.__layerCount;


    class Node:
        def __init__(self, depth, dataSet, featureIndex, featureValue, parentNode = None, leftChild = None, rightChild = None):
            self.depth = depth;
            self.dataSet = dataSet;
            self.featureIndex = featureIndex;
            self.featureValue = featureValue;
            self.parentNode = parentNode;
            self.leftChild = leftChild;
            self.rightChild = rightChild;


        def __repr__(self):
            return "{3} : ({0})[{1}] = {2}".format(self.dataSet.shape[0], self.featureIndex, self.featureValue, self.depth);


        def __str__(self):
            return self.__repr__();


        def isLeaf(self):
            return self.leftChild is None and self.rightChild is None;


    class __NodeWrapper:
        def __init__(self, depth, node):
            self.depth = depth;
            self.node = node;
            self.hasLeft = False;
            self.hasRight = False;


        def __repr__(self):
            return str(self.node if self.node is not None else self.depth);


        def __str__(self):
            return self.__repr__();


        def setChild(self, node):
            if not self.hasLeft or not self.hasRight:
                if not self.hasLeft:
                    self.hasLeft = True;
                    self.node.leftChild = node;
                else:
                    self.hasRight = True;
                    self.node.rightChild = node;

                if node is not None:
                    node.parentNode = self.node;


    class __DataSetWrapper:
        def __init__(self, depth, dataSet):
            self.depth = depth;
            self.dataSet = dataSet;


        def __repr__(self):
            return "{0}: {1}".format(self.depth, self.dataSet);


        def __str__(self):
            return self.__repr__();
