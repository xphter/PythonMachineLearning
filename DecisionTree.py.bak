import math;
import pickle;
import matplotlib.pyplot as plt;
from collections import deque;


#
# helper classes
#


class Condition:
    def __init__(self, name, checker, value):
        self.__name = name;
        self.__checker = checker;
        self.__value = value;

    def __str__(self):
        return "{0} {1}".format(self.__name, self.__value);

    def __repr__(self):
        return self.__str__();


    def check(self, target):
        return self.__checker(target, self.__value);

    @staticmethod
    def equal(value):
        return Condition("==", lambda x, y: x == y, value);

    @staticmethod
    def notEqual(value):
        return Condition("!=", lambda x, y: x != y, value);

    @staticmethod
    def lessEqual(value):
        return Condition("<=", lambda x, y: x <= y, value);

    @staticmethod
    def greaterThan(value):
        return Condition(">", lambda x, y: x > y, value);


class Node:
    def __init__(self, dataSet, level, feature = None):
        self.dataSet = dataSet;
        self.level = level;
        self.probability = Node.__calcProbability(dataSet);
        self.category = (sorted(self.probability, key = lambda x: self.probability[x], reverse = True)[0]) if len(self.probability) > 0 else None;
        self.feature = feature;
        self.__isLeaf = feature is None;
        self.childNodes = {};

    def __str__(self):
        return "category: {0}, feature: {1}, probability: {2}".format(self.category, self.feature, self.probability);


    def __repr__(self):
        return self.__str__();


    @staticmethod
    def __calcProbability(matrix):
        labels = {};
        value = None;
        count = len(matrix);

        if count == 0:
            return labels;

        for row in matrix:
            value = row[-1];

            if value not in labels:
                labels[value] = 1;
            else:
                labels[value] += 1;

        return dict([(k, v / count) for k, v in labels.items()]);


    def isLeaf(self):
        return self.__isLeaf;


    def setLeaf(self, isLeaf):
        self.__isLeaf = bool(isLeaf);


    def toLeaf(self):
        self.__isLeaf = True;
        self.feature = None;
        self.childNodes.clear();


    def getLeafCount(self):
        if self.isLeaf():
            return 1;

        return sum([self.childNodes[key].getLeafCount() for key in self.childNodes]);


    def getDepth(self):
        if self.isLeaf():
            return 1;

        return 1 + max([self.childNodes[key].getDepth() for key in self.childNodes]);


#
# helper methods
#


def __calcEntropy(vector):
    labels = {};
    count = len(vector);

    if count == 0:
        return 0;

    for item in vector:
        if item not in labels:
            labels[item] = 1;
        else:
            labels[item] += 1;

    p = 0;
    result = 0;

    for item in labels.values():
        if item == 0 or item == count:
            continue;

        p = item / count;
        result -= p * math.log(p, 2);

    return result;


def __calcGini(vector):
    labels = {};
    count = len(vector);

    if count == 0:
        return 0;

    for item in vector:
        if item not in labels:
            labels[item] = 1;
        else:
            labels[item] += 1;

    result = 1;

    for item in labels.values():
        result -= (item / count) ** 2;

    return result;


def __isSameClass(matrix):
    if len(matrix) == 0:
        return True;

    label = matrix[0][-1];

    for row in matrix:
        if row[-1] != label:
            return False;

    return True;


def __isSaveValue(matrix, feature):
    if len(matrix) == 0:
        return True;

    result = True;
    value = matrix[0][feature];

    for row in matrix:
        if row[feature] != value:
            result = False;
            break;

    return result;


def __getSplitPoints(matrix, feature, isContinuous) :
    result = [];

    if len(matrix) == 0:
        return result;

    if isContinuous:
        values = sorted(set([row[feature] for row in matrix]));
        if len(values) > 1:
            result = [(values[i] + values[i + 1]) / 2 for i in range(0, len(values) - 1)];
    else:
        result = set([row[feature] for row in matrix]);

    return result;

#
# C4.5
#


def __calcContinuousGain(hd, matrix, feature, point):
    values = [row[feature] for row in matrix];
    count = len(values);

    if point < min(values) or point > max(values):
        raise ValueError();

    if count == 0:
        return 0, 0;

    entropy = 0;
    left = [row[-1] for row in matrix if row[feature] <= point];
    right = [row[-1] for row in matrix if row[feature] > point];
    hda = __calcEntropy([0] * len(left) + [1] * len(right));

    if hda == 0:
        return 0, 0;

    entropy += len(left) / count * __calcEntropy(left);
    entropy += len(right) / count * __calcEntropy(right);

    return hd - entropy,  (hd - entropy) / hda;


def __calcDiscreteGain(hd, matrix, feature):
    value = None;
    partitions = {};
    count = len(matrix);

    if count < 2:
        return 0, 0;

    hda = __calcEntropy([row[feature] for row in matrix]);
    if hda == 0:
        return 0, 0;

    for row in matrix:
        value = row[feature];

        if value not in partitions:
            partitions[value] = [row[-1]];
        else:
            partitions[value].append(row[-1]);

    entropy = 0;

    for vector in partitions.values():
        entropy += len(vector) / count * __calcEntropy(vector);

    return hd - entropy, (hd - entropy) / hda;


def __calcGain(hd, matrix, feature, isContinuous):
    if not isContinuous:
        return __calcDiscreteGain(hd, matrix, feature) + (None,);

    values = sorted(set([row[feature] for row in matrix]));
    if len(values) < 2:
        return 0, 0, None;

    values = [(values[i] + values[i + 1]) / 2 for i in range(0, len(values) - 1)];
    maxGain, maxRate, gain, rate, point = 0, 0, 0, 0, 0;

    for item in values:
        gain, rate = __calcContinuousGain(hd, matrix, feature, item);

        if rate > maxRate:
            point = item;
            maxGain = gain;
            maxRate = rate;

    return maxGain, maxRate, point;


# choose maximum rate greater than average gain.
def __chooseBeastGainFeature(matrix, features, types):
    if len(matrix) == 0:
        return -1, None;
    if len(features) == 1 and not types[features[0]]:
        return features[0], None;

    beastFeature, beastValue = None, None;
    hd = __calcEntropy([row[-1] for row in matrix]);

    allGains = [(index,) + __calcGain(hd, matrix, index, types[index]) for index in features];
    averageGain = sum([item[1] for item in allGains]) / len(allGains);

    beastFeature, gain, rate, beastValue = sorted([item for item in allGains if item[1] >= averageGain], key = lambda x: x[2], reverse = True)[0];

    return beastFeature, beastValue;


def __splitByDiscrete(matrix, feature):
    result = {};
    value = None;

    for row in matrix:
        value = row[feature];

        if value not in result:
            result[value] = [row];
        else:
            result[value].append(row);

    return dict([(Condition.equal(k), v) for k, v in result.items()]);


def __splitByContinuous(matrix, feature, value):
    return {Condition.lessEqual(value): [row for row in matrix if row[feature] <= value], Condition.greaterThan(value): [row for row in matrix if row[feature] > value]};


def __splitDataset(matrix, feature, value):
    if value is None:
        return __splitByDiscrete(matrix, feature);
    else:
        return __splitByContinuous(matrix, feature, value);


def __createTreeNode(level, matrix, features, types):
    if __isSameClass(matrix) or len(features) == 0:
        return Node(matrix, level);

    beastFeature, beastValue = __chooseBeastGainFeature(matrix, features, types);
    if beastFeature < 0:
        return Node(matrix, level);

    subSet = __splitDataset(matrix, beastFeature, beastValue);
    if len(subSet) == 1:
        return Node(matrix, level);

    parentNode = Node(matrix, level, beastFeature);
    if not types[beastFeature]:
        features.remove(beastFeature);

    for value in subSet:
        parentNode.childNodes[value] = __createTreeNode(level + 1, subSet[value], features[:], types);

    return parentNode;


def __calcCostValue(tree, alpha):
    node = None;
    costValue = 0;
    queue = deque([tree]);

    while len(queue) > 0:
        node = queue.popleft();

        if node.isLeaf():
            costValue += len(node.dataSet) * __calcEntropy([row[-1] for row in node.dataSet]);
        else:
            for item in node.childNodes.values():
                queue.append(item);

    return costValue + alpha * tree.getLeafCount();


def __pruningBranch(tree, level, alpha):
    node = None;
    queue = deque([tree]);
    hasPruning, beforeCostValue, afterCostValue = False, 0, 0;

    while len(queue) > 0:
        node = queue.popleft();

        if node.level < level:
            for item in node.childNodes.values():
                queue.append(item);
        elif node.level == level and not node.isLeaf() and all([item.isLeaf() for item in node.childNodes.values()]):
            beforeCostValue = __calcCostValue(tree, alpha);
            node.setLeaf(True);
            afterCostValue = __calcCostValue(tree, alpha);

            if afterCostValue <= beforeCostValue:
                node.toLeaf();
                hasPruning = True;
            else:
                node.setLeaf(False);

    return hasPruning;


def trainC45(matrix, types):
    if matrix is None:
        raise ValueError();

    return __createTreeNode(1, matrix, list(range(len(matrix[0]) - 1)), types);


def pruningTree(tree, alpha):
    if tree is None or not isinstance(tree, Node):
        raise ValueError();

    for level in reversed(range(1, tree.getDepth())):
        if not __pruningBranch(tree, level, alpha):
            break;


#
# CART
#


def __calcContinuousGini(matrix, feature, point):
    values = [row[feature] for row in matrix];
    count = len(values);

    if point < min(values) or point > max(values):
        raise ValueError();

    if count == 0:
        return 0;

    left = [row[-1] for row in matrix if row[feature] <= point];
    right = [row[-1] for row in matrix if row[feature] > point];

    return len(left) / count * __calcGini(left) + len(right) / count * __calcGini(right);


def __calcDiscreteGini(matrix, feature, value):
    values = [row[feature] for row in matrix];
    count = len(values);

    if value not in values:
        raise ValueError();

    if count == 0:
        return 0;

    left = [row[-1] for row in matrix if row[feature] == value];
    right = [row[-1] for row in matrix if row[feature] != value];

    return len(left) / count * __calcGini(left) + len(right) / count * __calcGini(right);


def __chooseBeastGiniFeature(matrix, features, types):
    beastFeature, beastValue = -1, None;

    if len(matrix) == 0 or len(features) == 0:
        return beastFeature, beastValue;

    giniData = [];

    for index in features:
        if types[index]:
            giniData.extend([(__calcContinuousGini(matrix, index, value), index, value) for value in __getSplitPoints(matrix, index, True)]);
        else:
            giniData.extend([(__calcDiscreteGini(matrix, index, value), index, value) for value in __getSplitPoints(matrix, index, False)]);

    if len(giniData) > 0:
        minGini, beastFeature, beastValue = sorted(giniData, key = lambda x: x[0])[0];

    return beastFeature, beastValue;


def __binarySplitDataset(matrix, feature, value, isContinuous):
    if not isContinuous:
        return {Condition.equal(value): [row for row in matrix if row[feature] == value], Condition.notEqual(value): [row for row in matrix if row[feature] != value]};
    else:
        return {Condition.lessEqual(value): [row for row in matrix if row[feature] <= value], Condition.greaterThan(value): [row for row in matrix if row[feature] > value]};


def __createBinaryTreeNode(level, matrix, features, types):
    if __isSameClass(matrix):
        return Node(matrix, level);

    for index in [item for item in features if __isSaveValue(matrix, item)]:
        features.remove(index);

    if len(features) == 0:
        return Node(matrix, level);

    beastFeature, beastValue = __chooseBeastGiniFeature(matrix, features, types);
    if beastFeature < 0:
        return Node(matrix, level);

    subSet = __binarySplitDataset(matrix, beastFeature, beastValue, types[beastFeature]);
    if len(subSet) == 1:
        return Node(matrix, level);

    parentNode = Node(matrix, level, beastFeature);

    for value in subSet:
        parentNode.childNodes[value] = __createBinaryTreeNode(level + 1, subSet[value], features[:], types);

    return parentNode;


def trainCART(matrix, types, features = None):
    if matrix is None:
        raise ValueError();

    if features is None:
        features = list(range(len(matrix[0]) - 1));

    return __createBinaryTreeNode(1, matrix, features, types);


#
# plot tree
#


__decisionNodeFormat = dict(boxstyle="sawtooth", fc="0.8");
__leafNodeFormat = dict(boxstyle="round4", fc="0.8");
__arrowFormat = dict(arrowstyle="<-");


def __plotNode(axes, nodeTxt, centerPt, parentPt, nodeFormat):
    axes.annotate(nodeTxt, xy = parentPt, xycoords = 'axes fraction',
                  xytext = centerPt, textcoords = 'axes fraction',
                  va = "center", ha = "center", bbox = nodeFormat, arrowprops = __arrowFormat);


def __plotMidText(axes, centerPt, parentPt, txtString):
    xMid = (parentPt[0] - centerPt[0]) / 2.0 + centerPt[0];
    yMid = (parentPt[1] - centerPt[1]) / 2.0 + centerPt[1];
    axes.text(xMid, yMid, txtString, va = "center", ha = "center", rotation = 30);


def __plotTree(tree, axes, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = tree.getLeafCount();
    depth = tree.getDepth();
    firstStr = str(tree.feature) + "?";
    centerPt = (__plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / __plotTree.totalW, __plotTree.yOff)
    __plotMidText(axes, centerPt, parentPt, nodeTxt);
    __plotNode(axes, firstStr, centerPt, parentPt, __decisionNodeFormat);
    secondDict = tree.childNodes;
    __plotTree.yOff = __plotTree.yOff - 1.0 / __plotTree.totalD;
    for key in secondDict:
        if not secondDict[key].isLeaf():
            __plotTree(secondDict[key], axes, centerPt, str(key));
        else:
            __plotTree.xOff = __plotTree.xOff + 1.0 / __plotTree.totalW;
            __plotNode(axes, secondDict[key].category, (__plotTree.xOff, __plotTree.yOff), centerPt, __leafNodeFormat);
            __plotMidText(axes, (__plotTree.xOff, __plotTree.yOff), centerPt, str(key));
    __plotTree.yOff = __plotTree.yOff + 1.0 / __plotTree.totalD;


def showTree(tree):
    if tree is None or not isinstance(tree, Node):
        raise ValueError();

    fig = plt.figure(1, facecolor = 'white');
    fig.clf();
    ax1 = plt.subplot(111, frameon=False);
    __plotTree.totalW = tree.getLeafCount();
    __plotTree.totalD = tree.getDepth();
    __plotTree.xOff = -0.5 / __plotTree.totalW;
    __plotTree.yOff = 1.0;
    __plotTree(tree, ax1, (0.5, 1.0), '');
    plt.show();


#
# save, load and predict
#


def saveTree(tree, filePath):
    if tree is None or not isinstance(tree, Node):
        raise ValueError();

    with open(filePath, "wb") as stream:
        pickle.dump(tree, stream);


def loadTree(filePath):
    with open(filePath, "rb") as stream:
        return pickle.load(stream);


def classifySingle(tree, vector):
    if tree is None or not isinstance(tree, Node):
        raise ValueError();

    if tree.isLeaf():
        return tree.category;

    value = vector[tree.feature];
    keys = [k for k in tree.childNodes.keys() if k.check(value)];
    if len(keys) == 0:
        raise KeyError();

    return classifySingle(tree.childNodes[keys[0]], vector);


def classifyForest(forest, vector):
    results = [classifySingle(tree, vector) for tree in forest];

    return sorted([(item, results.count(item)) for item in set(results)], key = lambda x: x[1], reverse = True)[0][0];