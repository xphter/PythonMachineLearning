import abc;
import math;
import multiprocessing;

import numpy as np;
import scipy.stats;
import psutil;

import DataHelper;

#
# Condition Definitions
#


class ICondition(metaclass = abc.ABCMeta):
    def __repr__(self):
        return self.__str__();


    @abc.abstractmethod
    def check(self, X):
        pass;


class TrueCondition(ICondition):
    def check(self, X):
        return np.mat([True] * X.shape[0]).T;


    def __str__(self):
        return "All data";


class SingleCondition(ICondition, metaclass = abc.ABCMeta):
    def __init__(self, j, value):
        self._j = j;
        self._value = value;


class LessThanCondition(SingleCondition):
    def check(self, X):
        return X[:, self._j] < self._value;


    def __str__(self):
        return "X[:, {0}] < {1}".format(self._j, self._value);


class LessEqualCondition(SingleCondition):
    def check(self, X):
        return X[:, self._j] <= self._value;


    def __str__(self):
        return "X[:, {0}] <= {1}".format(self._j, self._value);


class EqualCondition(SingleCondition):
    def check(self, X):
        return X[:, self._j] == self._value;


    def __str__(self):
        return "X[:, {0}] = {1}".format(self._j, self._value);


class NotEqualCondition(SingleCondition):
    def check(self, X):
        return X[:, self._j] != self._value;


    def __str__(self):
        return "X[:, {0}] != {1}".format(self._j, self._value);


class GreaterThanCondition(SingleCondition):
    def check(self, X):
        return X[:, self._j] > self._value;


    def __str__(self):
        return "X[:, {0}] > {1}".format(self._j, self._value);


class GreaterEqualCondition(SingleCondition):
    def check(self, X):
        return X[:, self._j] >= self._value;


    def __str__(self):
        return "X[:, {0}] >= {1}".format(self._j, self._value);


class AggregateCondition(ICondition, metaclass = abc.ABCMeta):
    def __init__(self, *conditions):
        if len(conditions) == 0:
            raise ValueError("conditions is empty");

        self._conditions = conditions;


class AndCondition(AggregateCondition):
    def check(self, X):
        return np.hstack(tuple([item.check(X) - 0 for item in self._conditions])).sum(1) == len(self._conditions);


    def __str__(self):
        return " and ".join(self._conditions);


class OrCondition(AggregateCondition):
    def check(self, X):
        return np.hstack(tuple([item.check(X) - 0 for item in self._conditions])).sum(1) > 1;


    def __str__(self):
        return " or ".join(self._conditions);


#
# Split Strategies
#


class IDataSplitter(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def split(self, D, types, indices):
        # return value: (index, error_changed, (conditions))
        pass;


class DataSplitterBase(IDataSplitter, metaclass = abc.ABCMeta):
    def _getDiscretePoints(self, x):
        return list(set(x.A.flatten().tolist()));


    def _getContinuousPoints(self, x):
        v = np.array(list(set(x.A.flatten().tolist())));

        return ((v[:-1] + v[1:]) / 2).flatten().tolist();


class InformationGainSplitter(DataSplitterBase):
    def _calcEntropy(self, y, multiplyBySamplesCount = False):
        entropy, n = 0, y.shape[0];
        data = y.A.flatten().tolist();
        v = np.mat([data.count(c) for c in set(data)]);

        if multiplyBySamplesCount:
            entropy = -(v * np.log2(v.T / n))[0, 0];
        else:
            v = v / n;
            entropy = -(v * np.log2(v.T))[0, 0];

        return entropy;


    def split(self, D, types, indices):
        n = D.shape[0];
        total = self._calcEntropy(D[:, -1]);

        index, value, flag, entropy = min([
            min([(j, s, True, (self._calcEntropy(D[(D[:, j] <= s).A.flatten(), -1], True) + self._calcEntropy(D[(D[:, j] > s).A.flatten(), -1], True)) / n) for s in self._getContinuousPoints(D[:, j])], key = lambda item: item[3])
            if types[j] else
            (j, self._getDiscretePoints(D[:, j]), False, sum([self._calcEntropy(D[(D[:, j] == s).A.flatten(), -1], True) for s in self._getDiscretePoints(D[:, j])]) / n)
            for j in indices],
            key = lambda item: item[3]);

        return index, entropy - total, (LessEqualCondition(index, value), GreaterThanCondition(index, value)) if flag else tuple([EqualCondition(index, s) for s in value]);


class GiniIndexSplitter(DataSplitterBase):
    def _calcGiniIndex(self, y, multiplyBySamplesCount = False):
        n = y.shape[0];
        data = y.A.flatten().tolist();

        v = np.mat([data.count(c) for c in set(data)]);

        return (n - (v * v.T)[0, 0] / n) if multiplyBySamplesCount else (1 - (v * v.T)[0, 0] / n ** 2);


    def split(self, D, types, indices):
        n = D.shape[0];
        total = self._calcGiniIndex(D[:, -1]);

        index, value, flag, gini = min([min(
            [(j, s, True, (self._calcGiniIndex(D[(D[:, j] <= s).A.flatten(), -1], True) + self._calcGiniIndex(D[(D[:, j] > s).A.flatten(), -1], True)) / n) for s in self._getContinuousPoints(D[:, j])]
            if types[j] else
            [(j, s, False, (self._calcGiniIndex(D[(D[:, j] == s).A.flatten(), -1], True) + self._calcGiniIndex(D[(D[:, j] != s).A.flatten(), -1], True)) / n) for s in self._getDiscretePoints(D[:, j])],
            key = lambda item: item[3]) for j in indices],
            key = lambda item: item[3]);

        return index, gini - total, (LessEqualCondition(index, value), GreaterThanCondition(index, value)) if flag else (EqualCondition(index, value), NotEqualCondition(index, value));


class RssDataSplitter(DataSplitterBase):
    def _calcSS(self, y):
        y = y - y.mean();

        return (y.T * y)[0, 0];


    def split(self, D, types, indices):
        tss = self._calcSS(D[:, -1]);

        index, value, flag, rss = min([min(
            [(j, s, True, self._calcSS(D[(D[:, j] <= s).A.flatten(), -1]) + self._calcSS(D[(D[:, j] > s).A.flatten(), -1])) for s in self._getContinuousPoints(D[:, j])]
            if types[j] else
            [(j, s, False, self._calcSS(D[(D[:, j] == s).A.flatten(), -1]) + self._calcSS(D[(D[:, j] != s).A.flatten(), -1])) for s in self._getDiscretePoints(D[:, j])],
            key = lambda item: item[3]) for j in indices],
            key = lambda item: item[3]);

        return index, rss - tss, (LessEqualCondition(index, value), GreaterThanCondition(index, value)) if flag else (EqualCondition(index, value), NotEqualCondition(index, value));


#
# Main Models
#


class Node(metaclass = abc.ABCMeta):
    def __init__(self, D, parent, condition):
        if D is None:
            raise ValueError("D is None");
        if condition is None:
            raise ValueError("condition is None");
        if not isinstance(condition, ICondition):
            raise ValueError("condition is not a object of ICondition");

        self._condition = condition;
        self._value = self._getValue(D) if D.shape[0] > 0 else parent.value;

        self._parentNode = parent;
        self._childNodes = [];


    def __repr__(self):
        return self.__str__();


    def __str__(self):
        return "{0}, {1}".format(self._value, self._condition);


    @abc.abstractmethod
    def _getValue(self, D):
        pass;


    @property
    def value(self):
        return self._value;


    @property
    def condition(self):
        return self._condition;


    @property
    def parentNode(self):
        return self.__parentNode;


    @property
    def childNodes(self):
        return self._childNodes;


    @property
    def isLeaf(self):
        return len(self._childNodes) == 0;


    @staticmethod
    @abc.abstractmethod
    def baggingPredict(values : list):
        pass;


    @staticmethod
    @abc.abstractmethod
    def baggingError(values: list, label):
        pass;


class RNode(Node):
    def _getValue(self, D):
        return D[:, -1].mean();


    @staticmethod
    def baggingPredict(values : list):
        return np.mean(values);


    @staticmethod
    def baggingError(values: list, label):
        return (RNode.baggingPredict(values) - label) ** 2;


class CNode(Node):
    def _getValue(self, D):
        return scipy.stats.mode(D[:, -1], axis = None)[0][0];


    @staticmethod
    def baggingPredict(values : list):
        return max([(c, values.count(c)) for c in set(values)], key = lambda item: item[1])[0];


    @staticmethod
    def baggingError(values: list, label):
        return 0 if CNode.baggingPredict(values) == label else 1;


class DecisionTree:
    def __init__(self, nodeType, splitter, name = ""):
        self.__nodeType = nodeType;
        self.__splitter = splitter;
        self.__name = name;

        self.__queue = None;
        self.__rootNode = None;
        self.__splitsCount = 0;
        self.__errorChanged = {};


    @property
    def errorChanged(self):
        return self.__errorChanged;


    def __isSame(self, X, j):
        return np.all(X[:, j] == X[0, j]);


    def __createNode(self, D, parentNode, condition):
        return self.__nodeType(D, parentNode, condition);


    def __splitNode(self, node, D, types, minDataLimit , variableCount):
        if D.shape[0] <= minDataLimit or self.__isSame(D, -1):
            return False;

        p = D.shape[1] - 1;
        columns = [j for j in (np.random.choice(p, variableCount, False).tolist() if variableCount is not None else list(range(p))) if not self.__isSame(D, j)];
        if len(columns) == 0:
            return False;

        childNode = None;
        index, errorChanged, conditions = self.__splitter.split(D, types, columns);
        self.__errorChanged[index] += errorChanged;

        print("method of split node ({0}): {1}, error changed: {2}".format(node, conditions, errorChanged));

        for d, c in [(D[c.check(D).A.flatten(), :], c) for c in conditions]:
            childNode = self.__createNode(d, node, c);
            node.childNodes.append(childNode);
            self.__queue.append((childNode, d, types, minDataLimit, variableCount));

        return True;


    def fit(self, D, types : list, minDataLimit = 5, maxSplitLimit = None, variableCount = None):
        if D is None:
            raise ValueError("D is none");

        p = D.shape[1] - 1;
        minDataLimit = max(1, minDataLimit);
        types = np.array(types) if types is not None else np.array([False] * p);

        if types.shape[0] != p:
            raise ValueError("the size of types is not match number of features");

        if variableCount is not None:
            variableCount = max(1, min(p, variableCount));

        self.__queue = [];
        self.__splitsCount = 0;
        self.__errorChanged = dict([(j, 0) for j in range(p)]);
        self.__rootNode = self.__createNode(D, None, TrueCondition());

        self.__queue.append((self.__rootNode, D, types, minDataLimit, variableCount));
        while len(self.__queue) > 0:
            if maxSplitLimit is not None and self.__splitsCount >= maxSplitLimit:
                break;

            if self.__splitNode(*self.__queue.pop(0)):
                self.__splitsCount += 1;

        print("fit decision tree {0} completed, split a total of {1} times".format(self.__name, self.__splitsCount));

        return self;


    def predict(self, x):
        if x is None:
            raise ValueError("x is none");
        if x.shape[0] != 1:
            raise ValueError("x is empty or more than one sample");
        if self.__rootNode is None:
            raise ValueError("the tree has not fitted");

        value, node = None, self.__rootNode;

        while node is not None:
            value = node.value;

            node = ([None] + [c for c in node.childNodes if c.condition.check(x)[0, 0]]).pop(-1) if not node.isLeaf else None;

        return value;


class RandomForest:
    def __init__(self, nodeType, splitter):
        self.__nodeType = nodeType;
        self.__splitter = splitter;

        self.__trees = [];
        self.__OOBError = None;
        self.__importance = None;


    @property
    def OOBError(self):
        return self.__OOBError;


    @property
    def variableImportance(self):
        return self.__importance;


    def _fitTree(self, tree, D, types, minDataLimit, variableCount):
        return tree.fit(D, types, minDataLimit, None, variableCount);


    def fit(self, D, types : list, treesCount : int, minDataLimit = 5, variableCount = None):
        if D is None:
            raise ValueError("D is none");
        if treesCount < 1:
            raise ValueError("number of trees less than one");

        n, p = D.shape[0], D.shape[1] - 1;
        if variableCount is None:
            variableCount = math.ceil(math.sqrt(p));

        self.__importance = [0] * p;
        self.__trees = [DecisionTree(self.__nodeType, self.__splitter, str(i + 1)) for i in range(treesCount)];
        bootstrapSets = [np.random.choice(n, n, True) for i in range(treesCount)];
        tasks = [(self.__trees[i], D[bootstrapSets[i], :], types, minDataLimit, variableCount) for i in range(treesCount)];

        if len(tasks) > 1:
            with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
                self.__trees = pool.starmap(self._fitTree, tasks);
        else:
            self._fitTree(*tasks[0]);

        errors, values = [], None;
        for i in range(n):
            values = [self.__trees[j].predict(D[i, :-1]) for j in [k for k in range(treesCount) if i not in bootstrapSets[k]]];

            if len(values) > 0:
                errors.append(self.__nodeType.baggingError(values, D[i, -1]));

        self.__OOBError = np.mean(errors);

        for errorChanged in [tree.errorChanged for tree in self.__trees]:
            for key, value in errorChanged.items():
                self.__importance[key] += value;
        self.__importance = (np.array(self.__importance) / treesCount).tolist();

        print("fit random forest with {0} trees completed, OOB error: {1}".format(treesCount, self.__OOBError));

        return self;


    def predict(self, x):
        if x is None:
            raise ValueError("x is none");
        if self.__trees is None:
            raise ValueError("the forest has not fitted");

        return self.__nodeType.baggingPredict([tree.predict(x) for tree in self.__trees]);
