import abc;
import math;

import numpy as np;
import scipy.stats;

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
        samplesCount = y.shape[0];
        data = y.A.flatten().tolist();

        v = [data.count(c) for c in list(set(data))];

        if multiplyBySamplesCount:
            v = [c * math.log2(c / samplesCount) if c > 0 else 0 for c in v];
        else:
            v = [c / samplesCount * math.log2(c / samplesCount) if c > 0 else 0 for c in v];

        return -1 * sum(v);


    def split(self, D, types, indices):
        samplesCount = D.shape[0];
        total = self._calcEntropy(D[:, -1]);

        index, value, flag, entropy = min([
            min([(j, s, True, (self._calcEntropy(D[(D[:, j] <= s).A.flatten(), -1], True) + self._calcEntropy(D[(D[:, j] > s).A.flatten(), -1], True)) / samplesCount) for s in self._getContinuousPoints(D[:, j])], key = lambda item: item[3])
            if types[j] else
            (j, self._getDiscretePoints(D[:, j]), False, sum([self._calcEntropy(D[(D[:, j] == s).A.flatten(), -1], True) for s in self._getDiscretePoints(D[:, j])]) / samplesCount)
            for j in indices],
            key = lambda item: item[3]);

        return index, entropy - total, (LessEqualCondition(index, value), GreaterThanCondition(index, value)) if flag else tuple([EqualCondition(index, s) for s in value]);


class GiniIndexSplitter(DataSplitterBase):
    def _calcGiniIndex(self, y, multiplyBySamplesCount = False):
        samplesCount = y.shape[0];
        data = y.A.flatten().tolist();

        v = np.mat([data.count(c) for c in list(set(data))]);

        return (samplesCount - (v * v.T)[0, 0] / samplesCount) if multiplyBySamplesCount else (1 - (v * v.T)[0, 0] / samplesCount ** 2);


    def split(self, D, types, indices):
        samplesCount = D.shape[0];
        total = self._calcGiniIndex(D[:, -1]);

        index, value, flag, gini = min([min(
            [(j, s, True, (self._calcGiniIndex(D[(D[:, j] <= s).A.flatten(), -1], True) + self._calcGiniIndex(D[(D[:, j] > s).A.flatten(), -1], True)) / samplesCount) for s in self._getContinuousPoints(D[:, j])]
            if types[j] else
            [(j, s, False, (self._calcGiniIndex(D[(D[:, j] == s).A.flatten(), -1], True) + self._calcGiniIndex(D[(D[:, j] != s).A.flatten(), -1], True)) / samplesCount) for s in self._getDiscretePoints(D[:, j])],
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


class RNode(Node):
    def _getValue(self, D):
        return D[:, -1].mean();


class CNode(Node):
    def _getValue(self, D):
        return scipy.stats.mode(D[:, -1], axis = None)[0][0];


class DecisionTree:
    def __init__(self, nodeType, splitter):
        self.__nodeType = nodeType;
        self.__splitter = splitter;

        self.__queue = None;
        self.__rootNode = None;
        self.__splitsCount = 0;
        self.__errorChanged = {};

    def __createNode(self, D, parentNode, condition):
        return self.__nodeType(D, parentNode, condition);


    def __splitNode(self, node, D, types, minDataLimit , variableCount):
        if D.shape[0] <= minDataLimit or D[:, -1].var() == 0:
            return;

        featureCount = D.shape[1] - 1;
        columns = set(np.random.randint(0, featureCount, variableCount).tolist()) if variableCount is not None else set(range(featureCount));
        columns = columns.difference(set(np.flatnonzero(D[:, list(columns)].var(0) == 0).tolist()));
        if len(columns) == 0:
            return;

        childNode = None;
        index, errorChanged, conditions = self.__splitter.split(D, types, list(columns));
        self.__errorChanged[index] += errorChanged;

        print("method of split node ({0}): {1}, error changed: {2}".format(node, conditions, errorChanged));

        for d, c in [(D[c.check(D).A.flatten(), :], c) for c in conditions]:
            childNode = self.__createNode(d, node, c);
            node.childNodes.append(childNode);
            self.__queue.append((childNode, d, types, minDataLimit, variableCount));


    def fit(self, D, types : list, minDataLimit = 5, maxSplitLimit = None, variableCount = None):
        if D is None:
            raise ValueError("D is none");

        featureCount = D.shape[1] - 1;
        minDataLimit = max(1, minDataLimit);
        types = np.array(types) if types is not None else np.array([False] * featureCount);

        if types.shape[0] != featureCount:
            raise ValueError("the size of types is not match number of features");

        if variableCount is not None:
            variableCount = max(1, min(featureCount, variableCount));

        self.__queue = [];
        self.__splitsCount = 0;
        self.__errorChanged = dict([(j, 0) for j in range(featureCount)]);
        self.__rootNode = self.__createNode(D, None, TrueCondition());

        self.__queue.append((self.__rootNode, D, types, minDataLimit, variableCount));
        while len(self.__queue) > 0:
            if maxSplitLimit is not None and self.__splitsCount >= maxSplitLimit:
                break;

            self.__splitNode(*self.__queue.pop(0));
            self.__splitsCount += 1;

        print("fit decision tree completed, split a total of {0} times".format(self.__splitsCount));
