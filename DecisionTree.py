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
    def split(self, D, types):
        # return value: (index, error_changed, (conditions))
        pass;


class DataSplitterBase(IDataSplitter, metaclass = abc.ABCMeta):
    def _getDiscretePoints(self, x):
        return list(set(x.A.flatten().tolist()));

    def _getContinuousPoints(self, x):
        v = np.array(list(set(x.A.flatten().tolist())));

        return ((v[:-1] + v[1:]) / 2).A.flatten().tolist();


class RssDataSplitter(DataSplitterBase):
    def _calcRss(self, y):
        y = y - y.mean();

        return (y.T * y)[0, 0];

    def split(self, D, types):
        tss = self._calcRss(D[:, -1]);

        index, rss, value, flag = min([min(
            [(j, s, True, self._calcRss(D[(D[:, j] <= s).A.flatten(), -1]) + self._calcRss(D[(D[:, j] > s).A.flatten(), -1])) for s in self._getContinuousPoints(D[:, j])]
            if types[j] else
            [(j, s, False, self._calcRss(D[(D[:, j] == s).A.flatten(), -1]) + self._calcRss(D[(D[:, j] != s).A.flatten(), -1])) for s in self._getDiscretePoints(D[:, j])],
            key = lambda item: item[3]) for j in range(D.shape[1] - 1)],
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

        self._condition = self.condition;
        self._value = self._getValue(D) if D.shape[0] > 0 else parent.value;

        self._parentNode = parent;
        self._childNodes = [];


    def __repr__(self):
        return self.__str__();


    def __str__(self):
        return "{0} {1}".format(self._value, self._condition);


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
        return scipy.stats.mode(D[:, -1], 0)[0][0];


class DecisionTree:
    pass;
