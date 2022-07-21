import abc;
import math;
import psutil;
import multiprocessing;
import numpy as np;
from matplotlib import pyplot as plt
from typing import List, Tuple;

import DataHelper;


class IOptimalKSelector(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def find(self, km, X : np.ndarray, data : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> int:
        pass;


class KMeans:
    @staticmethod
    def optimalClustering(X : np.ndarray, k : int, count : int) -> (np.ndarray, np.ndarray, np.ndarray):
        if X is None:
            raise ValueError("X is none.");
        if k < 1:
            raise ValueError("k is less than one.");
        if count < 1:
            raise ValueError("count is less than one.");

        result = None;
        km = KMeans();

        if count == 1:
            result = [KMeans._performClustering(km, X, k)];
        else:
            with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
                result = pool.starmap(KMeans._performClustering, [(km, X, k) for i in range(count)]);

        return min(result, key = lambda item: np.sum(item[1]));


    @staticmethod
    def optimalK(X : np.ndarray, maxK : int, count : int, selector : IOptimalKSelector):
        km = KMeans();

        if maxK == 1 and count == 1:
            data = [KMeans._performClustering(km, X, 1)];
        else:
            with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
                data = pool.starmap(KMeans._performClustering, [(km, X, k) for i in range(count) for k in range(1, maxK + 1)]);

        data = [min([item for item in data if len(item[2]) == k], key = lambda item: np.sum(item[1])) for k in range(1, maxK + 1)];

        index, distance, center = data[selector.find(km, X, data)];
        print("the optimal k is {0}, the centers are:\r\n{1}\r\n".format(len(center), center));

        return index, distance, center;


    @staticmethod
    def _performClustering(km, X : np.ndarray, k : int) -> (np.ndarray, np.ndarray, np.ndarray):
        return km.clustering(X, k);


    def __init__(self, initialCenterSelector = None):
        self._initialCenterSelector = initialCenterSelector if initialCenterSelector is not None else self._randomInitialCenter;


    def _randomInitialCenter(self, X : np.ndarray, k : int) -> np.ndarray:
        indices = list(range(0, len(X)));
        center = np.empty((k,) + X.shape[1:]);

        index = np.random.choice(indices, 1, False);
        indices.remove(index[0]);
        center[0] = X[index];

        if k == 1:
            return center;

        for i in range(1, k):
            index = self._randomFindCenter(X, center[:i], indices);
            indices.remove(index[0]);
            center[i] = X[index];

        return center;


    def _randomFindCenter(self, X : np.ndarray, center : np.ndarray, indices : List[int]):
        cluster, distance = self._findCluster(X[indices], center);
        return np.random.choice(indices, 1, False, distance / np.sum(distance));


    def _findCluster(self, X : np.ndarray, center : np.ndarray) -> (np.ndarray, np.ndarray):
        distance = DataHelper.calcEuclideanDistance(X, center);

        return np.argmin(distance, 1), np.amin(distance, 1);


    def _findCenter(self, X : np.ndarray, indices : np.ndarray) -> np.ndarray:
        return np.array([X[indices == i].mean(0) for i in set(indices.tolist())]);


    def _calcClusterDiameter(self, X):
        if X.shape[0] == 0:
            return 0;

        return DataHelper.calcEuclideanDistance(X, X).max();


    # return value: (index, distance, center)
    def clustering(self, X : np.ndarray, k : int, iterationCount : int = None, verbose : bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
        if X is None:
            raise ValueError("X is None");
        if X.ndim != 2:
            raise ValueError("X is not a matrix");

        if k < 1:
            raise ValueError("k is less than 1");

        count = 0;
        previousIndex, currentIndex, distance = None, None, None;
        center = self._initialCenterSelector(X, k);

        if verbose:
            print("the initial {0} centers are:\n{1}\r\n".format(k, center));

        while (iterationCount is None or iterationCount is not None and count < iterationCount) and \
                (previousIndex is None or np.any(currentIndex != previousIndex)):
            previousIndex = currentIndex;
            currentIndex, distance = self._findCluster(X, center);

            center = self._findCenter(X, currentIndex);

            count += 1;

        # diameter = np.mat([self._calcClusterDiameter(X[(currentIndex == i).A.flatten(), :]) for i in range(0, center.shape[0])]).T;

        if verbose:
            print("the final {0} centers are:\n{1}\n".format(k, center));

        return currentIndex, distance, center;


class ElbowMethod(IOptimalKSelector):
    def __init__(self, isShowPlot : bool = False):
        self._isShowPlot = bool(isShowPlot);


    def find(self, km : KMeans, X : np.ndarray, data : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> int:
        if len(data) == 1:
            return 0;

        error = [np.sum(item[1]) for item in data];

        # can't select zero index for ever
        index = np.diff(error, 2).argmax() + 1;

        if self._isShowPlot:
            plt.figure(1, (12, 8));
            plt.get_current_fig_manager().window.showMaximized();
            plt.plot([len(item[2]) for item in data], error, "-o");
            plt.show(block=True);

        return index;


class GapStatistic(IOptimalKSelector):
    def __init__(self, count : int, isShowPlot : bool = False):
        self._count = count;
        self._isShowPlot = bool(isShowPlot);


    def _calcMeasures(self, km : KMeans, X : np.ndarray, k : int, i : int) -> (int, float):
        index, distance, center = km.clustering(X, k);
        return i, math.log(distance.sum());


    def _calcSK(self, wk : np.ndarray, mu : float, B : int):
        v = wk - mu;

        return math.sqrt((1 + B) * np.dot(v, v)) / B;


    def find(self, km : KMeans, X : np.ndarray, data : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> int:
        if len(data) == 1:
            return 0;

        n, p = X.shape;
        B = self._count;
        minValue, maxValue = X.min(0), X.max(0);
        delta = maxValue - minValue;

        K = [len(item[2]) for item in data];
        WK = np.array([math.log(np.sum(item[1])) for item in data]);

        referenceWK = None;
        referenceX = [np.array([np.random.uniform(0, 1, n) for j in range(p)]).T * delta + minValue for b in range(B)];

        with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
            referenceWK = pool.starmap(self._calcMeasures, [(km, referenceX[b], K[i], i) for b in range(B) for i in range(len(data))]);

        mu = np.array([np.mean([item[1] for item in referenceWK if item[0] == i]) for i in range(len(data))]);

        sK = [self._calcSK(np.array([item[1] for item in referenceWK if item[0] == i]), mu[i], B) for i in range(len(data))];
        print("the sK is:\r\n{0}\r\n".format(sK));

        gapK = mu - WK;
        print("the Gap K is:\r\n{0}\r\n".format(gapK));

        # index = 0;
        # for i in range(len(data) - 1):
        #     if gapK[i] >= gapK[i + 1] - sK[i + 1]:
        #         index = i;
        #         break;
        index = np.argmax(gapK);

        if self._isShowPlot:
            plt.figure(1, (12, 8));
            plt.get_current_fig_manager().window.showMaximized();
            plt.plot(K, WK, "-ob", label = "logWk");
            plt.plot(K, mu, "-ok", label = "(Σ logWk) / B");
            plt.plot(K, gapK, "-or", label = "Gap");
            plt.legend(loc='upper right');
            plt.show(block=True);

        return index;


class CombinationOptimalKSelector(IOptimalKSelector):
    def __init__(self, selectors : List[IOptimalKSelector]):
        self._selectors = selectors;


    def find(self, km : KMeans, X : np.ndarray, data : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> int:
        if len(data) == 1:
            return 0;

        return max([s.find(km, X, data) for s in self._selectors]);
