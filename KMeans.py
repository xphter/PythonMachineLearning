import abc;
import math;
import psutil;
import multiprocessing;
import numpy as np;
from matplotlib import pyplot as plt

import DataHelper;


class KMeans:
    @staticmethod
    def optimalClustering(X, k, count):
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

        return min(result, key = lambda item: item[1].sum());


    @staticmethod
    def optimalK(X, maxK, count, selector):
        km = KMeans();

        with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
            data = pool.starmap(KMeans._performClustering, [(km, X, k) for i in range(count) for k in range(1, maxK + 1)]);

        data = [min([item for item in data if item[2].shape[0] == k], key = lambda item: item[1].sum()) for k in range(1, maxK + 1)];

        index, distance, center = data[selector.find(km, X, data)];
        print("the optimal k is {0}, the centers are:\r\n{1}\r\n".format(center.shape[0], center));

        return index, distance, center;


    @staticmethod
    def _performClustering(km, X, k):
        return km.clustering(X, k);


    def __init__(self, initialCenterSelector = None):
        self.__initialCenterSelector = initialCenterSelector if initialCenterSelector is not None else self._randomInitialCenter;


    def _randomInitialCenter(self, X, k):
        indices = list(range(0, X.shape[0]));

        index = np.random.choice(indices, 1, False);
        center = X[index, :];
        indices.remove(index[0]);

        if k == 1:
            return center;

        for i in range(1, k):
            index = self._randomFindCenter(X, center, indices);
            center = np.vstack((center, X[index, :]));
            indices.remove(index[0]);

        print("the initial {0} centers are:\r\n{1}\r\n".format(k, center));

        return center;


    def _randomFindCenter(self, X, center, indices):
        cluster, distance = self._findCluster(X[indices, :], center);
        return np.random.choice(indices, 1, False, distance.A.flatten() / distance.sum());


    def _findCluster(self, X, center):
        result = np.mat([DataHelper.calcEuclideanDistance(X, center[i, :]).A.flatten().tolist() for i in range(0, center.shape[0])]).T;

        return result.argmin(1), result.min(1);


    def _findCenter(self, X, indices):
        return np.mat([X[(indices == i).A.flatten(), :].mean(0).A.flatten().tolist() for i in set(indices.A.flatten().tolist())]);


    def _calcClusterDiameter(self, X):
        if X.shape[0] == 0:
            return 0;

        return DataHelper.calcEuclideanDistance(X, X).max();


    def clustering(self, X, k, iterationCount = None):
        if X is None or not isinstance(X, np.matrix):
            raise ValueError();

        if k < 1:
            raise ValueError();

        count = 0;
        previousIndex, currentIndex, distance = None, None, None;
        center = self.__initialCenterSelector(X, k);

        while (iterationCount is None or iterationCount is not None and count < iterationCount) and \
                (previousIndex is None or not (currentIndex == previousIndex).all()):
            previousIndex = currentIndex;
            currentIndex, distance = self._findCluster(X, center);

            center = self._findCenter(X, currentIndex);

            count += 1;

        # diameter = np.mat([self._calcClusterDiameter(X[(currentIndex == i).A.flatten(), :]) for i in range(0, center.shape[0])]).T;

        print("the final {0} centers are:\r\n{1}\r\n".format(k, center));

        return currentIndex, distance, center;


class IOptimalKSelector(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def find(self, km, X, data):
        pass;


class ElbowMethod(IOptimalKSelector):
    def __init__(self, isShowPlot = False):
        self.__isShowPlot = bool(isShowPlot);

    def find(self, km, X, data):
        error = [item[1].sum() for item in data];

        # can't select zero index for ever
        index = np.diff(error, 2).argmax() + 1;

        if self.__isShowPlot:
            plt.figure(1, (12, 8));
            plt.get_current_fig_manager().window.maximize();
            plt.plot([item[2].shape[0] for item in data], error, "-o");
            plt.show(block=True);

        return index;


class GapStatistic(IOptimalKSelector):
    def __init__(self, count, isShowPlot = False):
        self.__count = count;
        self.__isShowPlot = bool(isShowPlot);


    def _calcMeasures(self, km, X, k, i):
        index, distance, center = km.clustering(X, k);
        return i, math.log(distance.sum());


    def _calcSK(self, wk, mu, B):
        v = np.mat(wk) - mu;

        return math.sqrt((1 + B) * (v * v.T)[0, 0]) / B;


    def find(self, km, X, data):
        n, p = X.shape;
        B = self.__count;
        minValue, maxValue = X.min(0), X.max(0);
        delta = maxValue - minValue;

        K = [item[2].shape[0] for item in data];
        WK = [math.log(item[1].sum()) for item in data];

        referenceWK = None;
        referenceX = [np.multiply(np.mat([np.random.uniform(0, 1, n) for j in range(p)]).T, delta) + minValue for b in range(B)];

        with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
            referenceWK = pool.starmap(self._calcMeasures, [(km, referenceX[b], K[i], i) for b in range(B) for i in range(len(data))]);

        mu = [np.mean([item[1] for item in referenceWK if item[0] == i]) for i in range(len(data))];

        sK = [self._calcSK([item[1] for item in referenceWK if item[0] == i], mu[i], B) for i in range(len(data))];
        print("the sK is:\r\n{0}\r\n".format(sK));

        gapK = (np.array(mu) - np.array(WK)).tolist();
        print("the Gap K is:\r\n{0}\r\n".format(gapK));

        # index = 0;
        # for i in range(len(data) - 1):
        #     if gapK[i] >= gapK[i + 1] - sK[i + 1]:
        #         index = i;
        #         break;
        index = np.argmax(gapK);

        if self.__isShowPlot:
            plt.figure(1, (12, 8));
            plt.get_current_fig_manager().window.maximize();
            plt.plot(K, WK, "-ob", label = "logWk");
            plt.plot(K, mu, "-ok", label = "(Σ logWk) / B");
            plt.plot(K, gapK, "-or", label = "Gap");
            plt.legend(loc='upper right');
            plt.show(block=True);

        return index;
