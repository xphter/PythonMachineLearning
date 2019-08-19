import numpy as np;

import DataHelper;


class KMeans:
    def __init__(self, initialCenterSelector = None):
        self.__initialCenterSelector = initialCenterSelector if initialCenterSelector is not None else (
            lambda X, k: self.__randomInitialCenter(X, k));


    def __randomInitialCenter(self, dataSet, k):
        return dataSet[np.random.random_integers(0, dataSet.shape[0] - 1, k), :];


    def __findCluster(self, dataSet, center):
        result = None;

        for i in range(0, center.shape[0]):
            distance = DataHelper.calcEuclideanDistance(dataSet, center[i, :]);

            result = distance if result is None else np.hstack((result, distance));

        return result.argmin(1), result.min(1);


    def __findCenter(self, dataSet, indices):
        result = None;

        for i in set(np.sort(indices, None).A.flatten()):
            center = dataSet[(indices == i).A.flatten(), :].mean(0);

            result = center if result is None else np.vstack((result, center));

        return result;


    def clustering(self, dataSet, k, iterationCount = None):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        if k < 1:
            raise ValueError();

        count = 0;
        previousIndex, currentIndex, distance = None, None, None;
        center = self.__initialCenterSelector(dataSet, k);

        while (iterationCount is None or iterationCount is not None and count < iterationCount) and \
                (previousIndex is None or not (currentIndex == previousIndex).all()):
            previousIndex = currentIndex;
            currentIndex, distance = self.__findCluster(dataSet, center);

            center = self.__findCenter(dataSet, currentIndex);

            count += 1;

        return currentIndex, distance, center;
