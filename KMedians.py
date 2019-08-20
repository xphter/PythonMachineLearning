import numpy as np;

import DataHelper;


class KMedians:
    def __init__(self, initialCenterSelector = None):
        self.__initialCenterSelector = initialCenterSelector if initialCenterSelector is not None else (
            lambda X, k: self.__randomInitialCenter(X, k));


    def __randomInitialCenter(self, dataSet, k):
        index = np.random.randint(0, dataSet.shape[0], 1);
        center = dataSet[index, :];

        if k == 1:
            return center;

        for i in range(1, k):
            dataSet = np.delete(dataSet, index, 0);
            index = self.__randomFindCenter(dataSet, center);
            center = np.vstack((center, dataSet[index, :]));

        return center;


    def __randomFindCenter(self, dataSet, center):
        cluster, distance = self.__findCluster(dataSet, center);
        return DataHelper.choiceProportional(distance);


    def __findCluster(self, dataSet, center):
        result = None;

        for i in range(0, center.shape[0]):
            distance = DataHelper.calcManhattanDistance(dataSet, center[i, :]);

            result = distance if result is None else np.hstack((result, distance));

        return result.argmin(1), result.min(1);


    def __findCenter(self, dataSet, indices):
        result = None;

        for i in set(np.sort(indices, None).A.flatten()):
            center = np.median(dataSet[(indices == i).A.flatten(), :], 0);

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
