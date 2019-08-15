import numpy as np;
import numpy.matlib as npm;

import DataHelper;


# consume class is 0, 1, ..., discrete feature values is 0, 1, 2, ...
class NaiveBayes:
    def __init__(self, smoothingFactor):
        self.__smoothingFactor = smoothingFactor;
        self.__discreteFeatureIndices = None;
        self.__discreteFeatureValueNumbers = None;
        self.__continuousFeatureIndices = None;

        self.__classProbability = None;
        self.__discreteFeatureProbability = None;
        self.__continuousFeatureArguments = None;


    def __calcDiscreteProbability(self, dataSet, featureValueNumbers):
        if dataSet is None:
            return np.log(np.mat(np.ones((featureValueNumbers.max(), featureValueNumbers.shape[1]))) / featureValueNumbers);

        frequency = None;
        count = dataSet.shape[0];
        result = np.mat(np.zeros((featureValueNumbers.max(), dataSet.shape[1])));

        for i in range(0, result.shape[1]):
            frequency = DataHelper.statisticFrequency(dataSet[:, i]);
            result[:, i] = np.mat([np.log(((frequency[key] if key in frequency else 0) + self.__smoothingFactor) / (count + featureValueNumbers[0, i] * self.__smoothingFactor)) if key < featureValueNumbers[0, i] else np.nan for key in range(0, result.shape[0])]).T;

        return result;


    def __calcContinuousArguments(self, dataSet, featureCount):
        return np.vstack((dataSet.mean(axis = 0), dataSet.std(axis = 0))) if dataSet is not None else np.mat(np.zeros((2, featureCount)));


    def train(self, dataSet, featureValueNumbers):
        if dataSet is None or not isinstance(dataSet, np.matrix) or featureValueNumbers is None or not isinstance(featureValueNumbers, np.matrix):
            raise ValueError();

        self.__discreteFeatureIndices = np.where(featureValueNumbers.A.flatten() > 0)[0];
        self.__continuousFeatureIndices = np.where(featureValueNumbers.A.flatten() <= 0)[0];

        if len(self.__discreteFeatureIndices) > 0:
            self.__discreteFeatureValueNumbers = featureValueNumbers[np.where(featureValueNumbers > 0)];

        classSets = DataHelper.groupBy(dataSet, -1);
        classCount = int(max(classSets.keys())) + 1;
        self.__classProbability = np.mat([np.log(((classSets[key].shape[0] if key in classSets else 0) + self.__smoothingFactor) / (dataSet.shape[0] + classCount * self.__smoothingFactor)) for key in range(0, classCount)]);
        self.__discreteFeatureProbability = list(range(0, classCount));
        self.__continuousFeatureArguments = list(range(0, classCount));

        for key in range(0, classCount):
            if len(self.__discreteFeatureIndices) > 0:
                self.__discreteFeatureProbability[key] = self.__calcDiscreteProbability(classSets[key][:, self.__discreteFeatureIndices] if key in classSets else None, self.__discreteFeatureValueNumbers);

            if len(self.__continuousFeatureIndices) > 0:
                self.__continuousFeatureArguments[key] = self.__calcContinuousArguments(classSets[key][:, self.__continuousFeatureIndices] if key in classSets else None, len(self.__continuousFeatureIndices));


    def predict(self, dataSet):
        if dataSet is None or not isinstance(dataSet, np.matrix):
            raise ValueError();

        discreteRange = None;
        discreteSet, continuousSet = None, None;
        allProbability, discreteProbability, continuousProbability = None, None, None;
        result = np.mat(np.zeros((dataSet.shape[0], self.__classProbability.shape[1])));

        if len(self.__discreteFeatureIndices) > 0:
            discreteSet = dataSet[:, self.__discreteFeatureIndices];
            discreteRange = list(range(0, len(self.__discreteFeatureIndices)));

        if len(self.__continuousFeatureIndices) > 0:
            continuousSet = dataSet[:, self.__continuousFeatureIndices];

        for c in range(0, result.shape[1]):
            if discreteSet is not None:
                discreteProbability = self.__discreteFeatureProbability[c][np.mat(discreteSet, dtype = int), discreteRange];

            if continuousSet is not None:
                normalArguments = self.__continuousFeatureArguments[c];
                mean, var, std = normalArguments[0, :], np.power(normalArguments[1, :], 2), normalArguments[1, :];

                zeroStdIndices = np.where(std == 0)[1];
                if len(zeroStdIndices) > 0:
                    var[:, zeroStdIndices] = 1;
                    std[:, zeroStdIndices] = 1;

                continuousProbability = np.power(continuousSet - mean, 2) / (-2 * var) - np.log(std);
                if len(zeroStdIndices) > 0:
                    continuousProbability[:, zeroStdIndices] = 0;

            if discreteSet is not None and continuousSet is not None:
                allProbability = np.hstack((discreteProbability, continuousProbability));
            elif discreteSet is not None:
                allProbability = discreteProbability;
            else:
                allProbability = continuousProbability;

            result[:, c] = allProbability.sum(1);

        result = result + self.__classProbability;

        return np.mat(result.argmax(axis = 1));

