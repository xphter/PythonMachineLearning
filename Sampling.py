import math;
import random;
import numpy as np;


def __group(matrix):
    label = None;
    classes = {};

    for row in matrix:
        label = row[-1];

        if label not in classes:
            classes[label] = [row];
        else:
            classes[label].append(row);

    return classes;


def __splitDuplicate(matrix, featureIndex, differentData):
    if len(matrix) == 0:
        return matrix;

    value = matrix[0, 0];
    differentCount = len(differentData);
    rowCount, columnCount = matrix.shape;
    duplicateData = matrix[0, :].reshape(1,columnCount);

    for rowIndex in range(1, rowCount):
        if matrix[rowIndex, featureIndex] != value:
            differentData.append(matrix[rowIndex, :]);
        else:
            duplicateData = np.append(duplicateData, matrix[rowIndex, :].reshape(1, columnCount), axis = 0);

    if len(duplicateData) == 1:
        differentData.insert(differentCount, matrix[0, :]);
        duplicateData = np.delete(duplicateData, 0, axis = 0);

    return duplicateData;


def chooseFeature(total, count):
    if count >= total:
        return list(range(0, total));

    return random.sample(range(0, total), count);


def holdOutSampling(matrix, proportion):
    indices = None;
    trainSet, testSet = [], [];
    classes = __group(matrix);

    for subSet in classes.values():
        indices = random.sample(list(range(0, len(subSet))), math.ceil(proportion * len(subSet)));

        trainSet.extend([subSet[index] for index in indices]);
        testSet.extend([subSet[index] for index in range(0, len(subSet)) if index not in indices]);

    return trainSet, testSet;


def bootstrapSampling(matrix):
    result = [];
    classes = __group(matrix);

    for subSet in classes.values():
        result.extend([random.choice(subSet) for i in range(0, len(subSet))]);

    return result;


def removeDuplicate(matrix):
    differentData = [];
    duplicateData = matrix.copy();

    for featureIndex in range(0, matrix.shape[1] - 1):
        duplicateData = __splitDuplicate(duplicateData, featureIndex, differentData);

    return np.array(differentData), duplicateData;

