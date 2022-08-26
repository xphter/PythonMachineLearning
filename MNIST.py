import os;
import struct;

from ImportNumpy import *;


class MNIST:
    __TRAIN_DATA_FILE_NAME = "train-images-idx3-ubyte";
    __TRAIN_LABELS_FILE_NAME = "train-labels-idx1-ubyte";
    __TEST_DATA_FILE_NAME = "t10k-images-idx3-ubyte";
    __TEST_LABELS_FILE_NAME = "t10k-labels-idx1-ubyte";


    def __init__(self, folderPath : str, flatten : bool = False, normalize : bool = True, onehot : bool = False):
        self.__normalize = normalize;
        self.__trainX = self.__loadData(os.path.join(folderPath, MNIST.__TRAIN_DATA_FILE_NAME), flatten, normalize);
        self.__testX = self.__loadData(os.path.join(folderPath, MNIST.__TEST_DATA_FILE_NAME), flatten, normalize);
        self.__trainY = self.__loadLabels(os.path.join(folderPath, MNIST.__TRAIN_LABELS_FILE_NAME), onehot);
        self.__testY = self.__loadLabels(os.path.join(folderPath, MNIST.__TEST_LABELS_FILE_NAME), onehot);


    @property
    def trainX(self) -> np.ndarray:
        return self.__trainX;


    @property
    def trainY(self) -> np.ndarray:
        return self.__trainY;


    @property
    def testX(self) -> np.ndarray:
        return self.__testX;


    @property
    def testY(self) -> np.ndarray:
        return self.__testY;


    def __loadData(self, path : str, flatten : bool, normalize : bool) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(path);

        X = None;

        with open(path, "rb") as file:
            magic, n, rows, columns = struct.unpack(">IIII", file.read(16));

            # X = np.fromfile(file, np.uint8).reshape(n, rows * columns);
            X = np.fromfile(file, np.uint8);

        if flatten:
            X = X.reshape(n, rows * columns);
        else:
            X = X.reshape(n, 1, rows, columns);

        if normalize:
            X = (X / 255.0).astype(defaultDType);

        return X;


    def __loadLabels(self, path : str, onehot : bool) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(path);

        labels = None;

        with open(path, "rb") as file:
            magic, n = struct.unpack(">II", file.read(8));

            labels = np.fromfile(file, np.uint8);

        if onehot:
            Y = np.zeros((n, 10), dtype = np.int8);
            Y[list(range(n)), labels] = 1;

            return Y;
        else:
            return labels;
