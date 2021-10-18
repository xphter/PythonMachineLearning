import abc;
import math;
import matplotlib.pyplot as plt;
from typing import List, Tuple, Callable, Any;

from Functions import *;


class INetModule(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def isTrainingMode(self) -> bool:
        pass;


    @isTrainingMode.setter
    @abc.abstractmethod
    def isTrainingMode(self, value: bool):
        pass;


    @property
    @abc.abstractmethod
    def params(self) -> List[np.ndarray]:
        pass;


    @property
    @abc.abstractmethod
    def grads(self) -> List[np.ndarray]:
        pass;


    @abc.abstractmethod
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        pass;


    @abc.abstractmethod
    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        pass;


class INetLoss(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def loss(self) -> float:
        pass;


    @abc.abstractmethod
    def forward(self, *data: np.ndarray) -> float:
        pass;


    @abc.abstractmethod
    def backward(self) -> Tuple[np.ndarray]:
        pass;


class INetOptimizer(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def learningRate(self) -> float:
        pass;


    @abc.abstractmethod
    def update(self, params : List[np.ndarray], grads : List[np.ndarray]):
        pass;


class INetAccuracyEvaluator(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def accuracy(self) -> float:
        pass;


    @abc.abstractmethod
    def update(self, *data : np.ndarray):
        pass;


    @abc.abstractmethod
    def reset(self):
        pass;


class NetModuleBase(INetModule, metaclass = abc.ABCMeta):
    def __init__(self):
        self._name = None;
        self._params = [];
        self._grads = [];
        self._isTrainingMode = True;


    def __repr__(self):
        return self.__str__();


    def __str__(self):
        return self._name;


    @property
    def isTrainingMode(self) -> bool:
        return self._isTrainingMode;


    @isTrainingMode.setter
    def isTrainingMode(self, value : bool):
        self._isTrainingMode = value;
        self._setTrainingMode(value);


    @property
    def params(self) -> List[np.ndarray]:
        return self._params;


    @property
    def grads(self) -> List[np.ndarray]:
        return self._grads;


    def _setTrainingMode(self, value : bool):
        pass;


class NetLossBase(INetLoss, metaclass = abc.ABCMeta):
    def __init__(self):
        self._loss = None;


    @property
    def loss(self) -> float:
        return self._loss;


class NetOptimizerBase(INetOptimizer, metaclass = abc.ABCMeta):
    def __init__(self, lr : float):
        self._lr = lr;


    def learningRate(self) -> float:
        return self._lr;


class ReluLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._mask = None;
        self._name = "Relu";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        self._mask = (X > 0) - 0;

        return relu(X), ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dX = dY * self._mask;

        return dX, ;


class SigmoidLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._Y = None;
        self._name = "Sigmoid";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        self._Y = sigmoid(X);

        return self._Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dX = dY * sigmoidGradient(self._Y);

        return dX, ;


class TanhLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._Y = None;
        self._name = "Tanh";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        self._Y = tanh(X);

        return self._Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dX = dY * tanhGradient(self._Y);

        return dX, ;


class DropoutLayer(NetModuleBase):
    def __init__(self, dropoutRatio = 0.5):
        super().__init__();

        self._mask = None;
        self._dropoutRatio = dropoutRatio;
        self._name = f"Dropout {dropoutRatio}";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];

        if self._isTrainingMode:
            self._mask = (np.random.rand(*X.shape) > self._dropoutRatio) / (1 - self._dropoutRatio);
            return X * self._mask, ;
        else:
            return X, ;
        # if isTraining:
        #     self._mask = np.random.rand(*X.shape) > self._dropoutRatio;
        #     return X * self._mask;
        # else:
        #     return X * (1.0 - self._dropoutRatio);


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dX = dY * self._mask;

        return dX, ;


class AffineLayer(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, includeBias : bool = True, W : np.ndarray = None, b : np.ndarray = None):
        super().__init__();

        self._X = None;
        self._shape = None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"Affine {inputSize}*{outputSize}";

        self._weight = 0.01 * np.random.randn(inputSize, outputSize) if W is None else W;
        self._bias = (np.zeros(outputSize) if b is None else b) if includeBias else None;

        self._params.append(self._weight);
        self._grads.append(np.zeros_like(self._weight));
        if self._bias is not None:
            self._params.append(self._bias);
            self._grads.append(np.zeros_like(self._bias));


    @property
    def weight(self):
        return self._weight;


    @property
    def bias(self):
        return self._bias;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];

        if X.ndim <= 2:
            self._X = X;
            self._shape = None;
        else:
            self._X = X.reshape(len(X), -1);
            self._shape = X.shape;

        Y = self._X @ self._weight;
        if self._bias is not None:
            Y += self._bias;

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dW = self._X.T @ dY;
        self._grads[0][...] = dW;

        if self._bias is not None:
            db = np.sum(dY, 0);
            self._grads[1][...] = db;

        dX = dY @ self._weight.T;
        if self._shape is not None:
            dX = dX.reshape(*self._shape);

        return dX, ;


class BatchNormalizationLayer(NetModuleBase):
    def __init__(self, inputSize : int, gamma : np.ndarray = None, beta : np.ndarray = None, epsilon = 1e-8):
        super().__init__();

        self._epsilon = epsilon;
        self._n = None;
        self._Xmu = None;
        self._std = None;
        self._XHat = None;
        self._shape = None;
        self._name = "BatchNormalization";

        self._gamma = np.ones(inputSize) if gamma is None else gamma;
        self._beta = np.zeros(inputSize) if beta is None else beta;
        self._params.append(self._gamma);
        self._params.append(self._beta);
        self._grads.append(np.zeros_like(self._gamma));
        self._grads.append(np.zeros_like(self._beta));


    @property
    def gamma(self) -> np.ndarray:
        return self._gamma;


    @property
    def beta(self) -> np.ndarray:
        return self._beta;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];

        if X.ndim <= 2:
            self._shape = None;
        else:
            X = X.reshape(len(X), -1);
            self._shape = X.shape;

        mu = X.mean(0);
        self._n = len(X);
        self._Xmu = X - mu;
        self._std = np.sqrt(np.square(self._Xmu).mean(0) + self._epsilon);
        self._XHat = self._Xmu / self._std;
        Y = self._gamma * self._XHat + self._beta;

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dXHat = dY * self._gamma;
        dXmu = dXHat / self._std - np.sum(dXHat * self._Xmu, 0) / np.power(self._std, 3) * self._Xmu / self._n;

        dGamma = np.sum(dY * self._XHat, 0);
        dBeta = np.sum(dY, 0);
        dX = dXmu - np.sum(dXmu, 0) / self._n;

        self._grads[0][...] = dGamma;
        self._grads[1][...] = dBeta;
        if self._shape is not None:
            dX = dX.reshape(*self._shape);

        return dX, ;


class SoftmaxLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._Y = None;
        self._name = "Softmax";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        self._Y = softmax(X);

        return self._Y, ;


    # dX = Y * (dY - ∑(dY * Y))
    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        Z = dY * self._Y;
        dX = Z - self._Y * np.sum(Z, Z.ndim - 1, keepdims = True);

        return dX, ;


class CrossEntropyLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._Y = None;
        self._T = None;


    def forward(self, *data: np.ndarray) -> float:
        self._Y, self._T = data;
        self._loss = crossEntropyError(self._Y, self._T);

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        dY = -self._T / self._Y / len(self._T);

        return dY, ;


class SoftmaxWithCrossEntropyLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._Y = None;
        self._T = None;


    def forward(self, *data: np.ndarray) -> float:
        X, T = data;
        self._T = T;
        self._Y = softmax(X);
        self._loss = crossEntropyError(self._Y, self._T);

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        dX = (self._Y - self._T) / len(self._T);

        return dX, ;


class SigmoidWithCrossEntropyLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._Y = None;
        self._T = None;


    def forward(self, *data: np.ndarray) -> float:
        X, T = data;
        self._T = T;
        self._Y = sigmoid(X);
        self._loss = crossEntropyError(np.column_stack((self._Y, 1 - self._Y)),
                                       np.column_stack((self._T, 1 - self._T)));

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        dX = (self._Y - self._T) / len(self._T);

        return dX, ;


class IdentityWithMeanSquareLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._Y = None;
        self._T = None;


    def forward(self, *data: np.ndarray) -> float:
        self._Y, self._T = data;
        self._loss = meanSquareError(self._Y, self._T);

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        dX = (self._Y - self._T) / len(self._T);

        return dX, ;


class SumWithMeanSquareLossLayer(NetLossBase):
    def __init__(self):
        super().__init__();

        self._Y = None;
        self._T = None;
        self._shape = None;


    def forward(self, *data: np.ndarray) -> float:
        X, T = data;
        self._T = T;
        self._Y = np.sum(X, 1, keepdims = True);
        self._loss = meanSquareError(self._Y, self._T);
        self._shape = X.shape;

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        dY = (self._Y - self._T) / len(self._T);
        dX = dY * np.ones_like(self._shape);

        return dX, ;


class ConvolutionLayer(NetModuleBase):
    def __init__(self, FN : int, C : int, FH : int, FW : int, stride = 1, pad = 0, W : np.ndarray = None, b : np.ndarray = None):
        super().__init__();

        self._stride = stride;
        self._pad = pad;
        self._shape = None;
        self._colX = None;
        self._colW = None;
        self._name = f"Convolution {FN}*{C}*{FH}*{FW}";

        self._weight = 0.01 * np.random.randn(FN, C, FH, FW) if W is None else W;
        self._bias = np.zeros(FN) if b is None else b;

        self._params.append(self._weight);
        self._params.append(self._bias);
        self._grads.append(np.zeros_like(self._weight));
        self._grads.append(np.zeros_like(self._bias));


    @property
    def weight(self):
        return self._weight;


    @property
    def bias(self):
        return self._bias;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        self._shape = X.shape;

        N, C, H, W = X.shape;
        FN, C, FH, FW = self._weight.shape;
        OH = convOutputSize(H, FH, self._stride, self._pad);
        OW = convOutputSize(W, FW, self._stride, self._pad);

        self._colX = im2col(X, FH, FW, self._stride, self._pad);
        self._colW = self._weight.reshape(FN, -1).T;
        Y = self._colX @ self._colW + self._bias;
        Y = Y.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        FN, C, FH, FW = self._weight.shape;

        colDY = dY.transpose(0, 2, 3, 1).reshape(-1, FN);
        dW = self._colX.T @ colDY;
        db = np.sum(colDY, 0);
        dX = colDY @ self._colW.T;
        dX = col2im(dX, self._shape, FH, FW, self._stride, self._pad, True);

        self._grads[0][...] = dW.T.reshape(FN, C, FH, FW);
        self._grads[1][...] = db;

        return dX, ;


class MaxPoolingLayer(NetModuleBase):
    def __init__(self, PH : int, PW : int, stride = 1, pad = 0):
        super().__init__();

        self._PH = PH;
        self._PW = PW;
        self._stride = stride;
        self._pad = pad;
        self._shape = None;
        self._argMax = None;
        self._name = "MaxPooling";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        self._shape = X.shape;

        N, C, H, W = X.shape;
        OH = convOutputSize(H, self._PH, self._stride, self._pad);
        OW = convOutputSize(W, self._PW, self._stride, self._pad);

        col = im2col(X, self._PH, self._PW, self._stride, self._pad).reshape(-1, self._PH * self._PW);
        Y = np.amax(col, 1).reshape(N, OH, OW, C).transpose(0, 3, 1, 2);
        self._argMax = np.argmax(col, 1);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        N, C, OH, OW = dY.shape;
        poolingSize = self._PH * self._PW;

        colDY = dY.transpose(0, 2, 3, 1);
        dMax = np.zeros((colDY.size, poolingSize));
        dMax[np.arange(self._argMax.size), self._argMax.flatten()] = colDY.flatten();
        dMax = dMax.reshape(-1, C * poolingSize);
        dX = col2im(dMax, self._shape, self._PH, self._PW, self._stride, self._pad, True);

        return dX, ;


class SGD(NetOptimizerBase):
    def __init__(self, lr : float = 0.001):
        super().__init__(lr);


    def update(self, params : List[np.ndarray], grads : List[np.ndarray]):
        for i in range(len(params)):
            params[i] -= self._lr * grads[i];


class Momentum(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, momentum : float = 0.9):
        super().__init__(lr);

        self._v = None;
        self._momentum = momentum;


    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self._v is None:
            self._v = [np.zeros_like(item) for item in params];

        for i in range(len(params)):
            self._v[i] = self._momentum * self._v[i] + self._lr * grads[i];
            params[i] -= self._v[i];


class Nesterov(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, momentum : float = 0.9):
        super().__init__(lr);

        self._v = None;
        self._momentum = momentum;


    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self._v is None:
            self._v = [np.zeros_like(item) for item in params];

        for i in range(len(params)):
            v = self._v[i];
            self._v[i] = self._momentum * self._v[i] + self._lr * grads[i];
            params[i] -= (1 + self._momentum) * self._v[i] - self._momentum * v;


class AdaGrad(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, epsilon : float = 1e-8):
        super().__init__(lr);

        self._s = None;
        self._epsilon = epsilon;


    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self._s is None:
            self._s = [np.zeros_like(item) for item in params];

        for i in range(len(params)):
            self._s[i] += grads[i] ** 2;
            params[i] -= self._lr * grads[i] / (np.sqrt(self._s[i]) + self._epsilon);


class RMSprop(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, beta : float = 0.9, epsilon : float = 1e-8):
        super().__init__(lr);

        self._s = None;
        self._beta = beta;
        self._epsilon = epsilon;


    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self._s is None:
            self._s = [np.zeros_like(item) for item in params];

        for i in range(len(params)):
            self._s[i] = self._beta * self._s[i] + (1 - self._beta) * grads[i] ** 2;
            params[i] -= self._lr * grads[i] / (np.sqrt(self._s[i]) + self._epsilon);


class Adam(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, beta1 : float = 0.9, beta2 : float = 0.999, epsilon : float = 1e-8):
        super().__init__(lr);

        self._v = None;
        self._s = None;
        self._beta1 = beta1;
        self._beta2 = beta2;
        self._epsilon = epsilon;
        self._t = 0;


    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self._v is None:
            self._v = [np.zeros_like(item) for item in params];
        if self._s is None:
            self._s = [np.zeros_like(item) for item in params];

        self._t += 1;

        for i in range(len(params)):
            self._v[i] = self._beta1 * self._v[i] + (1 - self._beta1) * grads[i];
            self._s[i] = self._beta2 * self._s[i] + (1 - self._beta2) * grads[i] ** 2;

            v = self._v[i] / (1 - self._beta1 ** self._t);
            s = self._s[i] / (1 - self._beta2 ** self._t);

            params[i] -= self._lr * v / (np.sqrt(s) + self._epsilon);


class SequentialContainer(NetModuleBase):
    def __init__(self, *modules : INetModule):
        super().__init__();

        self._modules = modules;
        for m in modules:
            self._params.extend(m.params);
            self._grads.extend(m.grads);

        self._name = "  -->  ".join([str(m) for m in modules]);


    @property
    def modules(self) -> Tuple[INetModule]:
        return self._modules;


    def _setTrainingMode(self, value: bool):
        for m in self._modules:
            m.isTrainingMode = value;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        for m in self._modules:
            data = m.forward(*data);

        return data;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        for m in reversed(self._modules):
            dout = m.backward(*dout);

        return dout;


    def apply(self, func : Callable):
        for m in self._modules:
            func(m);


class DataIterator:
    def __init__(self, data : List[np.ndarray], batchSize = 2 ** 8, shuffle : bool = True):
        self._step = 0;
        self._data = data;
        self._length = len(data[0]);
        self._batchSize = batchSize;
        self._index = np.arange(self._length);
        if shuffle:
            np.random.shuffle(self._index);


    def _iterate(self):
        while self._step * self._batchSize < self._length:
            startIndex = self._step * self._batchSize;
            index = self._index[startIndex: startIndex + self._batchSize];
            self._step += 1;

            yield tuple([d[index] for d in self._data]);


    def __iter__(self):
        self._step = 0;
        return self._iterate();


class ClassifierAccuracyEvaluator(INetAccuracyEvaluator):
    def __init__(self):
        self._rightCount = 0.0;
        self._totalCount = 0.0;


    @property
    def accuracy(self) -> float:
        return self._rightCount / self._totalCount;


    def update(self, *data: np.ndarray):
        Y, T = data;
        self._rightCount += int(np.sum(np.argmax(Y, 1) == np.argmax(T, 1)));
        self._totalCount += len(Y);


    def reset(self):
        self._rightCount = 0.0;
        self._totalCount = 0.0;


class NetTrainer:
    def __init__(self, model : INetModule, lossFunc : INetLoss, optimizer : INetOptimizer, evaluator : INetAccuracyEvaluator):
        self._model = model;
        self._lossFunc = lossFunc;
        self._optimizer = optimizer;
        self._evaluator = evaluator;

        self._lossData = [];
        self._trainAccuracyData = [];
        self._testAccuracyData = [];


    def _calcAccuracy(self, iterator : DataIterator) -> float:
        self._evaluator.reset();

        for X, T in iterator:
            Y = self._model.forward(X);
            self._evaluator.update(*Y, T);

        return self._evaluator.accuracy;


    def train(self, maxEpoch : int, trainIterator : DataIterator, testIterator : DataIterator = None):
        lossValues = [];

        self._lossData.clear();
        self._trainAccuracyData.clear();
        self._testAccuracyData.clear();

        for epoch in range(maxEpoch):
            lossValues.clear();

            for X, T in trainIterator:
                Y = self._model.forward(X);
                loss = self._lossFunc.forward(*Y, T);
                lossValues.append(loss);

                self._model.backward(*self._lossFunc.backward());
                self._optimizer.update(self._model.params, self._model.grads);

            self._lossData.append(sum(lossValues) / len(lossValues));
            self._trainAccuracyData.append(self._calcAccuracy(trainIterator));
            if testIterator is not None:
                self._testAccuracyData.append(self._calcAccuracy(testIterator));

            if testIterator is not None:
                print(f"epoch {epoch}, loss: {lossValues[-1]}, train accuracy: {self._trainAccuracyData[-1]}, test accuracy: {self._testAccuracyData[-1]}");
            else:
                print(f"epoch {epoch}, loss: {lossValues[-1]}, train accuracy: {self._trainAccuracyData[-1]}");


    def plot(self):
        plt.figure(1);
        plt.xlabel("epoch");
        plt.ylabel('loss');

        plt.plot(self._lossData, "o-k", label = "loss");
        if len(self._trainAccuracyData) > 0:
            plt.plot(self._trainAccuracyData, "D-b", label = "train accuracy");
        if len(self._testAccuracyData) > 0:
            plt.plot(self._testAccuracyData, "s-r", label = "test accuracy");

        plt.legend(loc ='upper right');
        plt.show(block = True);
        plt.close();
