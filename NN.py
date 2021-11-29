import abc;
import math;
import time;
import datetime;
import collections;

import matplotlib.pyplot as plt;

from typing import List, Tuple, Callable, Any, Optional, Iterable;
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
    def reset(self):
        pass;


    @abc.abstractmethod
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        pass;


    @abc.abstractmethod
    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        pass;


class INetModel(INetModule, metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def getFinalTag(self, T : np.ndarray) -> np.ndarray:
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


class IDataIterator(Iterable, metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def totalIterations(self) -> int:
        pass;


class INetAccuracyEvaluator(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass;


    @property
    @abc.abstractmethod
    def accuracy(self) -> float:
        pass;


    @abc.abstractmethod
    def fromLoss(self, lossValues : List[float]) -> bool:
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


    def reset(self):
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
        # if self._isTrainingMode:
        #     self._mask = np.random.rand(*X.shape) > self._dropoutRatio;
        #     return X * self._mask, ;
        # else:
        #     return X * (1.0 - self._dropoutRatio), ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dX = dY * self._mask;

        return dX, ;


class ReshapeLayer(NetModuleBase):
    def __init__(self, *newShapes : Tuple):
        super().__init__();

        self._originalShapes = [];
        self._newShapes = newShapes;
        self._name = "Reshape";


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray]:
        self._originalShapes.clear();
        output : List[np.ndarray] = [];

        for X, shape in zip(data, self._newShapes):
            self._originalShapes.append(X.shape);
            output.append(X.reshape(shape));

        return tuple(output);


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray]:
        dX : List[np.ndarray] = [];

        for dY, shape in zip(dout, self._originalShapes):
            dX.append(dY.reshape(shape));

        return tuple(dX);


class AffineLayer(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, includeBias : bool = True, W : np.ndarray = None, b : np.ndarray = None):
        super().__init__();

        self._X = None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"Affine {inputSize}*{outputSize}";

        self._weight = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, outputSize) if W is None else W;
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
        self._X = data[0];

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

        return dX, ;


class BatchNormalizationLayer(NetModuleBase):
    def __init__(self, inputSize : int, gamma : np.ndarray = None, beta : np.ndarray = None, epsilon = 1e-8):
        super().__init__();

        self._epsilon = epsilon;
        self._n = None;
        self._Xmu = None;
        self._std = None;
        self._XHat = None;
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

        self._weight = math.sqrt(2.0 / (C * FH * FW)) * np.random.randn(FN, C, FH, FW) if W is None else W;
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


class EmbeddingLayer(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, W : np.ndarray = None):
        super().__init__();

        self._index = None;
        self._shape = None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"Embedding {inputSize}*{outputSize}";

        self._weight = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, outputSize) if W is None else W;
        self._params.append(self._weight);
        self._grads.append(np.zeros_like(self._weight));


    @property
    def weight(self):
        return self._weight;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        self._shape = X.shape;
        self._index = X.flatten();
        Y = self._weight[self._index].reshape(X.shape + (self._outputSize, ));

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dY = dY.reshape(-1, self._outputSize);

        dW = self._grads[0];
        dW[...] = 0;

        # np.add.at(dW, self._index, dY);
        npAddAt(dW, self._index, dY);

        # the dX is always zero!!!
        return np.zeros(self._shape), ;


class EmbeddingWithDotLayer(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, W : np.ndarray = None):
        super().__init__();

        self._X = None;
        self._W = None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"EmbeddingWithDot {outputSize}*{inputSize}";

        self._weight = math.sqrt(2.0 / inputSize) * np.random.randn(outputSize, inputSize) if W is None else W;
        self._embeddingLayer = EmbeddingLayer(outputSize, inputSize, W = self._weight);


    @property
    def params(self) -> List[np.ndarray]:
        return self._embeddingLayer.params;


    @property
    def grads(self) -> List[np.ndarray]:
        return self._embeddingLayer.grads;


    @property
    def weight(self):
        return self._weight;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X, T = data;
        self._X = X;
        self._W = self._embeddingLayer.forward(T)[0];
        Y = np.sum(self._X * self._W, axis = -1);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dY = np.expand_dims(dY, axis = -1);

        dW = dY * self._X;
        dX = dY * self._W;
        self._embeddingLayer.backward(dW);

        return dX, ;


class RnnCell(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, Wx : np.ndarray = None, Wh : np.ndarray = None, b : np.ndarray = None):
        super().__init__();

        self._X = None;
        self._H = None;
        self._Y = None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"RNN Cell {inputSize}*{outputSize}";

        self._weightX = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, outputSize) if Wx is None else Wx;
        self._weightH = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, outputSize) if Wh is None else Wh;
        self._bias = np.zeros(outputSize) if b is None else b;

        weights = [self._weightX, self._weightH, self._bias];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    @property
    def weightX(self) -> np.ndarray:
        return self._weightX;


    @property
    def weightH(self) -> np.ndarray:
        return self._weightH;


    @property
    def bias(self) -> np.ndarray:
        return self._bias;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        self._X, self._H = data;
        self._Y = tanh(self._X @ self._weightX + self._H @ self._weightH + self._bias);

        return self._Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dY = dY * tanhGradient(self._Y);

        dWx = self._X.T @ dY;
        dWh = self._H.T @ dY;
        db = np.sum(dY, axis = 0);

        dX = dY @ self._weightX.T;
        dH = dY @ self._weightH.T;

        self._grads[0][...] = dWx;
        self._grads[1][...] = dWh;
        self._grads[2][...] = db;

        return dX, dH;


class RnnLayer(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, Wx : np.ndarray = None, Wh : np.ndarray = None, b : np.ndarray = None, stateful : bool = True):
        super().__init__();

        self._H = None;
        self._dH = None;
        self._stateful = stateful;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"RNN {inputSize}*{outputSize}";

        self._weightX = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, outputSize) if Wx is None else Wx;
        self._weightH = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, outputSize) if Wh is None else Wh;
        self._bias = np.zeros(outputSize) if b is None else b;
        self._rnnModules : List[RnnCell] = [];

        weights = [self._weightX, self._weightH, self._bias];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    @property
    def weightX(self) -> np.ndarray:
        return self._weightX;


    @property
    def weightH(self) -> np.ndarray:
        return self._weightH;


    @property
    def bias(self) -> np.ndarray:
        return self._bias;


    def reset(self):
        self._H = None;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        N, T, D = X.shape;

        if len(self._rnnModules) != T:
            self._rnnModules = [RnnCell(self._inputSize, self._outputSize, self._weightX, self._weightH, self._bias) for _ in range(T)];

        if not self._stateful or self._H is None:
            self._H = np.zeros((N, self._outputSize));

        Y = np.zeros((N, T, self._outputSize));
        for t in range(T):
            self._H = self._rnnModules[t].forward(X[:, t, :], self._H)[0];
            Y[:, t, :] = self._H;

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        N, T = dY.shape[: 2];

        # truncated BPTT
        self._dH = np.zeros_like(self._H);
        dX = np.zeros((N, T, self._inputSize));
        for i in range(len(self._grads)):
            self._grads[i][...] = 0;

        for t in reversed(range(T)):
            rnn = self._rnnModules[t];
            dX[:, t, :], self._dH = rnn.backward(dY[:, t, :] + self._dH);

            for i in range(len(self._grads)):
                self._grads[i] += rnn.grads[i];

        return dX, ;


class CorpusNegativeSampler:
    def __init__(self, corpus : np.ndarray, sampleSize : int, exponent : float = 0.75):
        vocab, counts = np.unique(corpus, return_counts = True);

        self._vocab = vocab;
        self._vocabSize = len(vocab);
        self._sampleSize = sampleSize;
        self._probability = np.power(counts, exponent);
        self._probability /= np.sum(self._probability);
        self._resampleNumber = 0;


    @property
    def sampleSize(self) -> int:
        return self._sampleSize;


    @property
    def resampleNumber(self) -> int:
        return self._resampleNumber;


    def _findSample(self, sampleSize : int, exceptIndex = None):
        p = self._probability;
        if exceptIndex is not None:
            p = p.copy();
            p[exceptIndex] = 0;
            p /= np.sum(p);

        return np.random.choice(self._vocab, sampleSize, replace = DeviceConfig.EnableGPU or sampleSize >= self._vocabSize, p = p);


    # return: negative samples, final tags
    def getSample(self, T : np.ndarray) -> (np.ndarray, np.ndarray):
        index = None;
        PS = T.flatten();
        NS = self._findSample(T.size * self._sampleSize);
        NT = np.zeros_like(NS);
        for i in range(len(PS)):
            if not np.any(index := (NS[i * self._sampleSize: (i + 1) * self._sampleSize] == PS[i])):
                continue;

            # NS[i * self._sampleSize: (i + 1) * self._sampleSize] = self._findSample(self._sampleSize, PS[i]);
            NT[i * self._sampleSize: (i + 1) * self._sampleSize][index] = 1;
            self._resampleNumber += 1;

        PS = np.expand_dims(T, axis = -1);
        PT = np.ones_like(PS);
        NS = NS.reshape(T.shape + (self._sampleSize, ));
        NT = NT.reshape(T.shape + (self._sampleSize, ));

        return np.concatenate((PS, NS), axis = -1), np.concatenate((PT, NT), axis = -1);


class CBOWModel(NetModuleBase, INetModel):
    def __init__(self, windowSize : int, vocabSize : int, hiddenSize : int, negativeSampler : CorpusNegativeSampler, inW : np.ndarray = None, outW : np.ndarray = None):
        super().__init__();

        self._finalTag = None;
        self._windowSize = windowSize;
        self._vocabSize = vocabSize;
        self._hiddenSize = hiddenSize;
        self._negativeSampler = negativeSampler;
        self._W0 = math.sqrt(2.0 / vocabSize) * np.random.randn(vocabSize, hiddenSize) if inW is None else inW;
        self._W1 = math.sqrt(2.0 / hiddenSize) * np.random.randn(hiddenSize, vocabSize).T if outW is None else outW;

        self._embeddingLayer = EmbeddingLayer(vocabSize, hiddenSize, W = self._W0);
        self._outputLayer = EmbeddingWithDotLayer(hiddenSize, vocabSize, W = self._W1);

        self._params.extend([self._W0, self._W1]);
        self._grads.extend([np.zeros_like(self._W0), np.zeros_like(self._W1)]);

        self._name = "CBOW";


    def _setTrainingMode(self, value : bool):
        self._embeddingLayer.isTrainingMode = value;
        self._outputLayer.isTrainingMode = value;


    @property
    def wordVector(self):
        return self._W0;


    @property
    def weights(self) -> (np.ndarray, np.ndarray):
        return self._W0, self._W1;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X, T = data;
        N, C = X.shape;
        S = self._negativeSampler.sampleSize;

        H = self._embeddingLayer.forward(X)[0];
        H = np.sum(H, axis = 1) / C;

        H = np.expand_dims(H, axis = -2);
        H = np.repeat(H, S + 1, axis = -2);
        T, self._finalTag = self._negativeSampler.getSample(T);

        Y = self._outputLayer.forward(H, T)[0];

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        N, C = len(dY), 2 * self._windowSize;

        dH = self._outputLayer.backward(dY)[0];
        dH = np.sum(dH, axis = -2) / C;
        dH = np.expand_dims(dH, axis = 1);
        dH = np.repeat(dH, C, axis = 1);
        self._embeddingLayer.backward(dH);

        self._grads[0][...] = self._embeddingLayer.grads[0];
        self._grads[1][...] = self._outputLayer.grads[0];

        return np.zeros((N, C)), ;


    def getFinalTag(self, T : np.ndarray) -> np.ndarray:
        return self._finalTag;


class SkipGramModel(NetModuleBase, INetModel):
    def __init__(self, windowSize : int, vocabSize : int, hiddenSize : int, negativeSampler : CorpusNegativeSampler, inW : np.ndarray = None, outW : np.ndarray = None):
        super().__init__();

        self._finalTag = None;
        self._windowSize = windowSize;
        self._vocabSize = vocabSize;
        self._hiddenSize = hiddenSize;
        self._negativeSampler = negativeSampler;
        self._W0 = math.sqrt(2.0 / vocabSize) * np.random.randn(vocabSize, hiddenSize) if inW is None else inW;
        self._W1 = math.sqrt(2.0 / hiddenSize) * np.random.randn(hiddenSize, vocabSize).T if outW is None else outW;

        self._embeddingLayer = EmbeddingLayer(vocabSize, hiddenSize, W = self._W0);
        self._outputLayer = EmbeddingWithDotLayer(hiddenSize, vocabSize, W = self._W1);

        self._params.extend([self._W0, self._W1]);
        self._grads.extend([np.zeros_like(self._W0), np.zeros_like(self._W1)]);

        self._name = "SkipGram";


    def _setTrainingMode(self, value: bool):
        self._embeddingLayer.isTrainingMode = value;
        self._outputLayer.isTrainingMode = value;


    @property
    def wordVector(self):
        return self._W0;


    @property
    def weights(self) -> (np.ndarray, np.ndarray):
        return self._W0, self._W1;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X, T = data;
        N, C = T.shape;
        S = self._negativeSampler.sampleSize;

        H = self._embeddingLayer.forward(X)[0];
        H = np.expand_dims(H, axis = -2);
        H = np.repeat(H, C, axis = -2);

        H = np.expand_dims(H, axis = -2);
        H = np.repeat(H, S + 1, axis = -2);
        T, self._finalTag = self._negativeSampler.getSample(T);

        Y = self._outputLayer.forward(H, T)[0];

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dH = self._outputLayer.backward(dY)[0];
        dH = np.sum(dH, axis = (-2, -3));
        dX = self._embeddingLayer.backward(dH)[0];

        self._grads[0][...] = self._embeddingLayer.grads[0];
        self._grads[1][...] = self._outputLayer.grads[0];

        return dX, ;


    def getFinalTag(self, T : np.ndarray) -> np.ndarray:
        return self._finalTag;


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
        dX = Z - self._Y * np.sum(Z, -1, keepdims = True);

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
        dY = -self._T / self._Y / lengthExceptLastDimension(self.T);

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
        dX = (self._Y - self._T) / lengthExceptLastDimension(self._T);

        return dX, ;


class SoftmaxWithCrossEntropy1DLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._Y = None;
        self._T = None;


    def forward(self, *data: np.ndarray) -> float:
        X, T = data;
        self._T = T;
        self._Y = softmax(X);
        self._loss = crossEntropyError1D(self._Y, self._T);

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        n = self._T.size;

        dX = self._Y.reshape((n, -1));
        dX[np.arange(n), self._T.flatten()] -= 1;
        dX /= n;

        return dX.reshape(self._Y.shape), ;


class SigmoidWithCrossEntropyLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._Y = None;
        self._T = None;


    def forward(self, *data: np.ndarray) -> float:
        X, T = data;
        self._T = T;
        self._Y = sigmoid(X);

        Y = self._Y.flatten();
        T = self._T.flatten();
        self._loss = crossEntropyError(np.column_stack((Y, 1 - Y)),
                                       np.column_stack((T, 1 - T)));

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        dX = (self._Y - self._T) / self._T.size;

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
        dX = (self._Y - self._T) / lengthExceptLastDimension(self._T);

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
        dY = (self._Y - self._T) / lengthExceptLastDimension(self._T);
        dX = dY * np.ones_like(self._shape);

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


class SequentialContainer(NetModuleBase, INetModel):
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


    def reset(self):
        for m in self._modules:
            m.reset();


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        for m in self._modules:
            data = m.forward(*data);

        return data;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        for m in reversed(self._modules):
            dout = m.backward(*dout);

        return dout;


    def getFinalTag(self, T : np.ndarray) -> np.ndarray:
        return T;


    def apply(self, func : Callable):
        for m in self._modules:
            func(m);


class SequentialDataIterator(IDataIterator):
    def __init__(self, data : List[np.ndarray], batchSize : int = 2 ** 8, shuffle : bool = True):
        self._step = 0;
        self._data = data;
        self._length = len(data[0]);
        self._batchSize = batchSize;
        self._totalIterations = self._length // self._batchSize + int(self._length % self._batchSize > 0);
        self._shuffle = shuffle;
        self._index = np.arange(self._length);


    def _iterate(self):
        while self._step * self._batchSize < self._length:
            startIndex = self._step * self._batchSize;
            index = self._index[startIndex: startIndex + self._batchSize];
            self._step += 1;

            yield tuple([d[index] for d in self._data]);


    def __iter__(self):
        self._step = 0;
        if self._shuffle:
            np.random.shuffle(self._index);

        return self._iterate();


    @property
    def totalIterations(self) -> int:
        return self._totalIterations;


class PartitionedDataIterator(IDataIterator):
    def __init__(self, data : List[np.ndarray], partitionNumber : int, batchSize : int, shuffle : bool = False):
        self._step = 0;
        self._data = data;
        self._length = len(data[0]);
        self._partitionNumber = partitionNumber;
        self._partitionSize = self._length // partitionNumber;
        self._batchSize = batchSize;
        self._totalIterations = self._partitionSize // self._batchSize + int(self._partitionSize % self._batchSize > 0);
        self._shuffle = shuffle;
        self._index = list(range(self._length));


    def _iterate(self):
        index = [];

        while self._step * self._batchSize < self._partitionSize:
            index.clear();
            batchSize = min(self._batchSize, self._partitionSize - self._step * self._batchSize);

            for i in range(self._partitionNumber):
                startIndex = i * self._partitionSize + self._step * self._batchSize;
                index.extend(self._index[startIndex: startIndex + batchSize]);

            self._step += 1;

            yield tuple([d[index].reshape((self._partitionNumber, -1) + d.shape[1:]) for d in self._data]);


    def __iter__(self):
        self._step = 0;
        if self._shuffle:
            np.random.shuffle(self._index);

        return self._iterate();


    @property
    def totalIterations(self) -> int:
        return self._totalIterations;


class ClassifierAccuracyEvaluator(INetAccuracyEvaluator):
    def __init__(self):
        self._rightCount = 0.0;
        self._totalCount = 0.0;


    @property
    def name(self) -> str:
        return "classification accuracy";


    @property
    def accuracy(self) -> float:
        return self._rightCount / self._totalCount;


    def fromLoss(self, lossValues : List[float]) -> bool:
        return False;


    def update(self, *data: np.ndarray):
        Y, T = data;
        self._rightCount += int(np.sum(np.argmax(Y, -1) == np.argmax(T, -1)));
        self._totalCount += Y.size / Y.shape[-1];


    def reset(self):
        self._rightCount = 0.0;
        self._totalCount = 0.0;


class PerplexityAccuracyEvaluator(INetAccuracyEvaluator):
    def __init__(self):
        self._perplexity = 1.0;


    @property
    def name(self) -> str:
        return "perplexity";


    @property
    def accuracy(self) -> float:
        return self._perplexity;


    def fromLoss(self, lossValues : List[float]) -> bool:
        self._perplexity = math.exp(sum(lossValues) / len(lossValues));
        return True;


    def update(self, *data: np.ndarray):
        raise NotImplemented();


    def reset(self):
        self._perplexity = 1.0;


class NetTrainer:
    def __init__(self, model : INetModel, lossFunc : INetLoss, optimizer : INetOptimizer, evaluator : INetAccuracyEvaluator = None):
        self._model = model;
        self._lossFunc = lossFunc;
        self._optimizer = optimizer;
        self._evaluator = evaluator;

        self._lossData = [];
        self._trainAccuracyData = [];
        self._testAccuracyData = [];


    def _calcAccuracy(self, lossValues : List[float], iterator : Iterable) -> float:
        self._model.isTrainingMode = False;

        try:
            self._evaluator.reset();

            if not self._evaluator.fromLoss(lossValues):
                for data in iterator:
                    Y = self._model.forward(*data[:-1]);
                    self._evaluator.update(*Y, self._model.getFinalTag(data[-1]));
        finally:
            self._model.reset();
            self._model.isTrainingMode = True;

        return self._evaluator.accuracy;


    def train(self, maxEpoch : int, trainIterator : IDataIterator, testIterator : IDataIterator = None):
        lossValues = [];

        self._lossData.clear();
        self._trainAccuracyData.clear();
        self._testAccuracyData.clear();

        print(f"[{datetime.datetime.now()}] start to train model {self._model}");

        for epoch in range(maxEpoch):
            lossValues.clear();
            startTime = time.time();

            for data in trainIterator:
                Y = self._model.forward(*data[:-1]);
                loss = self._lossFunc.forward(*Y, self._model.getFinalTag(data[-1]));
                lossValues.append(loss);

                self._model.backward(*self._lossFunc.backward());
                self._optimizer.update(self._model.params, self._model.grads);

            self._model.reset();
            self._lossData.append(sum(lossValues) / len(lossValues));

            if self._evaluator is not None:
                self._trainAccuracyData.append(self._calcAccuracy(lossValues, trainIterator));
                if testIterator is not None:
                    self._testAccuracyData.append(self._calcAccuracy(lossValues, testIterator));

            elapsedTime = time.time() - startTime;
            if len(self._trainAccuracyData) > 0 and len(self._testAccuracyData) > 0:
                print(f"epoch {epoch}, average loss: {lossValues[-1]}, train {self._evaluator.name}: {self._trainAccuracyData[-1]}, test  {self._evaluator.name}: {self._testAccuracyData[-1]}, elapsed time: {elapsedTime}s");
            elif len(self._trainAccuracyData) > 0:
                print(f"epoch {epoch}, average loss: {lossValues[-1]}, train {self._evaluator.name}: {self._trainAccuracyData[-1]}, elapsed time: {elapsedTime}s");
            else:
                print(f"epoch {epoch}, average loss: {lossValues[-1]}, elapsed time: {elapsedTime}s");


        print(f"[{datetime.datetime.now()}] complete to train model {self._model}");


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
