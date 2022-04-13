import abc;
import math;
import time;
import datetime;
import collections;

import matplotlib.pyplot as plt;

from typing import Union, List, Tuple, Callable, Any, Optional, Iterable;
from Functions import *;


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


class IDataScaler(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, X : np.ndarray) -> np.ndarray:
        pass;


    @abc.abstractmethod
    def transform(self, X : np.ndarray) -> np.ndarray:
        pass;


    @abc.abstractmethod
    def inverse(self, Y : np.ndarray, *args, **kwargs) -> np.ndarray:
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
    def accuracy(self) -> Optional[float]:
        pass;


    @abc.abstractmethod
    def fromLoss(self, lossValues : List[float] = None) -> bool:
        pass;


    @abc.abstractmethod
    def update(self, *data : np.ndarray):
        pass;


    @abc.abstractmethod
    def reset(self):
        pass;


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

    @params.setter
    @abc.abstractmethod
    def params(self, value : List[np.ndarray]):
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


    @abc.abstractmethod
    def fit(self, trainingIterator : IDataIterator, lossFunc: INetLoss, optimizer: INetOptimizer, maxEpoch : int, testIterator : IDataIterator = None,
            evaluator: INetAccuracyEvaluator = None, evalEpoch : bool = True, evalIterations : int = None, evalTrainingData : bool = False, evalTestData : bool = True,
            plot = False):
        pass;


    @abc.abstractmethod
    def predict(self, iterator : IDataIterator) -> Iterable:
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


    @params.setter
    def params(self, value: List[np.ndarray]):
        self._params = value;
        self._setParams(value);


    @property
    def grads(self) -> List[np.ndarray]:
        return self._grads;


    def _setTrainingMode(self, value : bool):
        pass;


    def _setParams(self, value: List[np.ndarray]):
        pass;


    def reset(self):
        pass;


class AggregateNetModule(NetModuleBase, metaclass = abc.ABCMeta):
    def __init__(self, *modules : INetModule):
        super().__init__();

        self._modules = modules;
        for m in modules:
            self._params.extend(m.params);
            self._grads.extend(m.grads);


    @property
    def modules(self) -> Tuple[INetModule]:
        return self._modules;


    def _setTrainingMode(self, value: bool):
        for m in self._modules:
            m.isTrainingMode = value;


    def _setParams(self, value: List[np.ndarray]):
        i = 0;

        for m in self._modules:
            if (n := len(m.params)) == 0:
                continue;

            m.params = value[i: i + n];
            i += n;


    def reset(self):
        for m in self._modules:
            m.reset();


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        Y = data;

        for m in self._modules:
            Y = m.forward(*Y);

        return Y;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dX = dout;

        for m in reversed(self._modules):
            dX = m.backward(*dX);

        return dX;


class NetModelBase(AggregateNetModule, INetModel, metaclass = abc.ABCMeta):
    def __init__(self, *modules : INetModule):
        super().__init__(*modules);


    def _calcAccuracy(self, lossFunc : INetLoss, optimizer : INetOptimizer, evaluator : INetAccuracyEvaluator, lossValues : List[float] = None, iterator : Iterable = None) -> float:
        self.isTrainingMode = False;

        try:
            evaluator.reset();

            if not (lossValues is not None and evaluator.fromLoss(lossValues)) and iterator is not None:
                lossValues = [];

                for data in iterator:
                    Y = self.forward(*data);
                    loss = lossFunc.forward(*Y, self.getFinalTag(data[-1]));

                    lossValues.append(loss);
                    evaluator.update(*Y, self.getFinalTag(data[-1]));

                evaluator.fromLoss(lossValues);
        finally:
            self.reset();
            self.isTrainingMode = True;

        return evaluator.accuracy;


    def getFinalTag(self, T : np.ndarray) -> np.ndarray:
        return T;


    def fit(self, trainingIterator: IDataIterator, lossFunc: INetLoss, optimizer: INetOptimizer, maxEpoch: int, testIterator: IDataIterator = None,
            evaluator: INetAccuracyEvaluator = None, evalEpoch: bool = True, evalIterations: int = None, evalTrainingData: bool = False, evalTestData: bool = True,
            plot = False):
        lossData = [];
        lossValues = [];
        trainingAccuracyData = [];
        testAccuracyData = [];

        startTime = time.time();
        print(f"[{datetime.datetime.now()}] start to train model {self}");

        for epoch in range(maxEpoch):
            lossValues.clear();

            for data in trainingIterator:
                Y = self.forward(*data);
                loss = lossFunc.forward(*Y, self.getFinalTag(data[-1]));
                lossValues.append(loss);

                self.backward(*lossFunc.backward());
                optimizer.update(self.params, self.grads);

                if evaluator is not None and evalIterations is not None and len(lossValues) % evalIterations == 0:
                    accuracy = self._calcAccuracy(lossFunc, optimizer, evaluator, lossValues[-evalIterations:], None);
                    if accuracy is not None:
                        print(f"epoch {epoch}, iterations: {len(lossValues)} / {trainingIterator.totalIterations}, elapsed time: {int(time.time() - startTime)}s, training {evaluator.name}: {accuracy}");

            self.reset();
            lossData.append(sum(lossValues) / len(lossValues));

            if evaluator is not None and evalEpoch:
                if evalTrainingData:
                    print("evaluating training data...");
                    trainingAccuracyData.append(self._calcAccuracy(lossFunc, optimizer, evaluator, lossValues, trainingIterator));
                if testIterator is not None and evalTestData:
                    print("evaluating test data...");
                    testAccuracyData.append(self._calcAccuracy(lossFunc, optimizer, evaluator, None, testIterator,));

            trainingMessage = f", training {evaluator.name}: {trainingAccuracyData[-1]}" if len(trainingAccuracyData) > 0 else "";
            testMessage = f", test {evaluator.name}: {testAccuracyData[-1]}" if len(testAccuracyData) > 0 else "";
            print(f"epoch {epoch}, average loss: {lossData[-1]}{trainingMessage}{testMessage}, elapsed time: {int(time.time() - startTime)}s");

        if evaluator is not None:
            print("evaluating final training data...");
            print(f"the final training {evaluator.name} is {self._calcAccuracy(lossFunc, optimizer, evaluator, None, trainingIterator)}");

            if testIterator is not None:
                print("evaluating final test data...");
                print(f"the final test {evaluator.name} is {self._calcAccuracy(lossFunc, optimizer, evaluator, None, testIterator)}");

        print(f"[{datetime.datetime.now()}] complete to train model, elapsed time: {int(time.time() - startTime)}s");

        if plot:
            fig = plt.figure(1);

            ax1 = fig.add_subplot(111);
            ax1.set_xlabel("epoch");
            ax1.set_ylabel('loss');
            ax1.plot(lossData, "o-k", label = "loss");

            ax2 = ax1.twinx();
            ax2.set_ylabel('accuracy');
            if len(trainingAccuracyData) > 0:
                ax2.plot(trainingAccuracyData, "D-b", label = "training accuracy");
            if len(testAccuracyData) > 0:
                ax2.plot(testAccuracyData, "s-r", label = "test accuracy");

            fig.legend(loc = "upper right", bbox_to_anchor = (1, 1), bbox_transform = ax1.transAxes)
            plt.show(block = True);
            plt.close();


    def predict(self, iterator : IDataIterator) -> Iterable:
        for data in iterator:
            yield self.forward(*data);


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
        self._mask = (X > 0).astype(X.dtype);

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


    def _getMask(self, shape : Tuple, dtype) -> np.ndarray:
        return (np.random.rand(*shape) > self._dropoutRatio).astype(dtype) / (1.0 - self._dropoutRatio);


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];

        if self._isTrainingMode:
            self._mask = self._getMask(X.shape, X.dtype);
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


class VariationalDropoutLayer(DropoutLayer):
    def __init__(self, dropoutRatio = 0.5):
        super().__init__(dropoutRatio);

        self._name = f"VariationalDropout {dropoutRatio}";


    def _getMask(self, shape : Tuple, dtype) -> np.ndarray:
        D = len(shape);
        if D <= 2:
            return super()._getMask(shape, dtype);

        mask = super()._getMask((shape[0], shape[-1]), dtype);
        for _ in range(D - 2):
            mask = np.expand_dims(mask, axis = -2);
        for d in range(1, D - 1):
            mask = np.repeat(mask, shape[d], axis = d);

        return mask;


class ReshapeLayer(NetModuleBase):
    def __init__(self, *shapeSelector : Union[Tuple, Callable]):
        super().__init__();

        self._originalShapes = [];
        self._shapeSelector = shapeSelector;
        self._name = "Reshape";


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray]:
        self._originalShapes.clear();
        output : List[np.ndarray] = [];

        for X, selector in zip(data, self._shapeSelector):
            self._originalShapes.append(X.shape);
            output.append(X.reshape(selector if isinstance(selector, tuple) else selector(X)));

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
        self._shape = None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._includeBias = includeBias;
        self._name = f"Affine {inputSize}*{outputSize}";

        self._weight = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, outputSize).astype(defaultDType) if W is None else W;
        self._bias = (np.zeros(outputSize, dtype = defaultDType) if b is None else b) if includeBias else None;

        self._params.append(self._weight);
        self._grads.append(np.zeros_like(self._weight));
        if self._bias is not None:
            self._params.append(self._bias);
            self._grads.append(np.zeros_like(self._bias));


    def _setParams(self, value: List[np.ndarray]):
        self._weight, self._bias = value[0], value[1] if self._includeBias else None;


    @property
    def weight(self):
        return self._weight;


    @property
    def bias(self):
        return self._bias;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];

        if X.ndim > 2:
            self._shape = X.shape;
            self._X = np.reshape(X, (-1, X.shape[-1]));
        else:
            self._shape = None;
            self._X = X;

        Y = self._X @ self._weight;
        if self._bias is not None:
            Y += self._bias;

        if self._shape is not None:
            Y = np.reshape(Y, self._shape[:-1] + (Y.shape[-1], ));

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        if self._shape is not None:
            dY = np.reshape(dY, (-1, dY.shape[-1]));

        dW = self._X.T @ dY;
        self._grads[0][...] = dW;

        if self._bias is not None:
            db = np.sum(dY, 0);
            self._grads[1][...] = db;

        dX = dY @ self._weight.T;
        if self._shape is not None:
            dX = np.reshape(dX, self._shape);

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

        self._gamma = np.ones(inputSize, dtype = defaultDType) if gamma is None else gamma;
        self._beta = np.zeros(inputSize, dtype = defaultDType) if beta is None else beta;

        weights = [self._gamma, self._beta];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    def _setParams(self, value: List[np.ndarray]):
        self._gamma, self._beta = value[0], value[1];


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

        self._weight = math.sqrt(2.0 / (C * FH * FW)) * np.random.randn(FN, C, FH, FW).astype(defaultDType) if W is None else W;
        self._bias = np.zeros(FN, dtype = defaultDType) if b is None else b;

        weights = [self._weight, self._bias];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    def _setParams(self, value: List[np.ndarray]):
        self._weight, self._bias = value[0], value[1];


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
        dMax = np.zeros((colDY.size, poolingSize), dtype = dY.dtype);
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

        self._weight = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, outputSize).astype(defaultDType) if W is None else W;
        self._params.append(self._weight);
        self._grads.append(np.zeros_like(self._weight));


    def _setParams(self, value: List[np.ndarray]):
        self._weight = value[0];


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
        return np.zeros(self._shape, dtype = dY.dtype), ;


class EmbeddingWithDotLayer(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, W : np.ndarray = None):
        super().__init__();

        self._X = None;
        self._W = None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"EmbeddingWithDot {outputSize}*{inputSize}";

        weight = math.sqrt(2.0 / inputSize) * np.random.randn(outputSize, inputSize).astype(defaultDType) if W is None else W;
        self._embeddingLayer = EmbeddingLayer(outputSize, inputSize, W = weight);


    @property
    def params(self) -> List[np.ndarray]:
        return self._embeddingLayer.params;


    @params.setter
    def params(self, value: List[np.ndarray]):
        self._embeddingLayer.params = value;


    @property
    def grads(self) -> List[np.ndarray]:
        return self._embeddingLayer.grads;


    @property
    def weight(self):
        return self._embeddingLayer.weight;


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

        self._weightX = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, outputSize).astype(defaultDType) if Wx is None else Wx;
        self._weightH = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, outputSize).astype(defaultDType) if Wh is None else Wh;
        self._bias = np.zeros(outputSize, dtype = defaultDType) if b is None else b;

        weights = [self._weightX, self._weightH, self._bias];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    def _setParams(self, value: List[np.ndarray]):
        self._weightX, self._weightH, self._bias = value[0], value[1], value[2];


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

        self._weightX, self._weightH, self._bias = self._initParams(inputSize, outputSize, Wx, Wh, b);
        self._rnnModules : List[INetModule] = [];

        weights = [self._weightX, self._weightH, self._bias];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    def _setParams(self, value: List[np.ndarray]):
        self._weightX, self._weightH, self._bias = value[0], value[1], value[2];
        for cell in self._rnnModules:
            cell.params = value;


    def _initParams(self, inputSize : int, outputSize : int, Wx : np.ndarray, Wh : np.ndarray, b : np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        return (
            math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, outputSize).astype(defaultDType) if Wx is None else Wx,
            math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, outputSize).astype(defaultDType) if Wh is None else Wh,
            np.zeros(outputSize, dtype = defaultDType) if b is None else b
        );


    def _getCell(self, inputSize : int, outputSize : int, Wx : np.ndarray, Wh : np.ndarray, b : np.ndarray) -> INetModule:
        return RnnCell(inputSize, outputSize, Wx, Wh, b);


    @property
    def weightX(self) -> np.ndarray:
        return self._weightX;


    @property
    def weightH(self) -> np.ndarray:
        return self._weightH;


    @property
    def bias(self) -> np.ndarray:
        return self._bias;


    @property
    def dH(self) -> np.ndarray:
        return self._dH;


    def reset(self):
        self._H = None;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        N, T, D = X.shape;

        if len(self._rnnModules) != T:
            self._rnnModules = [self._getCell(self._inputSize, self._outputSize, self._weightX, self._weightH, self._bias) for _ in range(T)];

        if not self._stateful or self._H is None:
            self._H = np.zeros((N, self._outputSize), X.dtype);

        Y = np.zeros((N, T, self._outputSize), X.dtype);
        for t in range(T):
            self._H = self._rnnModules[t].forward(X[:, t, :], self._H)[0];
            Y[:, t, :] = self._H;

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        N, T = dY.shape[: 2];

        # truncated BPTT
        self._dH = np.zeros_like(self._H);
        dX = np.zeros((N, T, self._inputSize), dY.dtype);
        for i in range(len(self._grads)):
            self._grads[i][...] = 0;

        for t in reversed(range(T)):
            rnn = self._rnnModules[t];
            dX[:, t, :], self._dH = rnn.backward(dY[:, t, :] + self._dH);

            for i in range(len(self._grads)):
                self._grads[i] += rnn.grads[i];

        return dX, ;


    def setState(self, H : np.ndarray):
        self._H = H;


class LstmCell(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, Wx : np.ndarray = None, Wh : np.ndarray = None, b : np.ndarray = None):
        super().__init__();

        self._X, self._H, self._C = None, None, None;
        self._F, self._G, self._I, self._O = None, None, None, None;
        self._YC, self._tanhYC, self._YH = None, None, None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"LSTM Cell {inputSize}*{outputSize}";

        self._weightX = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, 4 * outputSize).astype(defaultDType) if Wx is None else Wx;
        self._weightH = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, 4 * outputSize).astype(defaultDType) if Wh is None else Wh;
        self._bias = np.zeros(4 * outputSize, dtype = defaultDType) if b is None else b;

        weights = [self._weightX, self._weightH, self._bias];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    def _setParams(self, value: List[np.ndarray]):
        self._weightX, self._weightH, self._bias = value[0], value[1], value[2];


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
        self._X, self._H, self._C = data;

        A = self._X @ self._weightX + self._H @ self._weightH + self._bias;
        self._F, self._G, self._I, self._O = tuple(np.hsplit(A, 4));
        self._F, self._G, self._I, self._O = sigmoid(self._F), tanh(self._G), sigmoid(self._I), sigmoid(self._O);

        self._YC = self._C * self._F + self._G * self._I;
        self._tanhYC = tanh(self._YC);
        self._YH = self._tanhYC * self._O;

        return self._YH, self._YC;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dYH, dYC = dout;

        dYC += dYH * self._O * tanhGradient(self._tanhYC);
        dF, dG, dI, dO = dYC * self._C, dYC * self._I, dYC * self._G, dYH * self._tanhYC;
        dF *= sigmoidGradient(self._F);
        dG *= tanhGradient(self._G);
        dI *= sigmoidGradient(self._I);
        dO *= sigmoidGradient(self._O);
        dA = np.hstack((dF, dG, dI, dO));

        dWx = self._X.T @ dA;
        dWh = self._H.T @ dA;
        db = np.sum(dA, axis = 0);

        dX = dA @ self._weightX.T;
        dH = dA @ self._weightH.T;
        dC = dYC * self._F;

        self._grads[0][...] = dWx;
        self._grads[1][...] = dWh;
        self._grads[2][...] = db;

        return dX, dH, dC;


class LstmLayer(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, Wx : np.ndarray = None, Wh : np.ndarray = None, b : np.ndarray = None, returnSequences : bool = False, stateful : bool = False):
        super().__init__();

        self._T = 0;
        self._H, self._C = None, None;
        self._dH, self._dC = None, None;
        self._returnSequences = returnSequences;
        self._stateful = stateful;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"LSTM {inputSize}*{outputSize}";

        self._weightX = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, 4 * outputSize).astype(defaultDType) if Wx is None else Wx;
        self._weightH = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, 4 * outputSize).astype(defaultDType) if Wh is None else Wh;
        self._bias = np.zeros(4 * outputSize, dtype = defaultDType) if b is None else b;
        self._lstmModules : List[LstmCell] = [];

        weights = [self._weightX, self._weightH, self._bias];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    def _setParams(self, value: List[np.ndarray]):
        self._weightX, self._weightH, self._bias = value[0], value[1], value[2];
        for cell in self._lstmModules:
            cell.params = value;


    @property
    def weightX(self) -> np.ndarray:
        return self._weightX;


    @property
    def weightH(self) -> np.ndarray:
        return self._weightH;


    @property
    def bias(self) -> np.ndarray:
        return self._bias;


    @property
    def dH(self) -> np.ndarray:
        return self._dH;


    @property
    def dC(self) -> np.ndarray:
        return self._dC;


    def reset(self):
        self._H, self._C = None, None;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        N, T, D = X.shape;

        self._T = T;
        if len(self._lstmModules) != T:
            self._lstmModules = [LstmCell(self._inputSize, self._outputSize, self._weightX, self._weightH, self._bias) for _ in range(T)];

        if not self._stateful or self._H is None:
            self._H = np.zeros((N, self._outputSize), dtype = X.dtype);
        if not self._stateful or self._C is None:
            self._C = np.zeros((N, self._outputSize), dtype = X.dtype);

        Y = np.zeros((N, T, self._outputSize), dtype = X.dtype);
        for t in range(T):
            self._H, self._C = self._lstmModules[t].forward(X[:, t, :], self._H, self._C);
            Y[:, t, :] = self._H;

        return Y if self._returnSequences else Y[:, -1, :], ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        N, T = len(dY), self._T;

        if not self._returnSequences:
            dH = dY;
            dY = np.zeros((N, T, dY.shape[-1]), dtype = dY.dtype);
            dY[:, -1, :] = dH;

        # truncated BPTT
        self._dH = np.zeros_like(self._H);
        self._dC = np.zeros_like(self._H);
        dX = np.zeros((N, T, self._inputSize), dtype = dY.dtype);
        for i in range(len(self._grads)):
            self._grads[i][...] = 0;

        for t in reversed(range(T)):
            lstm = self._lstmModules[t];
            dX[:, t, :], self._dH, self._dC = lstm.backward(dY[:, t, :] + self._dH, self._dC);

            for i in range(len(self._grads)):
                self._grads[i] += lstm.grads[i];

        return dX, ;


    def setState(self, H : np.ndarray, C : np.ndarray = None):
        self._H, self._C = H, C;


class GruCell(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, Wx : np.ndarray = None, Wh : np.ndarray = None, b : np.ndarray = None):
        super().__init__();

        self._X, self._H = None, None;
        self._R, self._Z, self._C, self._Y = None, None, None, None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"GRU Cell {inputSize}*{outputSize}";

        self._weightX = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, 3 * outputSize).astype(defaultDType) if Wx is None else Wx;
        self._weightH = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, 3 * outputSize).astype(defaultDType) if Wh is None else Wh;
        self._bias = np.zeros(3 * outputSize, dtype = defaultDType) if b is None else b;

        weights = [self._weightX, self._weightH, self._bias];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    def _setParams(self, value: List[np.ndarray]):
        self._weightX, self._weightH, self._bias = value[0], value[1], value[2];


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

        Wxr, Wxz, Wxh = tuple(np.hsplit(self._weightX, 3));
        Whr, Whz, Whh = tuple(np.hsplit(self._weightH, 3));
        br, bz, bh = tuple(np.hsplit(self._bias, 3));

        self._R = sigmoid(self._X @ Wxr + self._H @ Whr + br);
        self._Z = sigmoid(self._X @ Wxz + self._H @ Whz + bz);
        self._C = tanh(self._X @ Wxh + (self._H * self._R) @ Whh + bh);
        self._Y = self._H * self._Z + self._C * (1 - self._Z);

        return self._Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        Wxr, Wxz, Wxh = tuple(np.hsplit(self._weightX, 3));
        Whr, Whz, Whh = tuple(np.hsplit(self._weightH, 3));

        dC = dY * (1 - self._Z) * tanhGradient(self._C);
        dZ = dY * (self._H - self._C) * sigmoidGradient(self._Z);
        dR = dC @ Whh.T * self._H * sigmoidGradient(self._R);

        dWxr, dWhr, dbr = self._X.T @ dR, self._H.T @ dR, np.sum(dR, axis = 0);
        dWxz, dWhz, dbz = self._X.T @ dZ, self._H.T @ dZ, np.sum(dZ, axis = 0);
        dWxh, dWhh, dbh = self._X.T @ dC, (self._H * self._R).T @ dC, np.sum(dC, axis = 0);

        dX = dC @ Wxh.T + dZ @ Wxz.T + dR @ Wxr.T;
        dH = dY * self._Z + dC @ Whh.T * self._R + dZ @ Whz.T + dR @ Whr.T;

        self._grads[0][...] = np.hstack((dWxr, dWxz, dWxh));
        self._grads[1][...] = np.hstack((dWhr, dWhz, dWhh));
        self._grads[2][...] = np.hstack((dbr, dbz, dbh));

        return dX, dH;


class GruLayer(RnnLayer):
    def __init__(self, inputSize : int, outputSize : int, Wx : np.ndarray = None, Wh : np.ndarray = None, b : np.ndarray = None, stateful : bool = True):
        super().__init__(inputSize, outputSize, Wx, Wh, b, stateful);

        self._name = f"GRU {inputSize}*{outputSize}";


    def _initParams(self, inputSize : int, outputSize : int, Wx : np.ndarray, Wh : np.ndarray, b : np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        return (
            math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, 3 * outputSize).astype(defaultDType) if Wx is None else Wx,
            math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, 3 * outputSize).astype(defaultDType) if Wh is None else Wh,
            np.zeros(3 * outputSize, dtype = defaultDType) if b is None else b
        );


    def _getCell(self, inputSize : int, outputSize : int, Wx : np.ndarray, Wh : np.ndarray, b : np.ndarray) -> INetModule:
        return GruCell(inputSize, outputSize, Wx, Wh, b);


class BiRnnLayer(AggregateNetModule):
    def __init__(self, forwardRnn : INetModule, backwardRnn : INetModule):
        super().__init__(forwardRnn, backwardRnn);
        self._forwardRnn = forwardRnn;
        self._backwardRnn = backwardRnn;


    @property
    def forwardRnn(self) -> INetModule:
        return self._forwardRnn;


    @property
    def backwardRnn(self) -> INetModule:
        return self._backwardRnn;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        Y1 = self._forwardRnn.forward(X)[0];
        Y2 = self._backwardRnn.forward(X[..., ::-1, :])[0];
        return Y1, Y2[..., ::-1, :];


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY1, dY2 = dout;
        dX1 = self._forwardRnn.backward(dY1)[0];
        dX2 = self._backwardRnn.backward(dY2[..., ::-1, :])[0];
        dX = dX1 + dX2[..., ::-1, :];
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

        return np.random.choice(self._vocab, sampleSize, replace = DeviceConfig.enableGPU or sampleSize >= self._vocabSize, p = p);


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


class CBOWModel(NetModelBase):
    def __init__(self, windowSize : int, vocabSize : int, hiddenSize : int, negativeSampler : CorpusNegativeSampler, inW : np.ndarray = None, outW : np.ndarray = None):
        self._finalTag = None;
        self._windowSize = windowSize;
        self._vocabSize = vocabSize;
        self._hiddenSize = hiddenSize;
        self._negativeSampler = negativeSampler;
        self._W0 = math.sqrt(2.0 / vocabSize) * np.random.randn(vocabSize, hiddenSize).astype(defaultDType) if inW is None else inW;
        self._W1 = math.sqrt(2.0 / hiddenSize) * np.random.randn(hiddenSize, vocabSize).astype(defaultDType).T if outW is None else outW;

        self._embeddingLayer = EmbeddingLayer(vocabSize, hiddenSize, W = self._W0);
        self._outputLayer = EmbeddingWithDotLayer(hiddenSize, vocabSize, W = self._W1);

        super().__init__(self._embeddingLayer, self._outputLayer);
        self._name = "CBOW";


    def _setParams(self, value: List[np.ndarray]):
        super()._setParams(value);
        self._W0, self._W1 = value[0], value[1];


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

        return np.zeros((N, C), dtype = dY.dtype), ;


    def getFinalTag(self, T : np.ndarray) -> np.ndarray:
        return self._finalTag;


class SkipGramModel(NetModelBase):
    def __init__(self, windowSize : int, vocabSize : int, hiddenSize : int, negativeSampler : CorpusNegativeSampler, inW : np.ndarray = None, outW : np.ndarray = None):
        self._finalTag = None;
        self._windowSize = windowSize;
        self._vocabSize = vocabSize;
        self._hiddenSize = hiddenSize;
        self._negativeSampler = negativeSampler;
        self._W0 = math.sqrt(2.0 / vocabSize) * np.random.randn(vocabSize, hiddenSize).astype(defaultDType) if inW is None else inW;
        self._W1 = math.sqrt(2.0 / hiddenSize) * np.random.randn(hiddenSize, vocabSize).astype(defaultDType).T if outW is None else outW;

        self._embeddingLayer = EmbeddingLayer(vocabSize, hiddenSize, W = self._W0);
        self._outputLayer = EmbeddingWithDotLayer(hiddenSize, vocabSize, W = self._W1);

        super().__init__(self._embeddingLayer, self._outputLayer);
        self._name = "SkipGram";


    def _setParams(self, value: List[np.ndarray]):
        super()._setParams(value);
        self._W0, self._W1 = value[0], value[1];


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
        dY = -(self._T / self._Y).astype(self._Y.dtype) / lengthExceptLastDimension(self.T);

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
        dX = (self._Y - self._T).astype(self._Y.dtype) / lengthExceptLastDimension(self._T);

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
        dX = (self._Y - self._T).astype(self._Y.dtype) / self._T.size;

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


class _ParametersShareInfo:
    def __init__(self, index : int, target : int, isTranspose : bool = False):
        self.index = index;
        self.target = target;
        self.isTranspose = isTranspose;


    def __repr__(self):
        return self.__str__();


    def __str__(self):
        return f"[{self.target}] += [{self.index}]{'.T' if self.isTranspose else ''}";


class ParametersShare(INetOptimizer):
    def __init__(self, optimizer : INetOptimizer):
        self._optimizer = optimizer;
        self._sharesInfo: Optional[List[_ParametersShareInfo]] = None;


    def learningRate(self) -> float:
        return self._optimizer.learningRate;


    def _find(self, params : List[np.ndarray]):
        L = len(params);
        sharesInfo = [];
        params = params[:];

        for i in range(L - 1):
            if (p1 := params[i]) is None:
                continue;

            for j in range(i + 1, L):
                if (p2 := params[j]) is None:
                    continue;

                if p1 is p2:
                    # p1 == p2
                    sharesInfo.append(_ParametersShareInfo(j, i));
                    params[j] = None;
                elif p1.ndim == 2 and p2.ndim == 2 and (p1 is p2.base or p2 is p1.base):
                    # p1 == p2.T or p1.T == p2
                    s1, s2 = p1.shape, p2.shape;

                    if s1[0] == s2[1] and s1[1] == s2[0]:
                        if p1 is p2.base:
                            sharesInfo.append(_ParametersShareInfo(j, i, isTranspose = True));
                            params[j] = None;
                        else:
                            sharesInfo.append(_ParametersShareInfo(i, j, isTranspose = True));
                            params[i] = None;
                            break;

        self._sharesInfo = sorted(sharesInfo, key = lambda item: item.index, reverse = True);


    def update(self, params : List[np.ndarray], grads : List[np.ndarray]):
        L = len(params);
        params, grads = params[:], grads[:];

        if self._sharesInfo is None:
            self._find(params);

        for info in self._sharesInfo:
            grads[info.target] += (grads[info.index].T if info.isTranspose else grads[info.index]);
            params.pop(info.index);
            grads.pop(info.index);

        self._optimizer.update(params, grads);


class GradientsClipping(INetOptimizer):
    def __init__(self, maxL2 : float, optimizer : INetOptimizer, epsilon : float = 1e-8):
        self._maxL2 = maxL2;
        self._optimizer = optimizer;
        self._epsilon = epsilon;


    def learningRate(self) -> float:
        return self._optimizer.learningRate;


    def update(self, params : List[np.ndarray], grads : List[np.ndarray]):
        totalL2 = sum([float(np.sum(g ** 2)) for g in grads]);
        ratio = self._maxL2 / math.sqrt(totalL2 + self._epsilon);

        if ratio < 1:
            for g in grads:
                g *= ratio;

        self._optimizer.update(params, grads);


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


class AggregateScaler(IDataScaler):
    def __init__(self, *scalers : IDataScaler):
        self._scalers = scalers;


    def fit(self, X : np.ndarray):
        Y = X;
        for scaler in self._scalers:
            Y = scaler.fit(Y);


    def transform(self, X : np.ndarray) -> np.ndarray:
        Y = X;

        for scaler in self._scalers:
            Y = scaler.transform(Y);

        return Y;


    def inverse(self, Y : np.ndarray, *args, **kwargs) -> np.ndarray:
        X = Y;

        for scaler in reversed(self._scalers):
            X = scaler.inverse(X, *args, **kwargs);

        return X;


class ScalerBase(IDataScaler):
    def __init__(self):
        self._ndim = None;
        self._fitted = False;


    @abc.abstractmethod
    def _fit(self, X: np.ndarray) -> np.ndarray:
        pass;


    @abc.abstractmethod
    def _transform(self, X : np.ndarray) -> np.ndarray:
        pass;


    @abc.abstractmethod
    def _inverse(self, Y: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass;


    def _check(self):
        if not self._fitted:
            raise ValueError("scaler has not fitted");


    def fit(self, X: np.ndarray) -> np.ndarray:
        self._ndim = 1 if X.ndim <= 1 else 2;

        Y = self._fit((X.flatten() if self._ndim == 1 else X.reshape(-1, X.shape[-1])) if X.ndim > self._ndim else X);
        self._fitted = True;

        return Y;


    def transform(self, X : np.ndarray) -> np.ndarray:
        self._check();

        shape = X.shape;

        if X.ndim > self._ndim:
            X = X.flatten() if self._ndim == 1 else X.reshape(-1, shape[-1]);

        X = self._transform(X);

        return X.reshape((-1, ) + shape[1:]);


    def inverse(self, Y : np.ndarray, *args, **kwargs) -> np.ndarray:
        self._check();

        shape = Y.shape;

        if Y.ndim > self._ndim:
            Y = Y.flatten() if self._ndim == 1 else Y.reshape(-1, shape[-1]);

        Y = self._inverse(Y, *args, **kwargs);

        return Y.reshape((-1, ) + shape[1:]);


class MinMaxScaler(ScalerBase):
    def __init__(self, minRange : float = 0, maxRange : float = 1):
        super().__init__();

        self._minRange = minRange * 1.0;
        self._maxRange = maxRange * 1.0;
        self._rangeDelta = maxRange - minRange;
        self._minValue = None;
        self._maxValue = None;
        self._valueDelta = None;


    def _fit(self, X: np.ndarray) -> np.ndarray:
        self._minValue = np.amin(X, axis = 0);
        self._maxValue = np.amax(X, axis = 0);
        self._valueDelta = self._maxValue - self._minValue;
        return X;


    def _transform(self, X : np.ndarray) -> np.ndarray:
        return self._minRange + (X - self._minValue) / self._valueDelta * self._rangeDelta;


    def _inverse(self, Y : np.ndarray, *args, **kwargs) -> np.ndarray:
        return (Y - self._minRange) / self._rangeDelta * self._valueDelta + self._minValue;


class StandardScaler(ScalerBase):
    def __init__(self):
        super().__init__();

        self._mu = None;
        self._sigma = None;


    def _fit(self, X: np.ndarray):
        self._mu = np.mean(X, axis = 0);
        self._sigma = np.std(X, axis = 0);
        return X;


    def _transform(self, X : np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sigma;


    def _inverse(self, Y : np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._sigma * Y + self._mu;


class DiffScaler(ScalerBase):
    INDEX_ARGUMENT_NAME = "index";

    def __init__(self, interval : int = 1):
        super().__init__();

        self._X = None;
        self._interval = interval;


    def _fit(self, X: np.ndarray):
        self._X = X;
        return self._transform(X);


    def _transform(self, X : np.ndarray) -> np.ndarray:
        return X[self._interval:] - X[:-self._interval];


    def _inverse(self, Y : np.ndarray, *args, **kwargs) -> np.ndarray:
        index = int(kwargs[DiffScaler.INDEX_ARGUMENT_NAME]) if DiffScaler.INDEX_ARGUMENT_NAME in kwargs else 0;
        return np.concatenate((self._X[index: index + self._interval], self._X[index: index + len(Y)] + Y), axis = 0);


class SequentialContainer(NetModelBase):
    def __init__(self, *modules : INetModule):
        super().__init__(*modules);
        self._name = "  -->  ".join([str(m) for m in modules]);


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        return super().forward(*data[:-1]);


    def apply(self, func : Callable):
        for m in self._modules:
            func(m);


# select value by weights for 1 time step
class SelectByWeight1TModule(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._V = None;
        self._W = None;
        self._name = "SelectByWeight1T";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        # W: weights, N1 ... Nm * T1, V: values, N1 ...Nm * T1 * V
        W, V = data;
        self._V, self._W = V, np.expand_dims(W, axis = -1);
        Y = np.sum(self._V * self._W, axis = -2);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = np.repeat(np.expand_dims(dout[0], axis = -2), self._V.shape[-2], axis = -2);
        dV = dY * self._W;
        dW = np.sum(dY * self._V, axis = -1);

        return dW, dV;


# select value by weights for N time step
class SelectByWeightNTModule(SelectByWeight1TModule):
    def __init__(self):
        super().__init__();

        self._name = "SelectByWeightNT";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        # W: weights, N1 ... Nm * T2 * T1, V: values, N1 ...Nm * T1 * D
        W, V = data;
        self._V, self._W = np.expand_dims(V, axis = -3), np.expand_dims(W, axis = -1);
        Y = np.sum(self._V * self._W, axis = -2);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = np.repeat(np.expand_dims(dout[0], axis = -2), self._V.shape[-2], axis = -2);
        dV = np.sum(dY * self._W, axis = -3);
        dW = np.sum(dY * self._V, axis = -1);

        return dW, dV;


# additive attention weight for 1 time step
class AdditiveAttentionWeight1TModule(AggregateNetModule):
    def __init__(self, querySize : int, keySize : int, hiddenSize : int, Wq : np.ndarray = None, Wk : np.ndarray = None, wv : np.ndarray = None):
        self._qLayer = AffineLayer(querySize, hiddenSize, includeBias = False, W = Wq);
        self._kLayer = AffineLayer(keySize, hiddenSize, includeBias = False, W = Wk);
        self._vLayer = AffineLayer(hiddenSize, 1, includeBias = False, W = wv);
        self._softmax = SoftmaxLayer();

        super().__init__(self._qLayer, self._kLayer, self._vLayer, self._softmax);

        self._H = None;
        self._name = "AdditiveAttentionWeight1T";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        # Q: queries, N1 ... Nm * Q, K: keys, N1 ...Nm * T1 * K
        Q, K = data;

        QY = self._qLayer.forward(Q)[0];
        KY = self._kLayer.forward(K)[0];

        QY = np.expand_dims(QY, axis = -2);
        self._H = tanh(QY + KY);
        S = self._vLayer.forward(self._H)[0];
        S = np.squeeze(S, axis = -1);
        Y = self._softmax.forward(S)[0];

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dS = self._softmax.backward(*dout)[0];
        dS = np.expand_dims(dS, axis = -1);
        dH = self._vLayer.backward(dS)[0];
        dH = dH * tanhGradient(self._H);
        dQ = self._qLayer.backward(np.sum(dH, axis = -2))[0];
        dK = self._kLayer.backward(dH)[0];

        return dQ, dK;


# additive attention weight for N time step
class AdditiveAttentionWeightNTModule(AdditiveAttentionWeight1TModule):
    def __init__(self, querySize : int, keySize : int, hiddenSize : int, Wq : np.ndarray = None, Wk : np.ndarray = None, wv : np.ndarray = None):
        super().__init__(querySize, keySize, hiddenSize, Wq, Wk, wv);

        self._name = "AdditiveAttentionWeightNT";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        # Q: queries, N1 ... Nm * T2 * Q, K: keys, N1 ...Nm * T1 * K
        Q, K = data;

        QY = self._qLayer.forward(Q)[0];
        KY = self._kLayer.forward(K)[0];

        QY = np.expand_dims(QY, axis = -2);
        KY = np.expand_dims(KY, axis = -3);
        self._H = tanh(QY + KY);
        S = self._vLayer.forward(self._H)[0];
        S = np.squeeze(S, axis = -1);
        Y = self._softmax.forward(S)[0];

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dS = self._softmax.backward(*dout)[0];
        dS = np.expand_dims(dS, axis = -1);
        dH = self._vLayer.backward(dS)[0];
        dH = dH * tanhGradient(self._H);
        dQ = self._qLayer.backward(np.sum(dH, axis = -2))[0];
        dK = self._kLayer.backward(np.sum(dH, axis = -3))[0];

        return dQ, dK;


# dot-product attention weight for 1 time step
class DotProductAttentionWeight1TModule(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._K = None;
        self._Q = None;
        self._softmax = SoftmaxLayer();
        self._name = "DotProductAttentionWeight1T";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        # Q: queries, N1 ... Nm * D, K: keys, N1 ...Nm * T1 * D
        Q, K = data;
        self._K, self._Q = K, np.expand_dims(Q, axis = -2);
        Y = self._softmax.forward(np.sum(self._K * self._Q, axis = -1) / math.sqrt(Q.shape[-1]))[0];

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = self._softmax.backward(*dout)[0] / math.sqrt(self._Q.shape[-1]);
        dY = np.repeat(np.expand_dims(dY, axis = -1), self._Q.shape[-1], axis = -1);
        dK = dY * self._Q;
        dQ = np.sum(dY * self._K, axis = -2);

        return dQ, dK;


# dot-product attention weight for N time step
class DotProductAttentionWeightNTModule(DotProductAttentionWeight1TModule):
    def __init__(self):
        super().__init__();

        self._name = "DotProductAttentionWeightNT";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        # Q: queries, N1 ... Nm * T2 * D, K: keys N1 ...Nm * T1 * D
        Q, K = data;
        self._K, self._Q = np.expand_dims(K, axis = -3), np.expand_dims(Q, axis = -2);
        Y = self._softmax.forward(np.sum(self._K * self._Q, axis = -1) / math.sqrt(Q.shape[-1]))[0];

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = self._softmax.backward(*dout)[0] / math.sqrt(self._Q.shape[-1]);
        dY = np.repeat(np.expand_dims(dY, axis = -1), self._Q.shape[-1], axis = -1);
        dK = np.sum(dY * self._Q, axis = -3);
        dQ = np.sum(dY * self._K, axis = -2);

        return dQ, dK;


class QKVAttentionLayer(AggregateNetModule):
    def __init__(self, weightModule : INetModule, selectModule : INetModule):
        super().__init__(weightModule, selectModule);

        self._attentionWeight = None;
        self._weightModule = weightModule;
        self._selectModule = selectModule;
        self._name = "QKVAttention";


    @property
    def attentionWeight(self) -> np.ndarray:
        return self._attentionWeight;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        Q, K, V = data;
        self._attentionWeight = self._weightModule.forward(Q, K)[0];
        Y = self._selectModule.forward(self._attentionWeight, V)[0];

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dW, dV = self._selectModule.backward(*dout);
        dQ, dK = self._weightModule.backward(dW);

        return dQ, dK, dV;


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
    def __init__(self, data : List[np.ndarray], batchSize : int, stepSize : int, shuffle : bool = False):
        self._data = data;
        self._length = len(data[0]);
        self._batchSize = batchSize;
        self._stepSize = stepSize;
        self._totalIterations = 1;
        self._shuffle = shuffle;


    def _sequentialSample(self):
        offset = int(np.random.randint(0, self._stepSize));
        totalLength = ((self._length - offset) // self._batchSize) * self._batchSize;
        data = [d[offset: offset + totalLength].reshape((self._batchSize, -1) + d.shape[1:]) for d in self._data];
        self._totalIterations = data[0].shape[1] // self._stepSize

        for i in range(0, self._totalIterations * self._stepSize, self._stepSize):
            yield tuple([d[:, i: i + self._stepSize] for d in data]);


    def _randomSample(self):
        offset = int(np.random.randint(0, self._stepSize));
        data = [d[offset:] for d in self._data];
        subIdx = list(range(0, ((self._length - offset) // self._stepSize) * self._stepSize, self._stepSize));
        self._totalIterations = len(subIdx) // self._batchSize;

        np.random.shuffle(subIdx);
        for i in range(0, self._totalIterations * self._batchSize, self._batchSize):
            idx = subIdx[i: i + self._batchSize];
            yield tuple([np.array([d[j: j + self._stepSize] for j in idx]) for d in data]);


    def __iter__(self):
        return self._randomSample() if self._shuffle else self._sequentialSample();


    @property
    def totalIterations(self) -> int:
        return self._totalIterations;


class RegressionAccuracyEvaluator(INetAccuracyEvaluator):
    def __init__(self):
        self._rss = 0.0;
        self._totalCount = 0.0;


    @property
    def name(self) -> str:
        return "RMSE";


    @property
    def accuracy(self) -> Optional[float]:
        return math.sqrt(self._rss / self._totalCount) if self._totalCount > 0 else None;


    def fromLoss(self, lossValues : List[float] = None) -> bool:
        return False;


    def update(self, *data: np.ndarray):
        Y, T = data;
        self._rss += float(np.sum(np.square(Y - T)));
        self._totalCount += lengthExceptLastDimension(Y);


    def reset(self):
        self._rss = 0.0;
        self._totalCount = 0.0;


class ClassifierAccuracyEvaluator(INetAccuracyEvaluator):
    def __init__(self):
        self._rightCount = 0.0;
        self._totalCount = 0.0;


    @property
    def name(self) -> str:
        return "Classification Accuracy";


    @property
    def accuracy(self) -> Optional[float]:
        return self._rightCount / self._totalCount if self._totalCount > 0 else None;


    def fromLoss(self, lossValues : List[float] = None) -> bool:
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
        self._perplexity = None;


    @property
    def name(self) -> str:
        return "Perplexity";


    @property
    def accuracy(self) -> Optional[float]:
        return self._perplexity;


    def fromLoss(self, lossValues : Optional[List[float]] = None) -> bool:
        if lossValues is None:
            return False;

        self._perplexity = math.exp(sum(lossValues) / len(lossValues));
        return True;


    def update(self, *data: np.ndarray):
        pass;


    def reset(self):
        self._perplexity = None;


class NetTrainer:
    def __init__(self, model : INetModel, lossFunc : INetLoss, optimizer : INetOptimizer, evaluator : INetAccuracyEvaluator = None):
        self._model = model;
        self._lossFunc = lossFunc;
        self._optimizer = optimizer;
        self._evaluator = evaluator;

        self._lossData = [];
        self._trainingAccuracyData = [];
        self._testAccuracyData = [];


    def _calcAccuracy(self, lossValues : List[float] = None, iterator : Iterable = None) -> float:
        self._model.isTrainingMode = False;

        try:
            self._evaluator.reset();

            if not self._evaluator.fromLoss(lossValues) and iterator is not None:
                lossValues = [];

                for data in iterator:
                    Y = self._model.forward(*data[:-1]);
                    loss = self._lossFunc.forward(*Y, self._model.getFinalTag(data[-1]));

                    lossValues.append(loss);
                    self._evaluator.update(*Y, self._model.getFinalTag(data[-1]));

                self._evaluator.fromLoss(lossValues);
        finally:
            self._model.reset();
            self._model.isTrainingMode = True;

        return self._evaluator.accuracy;


    def train(self, maxEpoch : int, trainingIterator : IDataIterator, testIterator : IDataIterator = None,
              evalEpoch : bool = True, evalIterations : int = None, evalTrainingData : bool = False, evalTestData : bool = True):
        lossValues = [];

        self._lossData.clear();
        self._trainingAccuracyData.clear();
        self._testAccuracyData.clear();

        startTime = time.time();
        print(f"[{datetime.datetime.now()}] start to train model {self._model}");

        for epoch in range(maxEpoch):
            lossValues.clear();

            for data in trainingIterator:
                Y = self._model.forward(*data[:-1]);
                loss = self._lossFunc.forward(*Y, self._model.getFinalTag(data[-1]));
                lossValues.append(loss);

                self._model.backward(*self._lossFunc.backward());
                self._optimizer.update(self._model.params, self._model.grads);

                if self._evaluator is not None and evalIterations is not None and len(lossValues) % evalIterations == 0:
                    accuracy = self._calcAccuracy(lossValues[-evalIterations:], None);
                    if accuracy is not None:
                        print(f"epoch {epoch}, iterations: {len(lossValues)} / {trainingIterator.totalIterations}, elapsed time: {int(time.time() - startTime)}s, training {self._evaluator.name}: {accuracy}");

            self._model.reset();
            self._lossData.append(sum(lossValues) / len(lossValues));

            if self._evaluator is not None and evalEpoch:
                if evalTrainingData:
                    print("evaluating training data...");
                    self._trainingAccuracyData.append(self._calcAccuracy(lossValues, trainingIterator));
                if testIterator is not None and evalTestData:
                    print("evaluating test data...");
                    self._testAccuracyData.append(self._calcAccuracy(None, testIterator));

            trainingMessage = f", training {self._evaluator.name}: {self._trainingAccuracyData[-1]}" if len(self._trainingAccuracyData) > 0 else "";
            testMessage = f", test {self._evaluator.name}: {self._testAccuracyData[-1]}" if len(self._testAccuracyData) > 0 else "";
            print(f"epoch {epoch}, average loss: {self._lossData[-1]}{trainingMessage}{testMessage}, elapsed time: {int(time.time() - startTime)}s");

        if self._evaluator is not None:
            print("evaluating final training data...");
            print(f"the final training {self._evaluator.name} is {self._calcAccuracy(None, trainingIterator)}");

            if testIterator is not None:
                print("evaluating final test data...");
                print(f"the final test {self._evaluator.name} is {self._calcAccuracy(None, testIterator)}");

        print(f"[{datetime.datetime.now()}] complete to train model, elapsed time: {int(time.time() - startTime)}s");


    def plot(self):
        plt.figure(1);
        plt.xlabel("epoch");
        plt.ylabel('loss');

        plt.plot(self._lossData, "o-k", label = "loss");
        if len(self._trainingAccuracyData) > 0:
            plt.plot(self._trainingAccuracyData, "D-b", label = "training accuracy");
        if len(self._testAccuracyData) > 0:
            plt.plot(self._testAccuracyData, "s-r", label = "test accuracy");

        plt.legend(loc ='upper right');
        plt.show(block = True);
        plt.close();
