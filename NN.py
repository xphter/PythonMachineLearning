# -*- coding: utf-8 -*-
#
# Copyright (C) 2024 XphteR, Inc. All Rights Reserved
#
# @Time    : 2024-11-10
# @Author  : Du Peng
# @Email   : 278770518@qq.com
# @File    : NN.py
######################################################


import abc;
import copy;
import math;
import time;
import datetime;
import collections;

import matplotlib.pyplot as plt;

from typing import Union, List, Tuple, Callable, Any, Optional, Iterable, Generator;
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


    @learningRate.setter
    @abc.abstractmethod
    def learningRate(self, value : float):
        pass;


    @abc.abstractmethod
    def update(self, params : List[np.ndarray], grads : List[np.ndarray]):
        pass;


class IDataScaler(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def params(self) -> List:
        pass;


    @params.setter
    @abc.abstractmethod
    def params(self, value: List):
        pass;

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
    def high(self) -> bool:
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
    def trainingEpoch(self) -> int:
        pass;


    @trainingEpoch.setter
    @abc.abstractmethod
    def trainingEpoch(self, value: int):
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
    def copy(self, shareParams : bool = False):
        pass;


    @abc.abstractmethod
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        pass;


    @abc.abstractmethod
    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        pass;


    @abc.abstractmethod
    def clearGrads(self):
        pass;


class INetModel(INetModule, metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def getFinalTag(self, T : np.ndarray) -> Optional[np.ndarray]:
        pass;


    @abc.abstractmethod
    def eval(self, lossFunc: INetLoss, evaluator: INetAccuracyEvaluator, lossValues: List[float] = None, iterator: Iterable = None) -> Tuple[float, float]:
        pass;


    @abc.abstractmethod
    def fit(self, trainingIterator : IDataIterator, lossFunc: INetLoss, optimizer: INetOptimizer, maxEpoch : int, testIterator : IDataIterator = None,
            evaluator: INetAccuracyEvaluator = None, evalEpoch : bool = True, evalIterations : int = None, evalTrainingData : bool = False, evalTestData : bool = True,
            minEpoch : int = None, plot = False) -> Tuple[List[float], List[float], List[float], List[float]]:
        pass;


    @abc.abstractmethod
    def predict(self, iterator : IDataIterator) -> Iterable:
        pass;


    @abc.abstractmethod
    def predictOne(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        pass;


class NetModuleBase(INetModule, metaclass = abc.ABCMeta):
    def __init__(self):
        self._name = None;
        self._params = [];
        self._grads = [];
        self._isTrainingMode = False;
        self._trainingEpoch = 0;


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
    def trainingEpoch(self) -> int:
        return self._trainingEpoch;


    @trainingEpoch.setter
    def trainingEpoch(self, value: int):
        self._trainingEpoch = value;
        self._setTrainingEpoch(value);


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


    @grads.setter
    def grads(self, value: List[np.ndarray]):
        self._grads = value;


    def _setTrainingMode(self, value : bool):
        pass;


    def _setTrainingEpoch(self, value : int):
        pass;


    def _setParams(self, value: List[np.ndarray]):
        pass;


    def _reset(self):
        pass;


    def _copyMembers(self, module : INetModule, shareParams : bool):
        pass;


    def _copyParams(self, module : INetModule, shareParams : bool):
        if shareParams:
            module.params = [p for p in self.params];
        else:
            module.params = [np.copy(p) for p in self.params];


    def _copyGrads(self, module : INetModule, shareParams : bool):
        module.grads = [np.zeros_like(g) for g in self.grads];


    def reset(self):
        self.isTrainingMode = False;
        self.trainingEpoch = 0;
        self._reset();


    def copy(self, shareParams : bool = False) -> INetModule:
        module = copy.copy(self);

        self._copyMembers(module, shareParams);
        self._copyParams(module, shareParams);
        self._copyGrads(module, shareParams);

        return module;


    def clearGrads(self):
        for g in self.grads:
            g[...] = 0;


class AggregateNetModule(NetModuleBase):
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


    def _setTrainingEpoch(self, value : int):
        for m in self._modules:
            m.trainingEpoch = value;


    def _setParams(self, value: List[np.ndarray]):
        i = 0;

        for m in self._modules:
            if (n := len(m.params)) == 0:
                continue;

            m.params = value[i: i + n];
            i += n;


    def _reset(self):
        for m in self._modules:
            m.reset();


    def _copyMembers(self, module : INetModule, shareParams : bool):
        module._modules = [m.copy(shareParams) for m in self.modules];


    def _copyParams(self, module : INetModule, shareParams : bool):
        module._params = [];
        for m in module.modules:
            module.params.extend(m.params);


    def _copyGrads(self, module : NetModuleBase, shareParams : bool):
        module._grads = [];
        for m in module.modules:
            module.grads.extend(m.grads);


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


    def getFinalTag(self, T : np.ndarray) -> Optional[np.ndarray]:
        return T;


    def eval(self, lossFunc : INetLoss, evaluator : INetAccuracyEvaluator, lossValues : List[float] = None, iterator : Iterable = None) -> Tuple[float, float]:
        self.isTrainingMode = False;

        try:
            evaluator.reset();

            if not (lossValues is not None and evaluator.fromLoss(lossValues)) and iterator is not None:
                lossValues = [];

                for data in iterator:
                    Y = self.forward(*data);
                    T = self.getFinalTag(data[-1]);
                    loss = lossFunc.forward(*Y, T) if T is not None else lossFunc.forward(*Y);

                    lossValues.append(loss);
                    evaluator.update(*Y, T);

                evaluator.fromLoss(lossValues);
        finally:
            self.reset();
            self.isTrainingMode = True;

        return sum(lossValues) / len(lossValues), evaluator.accuracy;


    def fit(self, trainingIterator: IDataIterator, lossFunc: INetLoss, optimizer: INetOptimizer, maxEpoch: int, testIterator: IDataIterator = None,
            evaluator: INetAccuracyEvaluator = None, evalEpoch: bool = True, evalIterations: int = None, evalTrainingData: bool = False, evalTestData: bool = True,
            minEpoch : int = None, plot = False) -> Tuple[List[float], List[float], List[float], List[float]]:
        lossValues = [];
        trainingLossData = [];
        trainingAccuracyData = [];
        testLossData = [];
        testAccuracyData = [];
        paramsData = [];

        startTime = time.time();
        print(f"[{datetime.datetime.now()}] start to train model {self}");

        for epoch in range(maxEpoch):
            lossValues.clear();
            if evaluator is not None:
                evaluator.reset();

            self.isTrainingMode = True;
            self.trainingEpoch = epoch;

            for data in trainingIterator:
                Y = self.forward(*data);
                T = self.getFinalTag(data[-1]);
                loss = lossFunc.forward(*Y, T) if T is not None else lossFunc.forward(*Y);

                if not math.isfinite(loss):
                    raise OverflowError("training fail: the return value of loss function is NaN or infinity");

                lossValues.append(loss);
                if evaluator is not None:
                    evaluator.update(*Y, T);

                self.backward(*lossFunc.backward());
                optimizer.update(self.params, self.grads);
                self.clearGrads();

                if evaluator is not None and evalIterations is not None and len(lossValues) % evalIterations == 0:
                    loss, accuracy = self.eval(lossFunc, evaluator, lossValues[-evalIterations:], None);
                    if accuracy is not None:
                        print(f"epoch {epoch}, iterations: {len(lossValues)} / {trainingIterator.totalIterations}, training loss: {loss}, training {evaluator.name}: {accuracy}, elapsed time: {int(time.time() - startTime)}s");

            self.reset();
            trainingLossData.append(sum(lossValues) / len(lossValues));
            if evaluator is not None:
                trainingAccuracyData.append(evaluator.accuracy);

            if evaluator is not None and evalEpoch:
                if evalTrainingData:
                    print("evaluating training data...");
                    loss, accuracy = self.eval(lossFunc, evaluator, lossValues, trainingIterator);
                    trainingLossData[-1] = loss;
                    trainingAccuracyData[-1] = accuracy;
                if testIterator is not None and evalTestData:
                    print("evaluating test data...");
                    loss, accuracy = self.eval(lossFunc, evaluator, None, testIterator);
                    testLossData.append(loss);
                    testAccuracyData.append(accuracy);
                    paramsData.append([np.copy(p) for p in self.params]);

            trainingMessage = f", training {evaluator.name}: {trainingAccuracyData[-1]}" if len(trainingAccuracyData) > 0 else "";
            testMessage = f", test loss: {testLossData[-1]}, test {evaluator.name}: {testAccuracyData[-1]}" if len(testAccuracyData) > 0 else "";
            print(f"epoch {epoch}, training loss: {trainingLossData[-1]}{trainingMessage}{testMessage}, elapsed time: {int(time.time() - startTime)}s");

        if minEpoch is not None and len(paramsData) > 0:
            index = np.argmax(testAccuracyData[minEpoch:]) if evaluator.high else np.argmin(testAccuracyData[minEpoch:]);
            self.params = paramsData[minEpoch + int(index)];

        if evaluator is not None:
            print("evaluating final training data...");
            print(f"the final training {evaluator.name} is {self.eval(lossFunc, evaluator, None, trainingIterator)[1]}");

            if testIterator is not None:
                print("evaluating final test data...");
                print(f"the final test {evaluator.name} is {self.eval(lossFunc, evaluator, None, testIterator)[1]}");

        self.isTrainingMode = False;
        print(f"[{datetime.datetime.now()}] complete to train model, elapsed time: {int(time.time() - startTime)}s");

        if plot:
            fig = plt.figure(1);

            ax1 = fig.add_subplot(111);
            ax1.set_xlabel("epoch");
            ax1.set_ylabel('loss');
            ax1.plot(trainingLossData, "o-k", label = "training loss");
            if len(testLossData) > 0:
                ax1.plot(testLossData, "o-b", label = "test loss");

            ax2 = ax1.twinx();
            ax2.set_ylabel('accuracy');
            if len(trainingAccuracyData) > 0:
                ax2.plot(trainingAccuracyData, "D-m", label = f"training {evaluator.name}");
            if len(testAccuracyData) > 0:
                ax2.plot(testAccuracyData, "D-r", label = f"test {evaluator.name}");

            fig.legend(loc = "upper left", bbox_to_anchor = (0, 1), bbox_transform = ax1.transAxes)
            plt.show(block = True);
            plt.close();

        return trainingLossData, trainingAccuracyData, testLossData, testAccuracyData;


    def predict(self, iterator : IDataIterator) -> Iterable:
        for data in iterator:
            yield self.forward(*data);


    def predictOne(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        return self.forward(*data);


class NetLossBase(INetLoss, metaclass = abc.ABCMeta):
    def __init__(self):
        self._loss = None;


    @property
    def loss(self) -> float:
        return self._loss;


class NetOptimizerBase(INetOptimizer, metaclass = abc.ABCMeta):
    def __init__(self, lr : float):
        self._lr = lr;


    @property
    def learningRate(self) -> float:
        return self._lr;


    @learningRate.setter
    def learningRate(self, value : float):
        self._lr = value;


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


class PReluLayer(NetModuleBase):
    def __init__(self, beta : Union[float, np.ndarray] = None, outputSize : int = None):
        super().__init__();

        self._X = None;
        self._name = "PRelu";

        if beta is not None:
            if isinstance(beta, np.ndarray):
                self._beta = beta;
            else:
                self._beta = np.array([float(beta)]);
        else:
            self._beta = sigmoid(np.random.randn(outputSize if outputSize is not None else 1).astype(defaultDType));

        self._params.append(self._beta);
        self._grads.append(np.zeros_like(self._beta));


    @property
    def beta(self) -> np.ndarray:
        return self._beta;


    def _setParams(self, value: List[np.ndarray]):
        self._beta = value[0];


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        self._X = data[0];
        Y = prelu(self._X, self._beta);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dX, dBeta = preluGradient(self._X, self._beta);

        dX *= dY;
        dBeta *= dY;
        if len(self._beta) == 1:
            dBeta = np.sum(dBeta);
        else:
            dBeta = np.sum(dBeta, axis = tuple(range(len(dBeta.shape) - 1)));

        self._grads[0][...] = dBeta;

        return dX, ;


class SoftplusLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._X = None;
        self._M = None;
        self._name = "Softplus";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        self._X = data[0];
        Y, self._M = softplus(self._X);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dX = dY * softplusGradient(self._X, self._M);

        return dX, ;


class SwishLayer(NetModuleBase):
    def __init__(self, beta : Union[float, np.ndarray] = None, outputSize : int = None):
        super().__init__();

        self._X = None;
        self._Y = None;
        self._S  = None;
        self._name = "Swish";

        if beta is not None:
            if isinstance(beta, np.ndarray):
                self._beta = beta;
            else:
                self._beta = np.array([float(beta)]);
        else:
            self._beta = sigmoid(np.random.randn(outputSize if outputSize is not None else 1).astype(defaultDType));

        self._params.append(self._beta);
        self._grads.append(np.zeros_like(self._beta));


    @property
    def beta(self) -> np.ndarray:
        return self._beta;


    def _setParams(self, value: List[np.ndarray]):
        self._beta = value[0];


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        self._X = data[0];
        self._S, self._Y = swish(self._X, self._beta);

        return self._Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        dX, dBeta = swishGradient(self._Y, self._S, self._X, self._beta);

        dX *= dY;
        dBeta *= dY;
        if len(self._beta) == 1:
            dBeta = np.sum(dBeta);
        else:
            dBeta = np.sum(dBeta, axis = tuple(range(len(dBeta.shape) - 1)));

        self._grads[0][...] = dBeta;

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
    def __init__(self, dropoutRatio = 0.5, reuseMask : bool = False):
        super().__init__();

        self._mask = None;
        self._reuseMask = reuseMask;
        self._dropoutRatio = dropoutRatio;
        self._name = f"Dropout {dropoutRatio}";


    def _getMask(self, shape : Tuple, dtype) -> np.ndarray:
        return (np.random.rand(*shape) > self._dropoutRatio).astype(dtype) / (1.0 - self._dropoutRatio);


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];

        if self._isTrainingMode:
            if self._mask is None or not self._reuseMask:
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


    def clearMask(self):
        self._mask = None;


# the index of time dimension is 1
class VariationalDropoutLayer(DropoutLayer):
    def __init__(self, dropoutRatio = 0.5):
        super().__init__(dropoutRatio);

        self._name = f"VariationalDropout {dropoutRatio}";


    def _getMask(self, shape : Tuple, dtype) -> np.ndarray:
        if len(shape) <= 2:
            return super()._getMask(shape, dtype);

        mask = super()._getMask((shape[0], ) + shape[2:], dtype);
        mask = np.repeat(np.expand_dims(mask, axis = 1), shape[1], axis = 1);

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


class FlattenLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._shapes = [];
        self._name = "Flatten";


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray]:
        self._shapes.clear();
        output : List[np.ndarray] = [];

        for X in data:
            self._shapes.append(X.shape);

            if X.ndim > 1:
                output.append(np.reshape(X, (len(X), -1)));
            else:
                output.append(X);

        return tuple(output);


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray]:
        dX : List[np.ndarray] = [];

        for dY, shape in zip(dout, self._shapes):
            dX.append(np.reshape(dY, shape));

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
        # dW[...] = 0;

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


    def _reset(self):
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
        # for i in range(len(self._grads)):
        #     self._grads[i][...] = 0;

        for t in reversed(range(T)):
            rnn = self._rnnModules[t];
            dX[:, t, :], self._dH = rnn.backward(dY[:, t, :] + self._dH);

            for i in range(len(self._grads)):
                self._grads[i] += rnn.grads[i];

        return dX, ;


    def setState(self, H : np.ndarray):
        self._H = H;


'''
dropout mechanism: https://arxiv.org/abs/1603.05118 <Recurrent Dropout without Memory Loss>
'''
class LstmCell(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, Wx : np.ndarray = None, Wh : np.ndarray = None, b : np.ndarray = None, inputDropout : float = 0, recurrentDropout : float = 0):
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
        self._inputDropout = inputDropout;
        self._recurrentDropout = recurrentDropout;
        self._inputDropoutMask = None;
        self._recurrentDropoutMask = None;

        weights = [self._weightX, self._weightH, self._bias];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    def _setParams(self, value: List[np.ndarray]):
        self._weightX, self._weightH, self._bias = value[0], value[1], value[2];


    def _reset(self):
        self.setInputDropoutMask();
        self.setRecurrentDropoutMask();


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

        if self._inputDropoutMask is None:
            self._inputDropoutMask = getDropoutMask(self._H, self._inputDropout);
        if self._recurrentDropoutMask is None:
            self._recurrentDropoutMask = getDropoutMask(self._C, self._recurrentDropout);

        if self.isTrainingMode:
            A = self._X @ self._weightX + (self._inputDropoutMask * self._H) @ self._weightH + self._bias;
        else:
            A = self._X @ self._weightX + ((1 - self._inputDropout) * self._H) @ self._weightH + self._bias;
        self._F, self._G, self._I, self._O = tuple(np.hsplit(A, 4));
        self._F, self._G, self._I, self._O = sigmoid(self._F), tanh(self._G), sigmoid(self._I), sigmoid(self._O);

        if self.isTrainingMode:
            self._YC = self._C * self._F + self._recurrentDropoutMask * self._G * self._I;
        else:
            self._YC = self._C * self._F + (1 - self._recurrentDropout) * self._G * self._I;
        self._tanhYC = tanh(self._YC);
        self._YH = self._tanhYC * self._O;

        return self._YH, self._YC;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dYH, dYC = dout;

        dYC += dYH * self._O * tanhGradient(self._tanhYC);
        dF, dG, dI, dO = dYC * self._C, dYC * self._I * self._recurrentDropoutMask, dYC * self._G * self._recurrentDropoutMask, dYH * self._tanhYC;
        dF *= sigmoidGradient(self._F);
        dG *= tanhGradient(self._G);
        dI *= sigmoidGradient(self._I);
        dO *= sigmoidGradient(self._O);
        dA = np.hstack((dF, dG, dI, dO));

        dWx = self._X.T @ dA;
        dWh = (self._inputDropoutMask * self._H).T @ dA;
        db = np.sum(dA, axis = 0);

        dX = dA @ self._weightX.T;
        dH = (dA @ self._weightH.T) * self._inputDropoutMask;
        dC = dYC * self._F;

        self._grads[0][...] = dWx;
        self._grads[1][...] = dWh;
        self._grads[2][...] = db;

        return dX, dH, dC;


    def setInputDropoutMask(self, mask : np.ndarray = None):
        self._inputDropoutMask = mask;


    def setRecurrentDropoutMask(self, mask : np.ndarray = None):
        self._recurrentDropoutMask = mask;


class LstmLayer(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, Wx : np.ndarray = None, Wh : np.ndarray = None, b : np.ndarray = None, returnSequences : bool = False, returnState : bool = False, stateful : bool = False, stepwise = False, inputDropout : float = 0, recurrentDropout : float = 0):
        super().__init__();

        self._T = 0;
        self._H, self._C = None, None;
        self._dH, self._dC = None, None;
        self._returnSequences = returnSequences;
        self._returnState = returnState;
        self._inputState = False;
        self._stateful = stateful;
        self._stepwise = stepwise;
        self._stepIndex = 0;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._name = f"LSTM {inputSize}*{outputSize}";

        self._weightX = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, 4 * outputSize).astype(defaultDType) if Wx is None else Wx;
        self._weightH = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, 4 * outputSize).astype(defaultDType) if Wh is None else Wh;
        self._bias = np.zeros(4 * outputSize, dtype = defaultDType) if b is None else b;
        self._inputDropout = inputDropout;
        self._recurrentDropout = recurrentDropout;
        self._inputDropoutMask = None;
        self._recurrentDropoutMask = None;
        self._lstmModules : List[LstmCell] = [];


        weights = [self._weightX, self._weightH, self._bias];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    def _setTrainingMode(self, value: bool):
        for cell in self._lstmModules:
            cell.isTrainingMode = value;


    def _setTrainingEpoch(self, value : int):
        for cell in self._lstmModules:
            cell.trainingEpoch = value;


    def _setParams(self, value: List[np.ndarray]):
        self._weightX, self._weightH, self._bias = value[0], value[1], value[2];
        for cell in self._lstmModules:
            cell.params = value;


    def _reset(self):
        for cell in self._lstmModules:
            cell.reset();


    def _getInputState(self, *data: np.ndarray):
        if len(data) > 1:
            return True, data[1], data[2];
        else:
            return False, None, None;


    def _createCell(self) -> LstmCell:
        cell = LstmCell(self._inputSize, self._outputSize, self._weightX, self._weightH, self._bias, inputDropout = self._inputDropout, recurrentDropout = self._recurrentDropout);
        cell.isTrainingMode = self.isTrainingMode;
        cell.trainingEpoch = self.trainingEpoch;
        cell.setInputDropoutMask(self._inputDropoutMask);
        cell.setRecurrentDropoutMask(self._recurrentDropoutMask);
        return cell;


    def _forwardStep(self, t : int, *data : np.ndarray):
        X = data[0];
        self._H, self._C = self._lstmModules[t].forward(X, self._H, self._C);


    def _backwardStep(self, t : int, *dout: np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        lstm = self._lstmModules[t];
        dX, self._dH, self._dC = lstm.backward(dY + self._dH, self._dC);

        for i in range(len(lstm.grads)):
            self._grads[i] += lstm.grads[i];

        return dX, ;


    def _forwardAll(self, *data : np.ndarray) -> np.ndarray:
        X = data[0];
        N, T = X.shape[: 2];

        Y = np.zeros((N, T, self._outputSize), dtype = X.dtype);
        for t in range(T):
            self._forwardStep(t, X[:, t]);
            Y[:, t] = self._H;

        return Y;


    def _backwardAll(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        N, T = len(dY), self._T;
        dX = np.zeros((N, T, self._inputSize), dtype = dY.dtype);

        for t in reversed(range(T)):
            dX[:, t], = self._backwardStep(t, dY[:, t]);

        return dX, ;


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


    def _reset(self):
        self._H, self._C = None, None;
        self.resetStepState();


    # input: X, H, C
    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray]:
        X = data[0];
        N = len(X);

        if not self._stepwise or self._stepIndex == 0:
            if not self._stateful or self._H is None:
                self._H = np.zeros((N, self._outputSize), dtype = X.dtype);
            if not self._stateful or self._C is None:
                self._C = np.zeros((N, self._outputSize), dtype = X.dtype);

            # only used in unit test!
            # if self._inputDropoutMask is None:
            #     self._inputDropoutMask = getDropoutMask(self._H, self._inputDropout);
            # if self._recurrentDropoutMask is None:
            #     self._recurrentDropoutMask = getDropoutMask(self._C, self._recurrentDropout);

            self._inputDropoutMask = getDropoutMask(self._H, self._inputDropout);
            self._recurrentDropoutMask = getDropoutMask(self._C, self._recurrentDropout);

            for cell in self._lstmModules:
                cell.setInputDropoutMask(self._inputDropoutMask);
                cell.setRecurrentDropoutMask(self._recurrentDropoutMask);

        self._inputState, H, C = self._getInputState(*data);
        if self._inputState:
            self._H = H if H is not None else np.zeros((N, self._outputSize), dtype = X.dtype);
            self._C = C if C is not None else np.zeros((N, self._outputSize), dtype = X.dtype);

        if not self._stepwise:
            self._T = T = X.shape[1];

            if len(self._lstmModules) < T:
                self._lstmModules.extend([self._createCell() for _ in range(T - len(self._lstmModules))]);

            Y = self._forwardAll(*data);
            if not self._returnSequences:
                Y = Y[:, -1];
        else:
            while len(self._lstmModules) < self._stepIndex + 1:
                self._lstmModules.append(self._createCell());

            self._forwardStep(self._stepIndex, *data);
            self._stepIndex += 1;

            Y = self._H;

        return (Y, self._H, self._C) if self._returnState else (Y, );


    # input: dY, dH, dC
    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];

        if not self._stepwise or self._stepIndex == len(self._lstmModules):
            # truncated BPTT
            self._dH = np.zeros_like(self._H);
            self._dC = np.zeros_like(self._H);
            # for i in range(len(self._grads)):
            #     self._grads[i][...] = 0;

        if self._returnState:
            self._dH += dout[1];
            self._dC += dout[2];

        if not self._stepwise:
            N, T = len(dY), self._T;

            if not self._returnSequences:
                dH = dY;
                dY = np.zeros((N, T, dY.shape[-1]), dtype = dY.dtype);
                dY[:, -1] = dH;

            din = self._backwardAll(*((dY, ) + dout[1:]));
        else:
            self._stepIndex -= 1;
            din = self._backwardStep(self._stepIndex, *((dY, ) + dout[1:]));

        if self._inputState:
            if not self._stepwise:
                din += (self._dH, self._dC);
            else:
                din += (np.copy(self._dH), np.copy(self._dC));
                self._dH[...] = 0;
                self._dC[...] = 0;

        return din;


    def setState(self, H : np.ndarray, C : np.ndarray = None):
        self._H, self._C = H, C;


    def resetStepState(self):
        if self._stepwise:
            self._stepIndex = 0;


    # only used in unit test!
    def setInputDropoutMask(self, mask : np.ndarray = None):
        self._inputDropoutMask = mask;

        for cell in self._lstmModules:
            cell.setInputDropoutMask(self._inputDropoutMask);


    # only used in unit test!
    def setRecurrentDropoutMask(self, mask : np.ndarray = None):
        self._recurrentDropoutMask = mask;

        for cell in self._lstmModules:
            cell.setRecurrentDropoutMask(self._recurrentDropoutMask);


class BahdanauAttentionLstmLayer(LstmLayer):
    def __init__(self, inputSize : int, outputSize : int, Wx : np.ndarray = None, Wh : np.ndarray = None, b : np.ndarray = None, Wq : np.ndarray = None, Wk : np.ndarray = None, wv : np.ndarray = None, returnSequences : bool = False, returnState : bool = False, stateful : bool = False, stepwise = False, inputDropout : float = 0, recurrentDropout : float = 0):
        super().__init__(outputSize + inputSize, outputSize, Wx, Wh, b, returnSequences, returnState, stateful, stepwise, inputDropout, recurrentDropout);

        self._shapeK = None;
        self._attentionWeight = None;
        self._name = f"BahdanauAttentionLSTM {inputSize}*{outputSize}";

        self._weightQ = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, outputSize).astype(defaultDType) if Wq is None else Wq;
        self._weightK = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, outputSize).astype(defaultDType) if Wk is None else Wk;
        self._weightV = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, 1).astype(defaultDType) if wv is None else wv;
        self._attentionModules : List[QKVAttentionLayer] = [];

        weights = [self._weightQ, self._weightK, self._weightV];
        self._params.extend(weights);
        self._grads.extend([np.zeros_like(w) for w in weights]);


    def _setTrainingMode(self, value: bool):
        super()._setTrainingMode(value);

        for m in self._attentionModules:
            m.isTrainingMode = value;


    def _setTrainingEpoch(self, value : int):
        super()._setTrainingEpoch(value);

        for m in self._attentionModules:
            m.trainingEpoch = value;


    def _setParams(self, value: List[np.ndarray]):
        self._weightX, self._weightH, self._bias = value[0], value[1], value[2];
        self._weightQ, self._weightK, self._weightV = value[3], value[4], value[5];

        for cell in self._lstmModules:
            cell.params = (self._weightX, self._weightH, self._bias);

        for m in self._attentionModules:
            m.params = (self._weightQ, self._weightK, self._weightV);


    def _reset(self):
        super()._reset();

        for m in self._attentionModules:
            m.reset();


    def _getInputState(self, *data: np.ndarray):
        if len(data) > 2:
            return True, data[2], data[3];
        else:
            return False, None, None;


    def _createAttention(self):
        layer = QKVAttentionLayer(AdditiveAttentionWeight1TModule(self._outputSize, self._outputSize, self._outputSize, Wq = self._weightQ, Wk = self._weightK, wv = self._weightV), SelectByWeight1TModule());
        layer.isTrainingMode = self.isTrainingMode;
        layer.trainingEpoch = self.trainingEpoch;
        return layer;


    def _forwardStep(self, t : int, *data : np.ndarray):
        X, K = data[: 2];

        lstm = self._lstmModules[t];
        attention = self._attentionModules[t];

        context = attention.forward(self._H, K, K)[0];
        self._attentionWeight.append(attention.attentionWeight);
        self._H, self._C = lstm.forward(np.concatenate((context, X), axis = -1), self._H, self._C);


    def _backwardStep(self, t : int, *dout: np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];

        lstm = self._lstmModules[t];
        attention = self._attentionModules[t];

        dX, self._dH, self._dC = lstm.backward(dY + self._dH, self._dC);
        dContext = dX[:, : self._outputSize];
        dQ, dK, dV = attention.backward(dContext);
        self._dH += dQ;

        grads = lstm.grads + attention.grads;
        for i in range(len(grads)):
            self._grads[i] += grads[i];

        return dX[:, self._outputSize:], dK + dV;


    def _forwardAll(self, *data : np.ndarray) -> np.ndarray:
        X, K = data[: 2];
        N, T = X.shape[: 2];
        Y = np.zeros((N, T, self._outputSize), dtype = X.dtype);

        for t in range(T):
            self._forwardStep(t, X[:, t], K);
            Y[:, t] = self._H;

        return Y;


    def _backwardAll(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];
        N, T = len(dY), self._T;
        dX = np.zeros((N, T, self._inputSize - self._outputSize), dtype = dY.dtype);
        dK = np.zeros(self._shapeK, dtype = dY.dtype);

        for t in reversed(range(T)):
            dX[:, t], dKS = self._backwardStep(t, dY[:, t]);
            dK += dKS;

        return dX, dK;


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray]:
        X, K = data[: 2];
        self._shapeK = K.shape;

        if not self._stepwise or self._stepIndex == 0:
            self._attentionWeight = [];

        if not self._stepwise:
            T = X.shape[1];

            if len(self._attentionModules) < T:
                self._attentionModules = [self._createAttention() for _ in range(T - len(self._attentionModules))];
        else:
            while len(self._attentionModules) < self._stepIndex + 1:
                self._attentionModules.append(self._createAttention());

        return super().forward(*data);


    @property
    def attentionWeight(self) -> np.ndarray:
        return np.array(self._attentionWeight).transpose(1, 0, 2);


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


class IdentityWithMeanAbsoluteLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._mask = None;


    def forward(self, *data: np.ndarray) -> float:
        Y, T = data;
        self._mask = Y < T;
        self._loss = meanAbsoluteError(Y, T);

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        dX = np.ones_like(self._mask, dtype = defaultDType);
        dX[self._mask] = -1;
        dX /= lengthExceptLastDimension(self._mask)

        return dX, ;


class IdentityWithHuberLoss(NetLossBase):
    def __init__(self, delta : float = 1.0):
        super().__init__();

        self._delta = max(0.0, float(delta));
        self._Y, self._T = None, None;
        self._maskL, self._maskM, self._maskH = None, None, None;


    def forward(self, *data: np.ndarray) -> float:
        self._Y, self._T = data;
        self._loss, self._maskL, self._maskM, self._maskH = huberError(self._Y, self._T, self._delta);

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        dX = np.ones_like(self._Y, dtype = defaultDType);

        dX[self._maskL] *= -self._delta;
        dX[self._maskM] = self._Y[self._maskM] - self._T[self._maskM];
        dX[self._maskH] *= self._delta;
        dX /= lengthExceptLastDimension(self._T);

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


    @property
    def learningRate(self) -> float:
        return self._optimizer.learningRate;


    @learningRate.setter
    def learningRate(self, value: float):
        self._optimizer.learningRate = value;


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


    @property
    def learningRate(self) -> float:
        return self._optimizer.learningRate;


    @learningRate.setter
    def learningRate(self, value: float):
        self._optimizer.learningRate = value;


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


class SGDM(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, beta : float = 0.9):
        super().__init__(lr);

        self._m = None;
        self._beta = beta;


    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self._m is None:
            self._m = [np.zeros_like(item) for item in params];

        for i in range(len(params)):
            self._m[i] = self._beta * self._m[i] + self._lr * grads[i];
            params[i] -= self._m[i];


class Nesterov(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, beta : float = 0.9):
        super().__init__(lr);

        self._m = None;
        self._beta = beta;


    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self._m is None:
            self._m = [np.zeros_like(item) for item in params];

        for i in range(len(params)):
            m = self._m[i];
            self._m[i] = self._beta * self._m[i] + self._lr * grads[i];
            params[i] -= (1 + self._beta) * self._m[i] - self._beta * m;


class AdaGrad(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, epsilon : float = 1e-8):
        super().__init__(lr);

        self._v = None;
        self._epsilon = epsilon;


    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self._v is None:
            self._v = [np.zeros_like(item) for item in params];

        for i in range(len(params)):
            self._v[i] += grads[i] ** 2;
            params[i] -= self._lr * grads[i] / (np.sqrt(self._v[i]) + self._epsilon);


class RMSProp(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, beta : float = 0.9, epsilon : float = 1e-8):
        super().__init__(lr);

        self._v = None;
        self._beta = beta;
        self._epsilon = epsilon;


    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self._v is None:
            self._v = [np.zeros_like(item) for item in params];

        for i in range(len(params)):
            self._v[i] = self._beta * self._v[i] + (1 - self._beta) * grads[i] ** 2;
            params[i] -= self._lr * grads[i] / (np.sqrt(self._v[i]) + self._epsilon);


class Adam(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, beta1 : float = 0.9, beta2 : float = 0.999, epsilon : float = 1e-8):
        super().__init__(lr);

        self._m = None;
        self._v = None;
        self._beta1 = beta1;
        self._beta2 = beta2;
        self._epsilon = epsilon;
        self._t = 0;


    def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self._m is None:
            self._m = [np.zeros_like(item) for item in params];
        if self._v is None:
            self._v = [np.zeros_like(item) for item in params];

        self._t += 1;

        for i in range(len(params)):
            self._m[i] = self._beta1 * self._m[i] + (1 - self._beta1) * grads[i];
            self._v[i] = self._beta2 * self._v[i] + (1 - self._beta2) * grads[i] ** 2;

            m = self._m[i] / (1 - self._beta1 ** self._t);
            v = self._v[i] / (1 - self._beta2 ** self._t);

            params[i] -= self._lr * m / (np.sqrt(v) + self._epsilon);


class AggregateScaler(IDataScaler):
    def __init__(self, *scalers : IDataScaler):
        self._scalers = scalers;


    @property
    def params(self) -> List:
        return [item.params for item in self._scalers];


    @params.setter
    def params(self, value: List):
        for i in range(len(value)):
            self._scalers[i].params = value[i];


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


    @property
    def params(self) -> List:
        return [self._ndim] + self._getParams();


    @params.setter
    def params(self, value: List):
        self._ndim = value[0];
        self._setParams(tuple(value[1:]));
        self._fitted = True;


    @abc.abstractmethod
    def _getParams(self) -> List:
        pass;


    @abc.abstractmethod
    def _setParams(self, value: Tuple):
        pass;


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


    def _getParams(self) -> List:
        return [self._minRange, self._maxRange, self._minValue, self._maxValue];


    def _setParams(self, value: Tuple):
        self._minRange, self._maxRange, self._minValue, self._maxValue = value;
        self._rangeDelta = self._maxRange - self._minRange;
        self._valueDelta = self._maxValue - self._minValue;


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
    INDEX_ARGUMENT_NAME = "index";

    def __init__(self):
        super().__init__();

        self._mu = None;
        self._sigma = None;


    def _getParams(self) -> List:
        return [self._mu, self._sigma];


    def _setParams(self, value: Tuple):
        self._mu, self._sigma = value;


    def _fit(self, X: np.ndarray):
        self._mu = np.mean(X, axis = 0);
        self._sigma = np.std(X, axis = 0);
        return X;


    def _transform(self, X : np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sigma;


    def _inverse(self, Y : np.ndarray, *args, **kwargs) -> np.ndarray:
        index = np.array(kwargs[StandardScaler.INDEX_ARGUMENT_NAME]) if StandardScaler.INDEX_ARGUMENT_NAME in kwargs else None;
        return self._sigma * Y + self._mu if index is None else self._sigma[index] * Y + self._mu[index];


class DiffScaler(ScalerBase):
    INDEX_ARGUMENT_NAME = "index";

    def __init__(self, interval : int = 1):
        super().__init__();

        self._X = None;
        self._interval = interval;


    def _getParams(self) -> List:
        return [self._interval];


    def _setParams(self, value: Tuple):
        self._interval, = value;


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


    # def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
    #     if self.isTrainingMode:
    #         return super().forward(*data[:-1]);
    #     else:
    #         return super().forward(*data);


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


class BinaryChoiceModule(NetModuleBase):
    def __init__(self, p : float = 0.5):
        super().__init__();

        self._p = p;
        self._select1 = None;
        self._name = "BinaryChoice";


    @property
    def p(self) -> float:
        return self._p;


    @p.setter
    def p(self, value: float):
        self._p = value;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        X1, X2 = data;

        if self._p == 1:
            self._select1 = True;
        elif self._p == 0:
            self._select1 = False;
        else:
            self._select1 = float(np.random.rand(1)[0]) <= self._p;

        return (X1 if self._select1 else X2), ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        dY = dout[0];

        return (dY, 0) if self._select1 else (0, dY);


# Note: the module must have no inner connection in each step!
class RepeatedWrapper(NetModuleBase):
    def __init__(self, target : INetModule):
        super().__init__();

        self._modules = [];
        self._target = target;
        self._stepIndex = 0;
        self._name = f"Repeated{str(self._target)}";

        self._params.extend(target.params);
        self._grads.extend(target.grads);


    def _setTrainingMode(self, value : bool):
        self._target.isTrainingMode = value;
        for m in self._modules:
            m.isTrainingMode = value;


    def _setTrainingEpoch(self, value : int):
        self._target.trainingEpoch = value;
        for m in self._modules:
            m.trainingEpoch = value;


    def _setParams(self, value: List[np.ndarray]):
        self._target.params = value;
        for m in self._modules:
            m.params = value;


    def _reset(self):
        self._stepIndex = 0;
        self._target.reset();
        for m in self._modules:
            m.reset();


    def _copyMembers(self, module : INetModule, shareParams : bool):
        raise NotImplemented();


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
        while len(self._modules) < self._stepIndex + 1:
            self._modules.append(self._target.copy(True));

        Y = self._modules[self._stepIndex].forward(*data);
        self._stepIndex += 1;

        return Y;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
        self._stepIndex -= 1;

        module = self._modules[self._stepIndex];
        dX = module.backward(*dout);

        for i in range(len(self._grads)):
            self._grads[i] += module.grads[i];

        return dX;


class GeneratorDataIterator(IDataIterator):
    def __init__(self, generator : Generator, epochSize : int):
        self._generator = generator;
        self._epochSize = epochSize;


    def _iterate(self):
        for _ in range(self._epochSize):
            yield next(self._generator);


    def __iter__(self):
        return self._iterate();


    @property
    def totalIterations(self) -> int:
        return self._epochSize;


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
    def __init__(self, data : List[np.ndarray], batchSize : int, stepSize : int, shuffle : bool = False, randomOffset : bool = True):
        self._data = data;
        self._length = len(data[0]);
        self._batchSize = batchSize;
        self._stepSize = stepSize;
        self._totalIterations = 1;
        self._shuffle = shuffle;
        self._randomOffset = randomOffset;


    def _sequentialSample(self):
        offset = int(np.random.randint(0, self._stepSize)) if self._randomOffset else 0;
        totalLength = ((self._length - offset) // self._batchSize) * self._batchSize;
        data = [d[offset: offset + totalLength].reshape((self._batchSize, -1) + d.shape[1:]) for d in self._data];
        self._totalIterations = data[0].shape[1] // self._stepSize

        for i in range(0, self._totalIterations * self._stepSize, self._stepSize):
            yield tuple([d[:, i: i + self._stepSize] for d in data]);


    def _randomSample(self):
        offset = int(np.random.randint(0, self._stepSize)) if self._randomOffset else 0;
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
        return "MAE";


    @property
    def high(self) -> bool:
        return False;


    @property
    def accuracy(self) -> Optional[float]:
        # return math.sqrt(self._rss / self._totalCount) if self._totalCount > 0 else None;
        return (self._rss / self._totalCount) if self._totalCount > 0 else None;


    def fromLoss(self, lossValues : List[float] = None) -> bool:
        return False;


    def update(self, *data: np.ndarray):
        Y, T = data;
        # self._rss += float(np.sum(np.square(Y - T)));
        self._rss += float(np.sum(np.abs(Y - T)));
        self._totalCount += lengthExceptLastDimension(Y);


    def reset(self):
        self._rss = 0.0;
        self._totalCount = 0.0;


class ClassifierAccuracyEvaluator(INetAccuracyEvaluator):
    def __init__(self, sigmoid4BinaryClass : bool = True):
        self._rightCount = 0.0;
        self._totalCount = 0.0;
        self._sigmoid4BinaryClass = sigmoid4BinaryClass;


    @property
    def name(self) -> str:
        return "Classification Accuracy";


    @property
    def high(self) -> bool:
        return True;


    @property
    def accuracy(self) -> Optional[float]:
        return self._rightCount / self._totalCount if self._totalCount > 0 else None;


    def fromLoss(self, lossValues : List[float] = None) -> bool:
        return False;


    def update(self, *data: np.ndarray):
        Y, T = data;
        if Y.shape[-1] == 1:
            if self._sigmoid4BinaryClass:
                Y = sigmoid(Y);

            Y = np.column_stack((Y, 1 - Y));
            T = np.column_stack((T, 1 - T));

        if T.ndim > 1:
            self._rightCount += int(np.sum(np.argmax(Y, axis = -1) == np.argmax(T, axis = -1)));
        else:
            self._rightCount += int(np.sum(np.argmax(Y, axis = -1) == T));
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
    def high(self) -> bool:
        return False;


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


# the output distribution of VAE is Gaussian
class GaussianVAE(NetModelBase):
    def __init__(self, encoder: INetModule, decoder: INetModule, latentSize: int, sampleSize: int = 32, minStd : float = 1e-4):
        super().__init__(encoder, decoder);

        self._name = "GaussianVAE";
        self._encoder = encoder;
        self._decoder = decoder;
        self._latentSize = latentSize;
        self._sampleSize = sampleSize;
        self._minStd = minStd;
        self._z0 = None;  # the standard normal distribution
        self._V = None;
        self._U = None;


    def getFinalTag(self, T: np.ndarray) -> Optional[np.ndarray]:
        return None;


    def encode(self, X : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M, V = tuple(np.split(self._encoder.forward(X)[0], 2, axis = -1));

        return M, softplus(V) + self._minStd;


    def reparameterize(self, mu : np.ndarray, sigma : np.ndarray, epsilon : np.ndarray = None) -> np.ndarray:
        N, L, H = len(mu), self._sampleSize, self._latentSize;
        self._z0 = np.random.randn(N, L, H).astype(defaultDType) if epsilon is None else epsilon;
        Z = self._z0 * np.expand_dims(sigma, axis = 1) + np.expand_dims(mu, axis = 1);

        return Z;


    def decode(self, Z : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E, U = tuple(np.split(self._decoder.forward(Z)[0], 2, axis = -1));

        return E, softplus(U) + self._minStd;


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray]:
        X, epsilon = data if len(data) > 1 else (data[0], None);
        M, V = self.encode(X);
        Z = self.reparameterize(M, V, epsilon = epsilon);
        E, U = self.decode(Z);

        self._V, self._U = V, U;

        return X, M, V, E, U;


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray]:
        dX, dM, dV, dE, dU = dout;

        dZ = self._decoder.backward(np.concatenate((dE, dU * softplusGradient(Y = self._U)), axis = -1))[0];
        dM += np.sum(dZ, axis = 1);
        dV += np.sum(dZ * self._z0, axis = 1);
        dX += self._encoder.backward(np.concatenate((dM, dV * softplusGradient(Y = self._V)), axis = -1))[0];

        return dX, ;


    def generate(self, X : np.ndarray, L : int = 32) -> Tuple[np.ndarray, np.ndarray]:
        epsilon = np.random.randn(len(X), L, self._latentSize).astype(defaultDType);
        X, M, V, E, U = self.forward(X, epsilon);

        return E, U;


    def reconstructionProbability(self, X : np.ndarray, L : int = 32) -> np.ndarray:
        E, U = self.generate(X, L);

        # P = -reconstruction_loss
        P = -(2 * np.log(U) + (np.expand_dims(X, axis = 1) - E) ** 2 / U ** 2) / 2;
        P = np.sum(P, axis = tuple(range(2, len(P.shape))));
        P = np.mean(P, axis = 1);

        return P;


# the output distribution of VAE is Gaussian
class GaussianVAELoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._X = None;
        self._M = None;
        self._V = None;
        self._E = None;
        self._U = None;


    '''
    input: X, M, V, E, U
    Z is the latent variable.
    X: the input data, N*D
    M: E(Z|X), N*H
    V: Std(Z|X), N*H
    E: E(X|Z), N*L*D
    U: Std(X|Z), N*L*D

    q(z|x) ～ N(M, diagonal{ V^2 })
    p(z)   ～ N(0, I)
    p(x|z) ～ N(E, diagonal{ U^2 })
    L(x) = KL(q(z|x) || p(z)) - E(log(p(x|z)))
    '''
    def forward(self, *data: np.ndarray) -> float:
        X, M, V, E, U = data;
        N, L = E.shape[: 2];

        self._X, self._M, self._V, self._E, self._U = np.repeat(np.expand_dims(X, axis = 1), L, axis = 1), M, V, E, U;

        kl_loss = M ** 2 + V ** 2 - 2 * np.log(V);
        kl_loss = float(np.sum(kl_loss)) / (2 * N);

        reconstruction_loss = 2 * np.log(U) + (self._X - E) ** 2 / U ** 2;
        reconstruction_loss = float(np.sum(reconstruction_loss)) / (2 * L * N);

        self._loss = kl_loss + reconstruction_loss;
        # print(f"kl_loss: {kl_loss}, reconstruction_loss: {reconstruction_loss}");

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        N, L = self._E.shape[: 2];
        U2 = self._U ** 2;

        dX = np.sum((self._X - self._E) / U2, axis = 1) / (N * L);
        dM = self._M / N;
        dV = (self._V - 1 / self._V) / N;
        dE = (self._E - self._X) / U2 / (N * L);
        dU = (1 - (self._X - self._E) ** 2 / U2) / self._U / (N * L);

        return dX, dM, dV, dE, dU;


# the output distribution of VAE is Bernoulli
class BernoulliVAE(NetModelBase):
    def __init__(self, encoder: INetModule, decoder: INetModule, latentSize: int, sampleSize: int = 32, minStd : float = 1e-4):
        super().__init__(encoder, decoder);

        self._name = "BernoulliVAE";
        self._encoder = encoder;
        self._decoder = decoder;
        self._latentSize = latentSize;
        self._sampleSize = sampleSize;
        self._minStd = minStd;
        self._z0 = None;  # the standard normal distribution
        self._V = None;


    def getFinalTag(self, T: np.ndarray) -> Optional[np.ndarray]:
        return None;


    def encode(self, X : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M, V = tuple(np.split(self._encoder.forward(X)[0], 2, axis = -1));

        return M, softplus(V) + self._minStd;


    def reparameterize(self, mu: np.ndarray, sigma: np.ndarray, epsilon: np.ndarray = None) -> np.ndarray:
        N, L, H = len(mu), self._sampleSize, self._latentSize;
        self._z0 = np.random.randn(N, L, H).astype(defaultDType) if epsilon is None else epsilon;
        Z = self._z0 * np.expand_dims(sigma, axis = 1) + np.expand_dims(mu, axis = 1);

        return Z;


    def decode(self, Z : np.ndarray, toProbability : bool = False) -> np.ndarray:
        Y = self._decoder.forward(Z)[0];
        if toProbability:
            Y = sigmoid(Y);

        return Y;


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray]:
        X, epsilon = data if len(data) > 1 else (data[0], None);
        M, V = self.encode(X);
        Z = self.reparameterize(M, V, epsilon = epsilon);
        Y = self.decode(Z);

        self._V = V;

        return X, M, V, Y;


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray]:
        dX, dM, dV, dY = dout;

        dZ = self._decoder.backward(dY)[0];
        dM += np.sum(dZ, axis = 1);
        dV += np.sum(dZ * self._z0, axis = 1);
        dX += self._encoder.backward(np.concatenate((dM, dV * softplusGradient(Y = self._V)), axis = -1))[0];

        return dX, ;


    def generate(self, X : np.ndarray, L : int = 32, toProbability : bool = True) -> np.ndarray:
        epsilon = np.random.randn(len(X), L, self._latentSize).astype(defaultDType);
        X, M, V, Y = self.forward(X, epsilon);

        return sigmoid(Y) if toProbability else Y;


    def reconstructionProbability(self, X : np.ndarray, L : int = 32) -> np.ndarray:
        Y = self.generate(X, L, toProbability = False);

        # P = -reconstruction_loss
        P = -(softplus(Y) - np.expand_dims(X, axis = 1) * Y);
        P = np.sum(P, axis = tuple(range(2, len(P.shape))));
        P = np.mean(P, axis = 1);

        return P;


# the output distribution of VAE is Bernoulli
class BernoulliVAELoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._X = None;
        self._M = None;
        self._V = None;
        self._Y = None;


    '''
    input: X, M, V, Y
    Z is the latent variable.
    X: the input data, N*D
    M: E(Z|X), N*H
    V: Std(Z|X), N*H
    Y: logits, N*L*D

    q(z|x) ～ N(M, diagonal{ V^2 })
    p(z)   ～ N(0, I)
    p(x|z) ～ Bin(1, sigmoid(Y))
    L(x) = KL(q(z|x) || p(z)) - E(log(p(x|z)))
    '''
    def forward(self, *data: np.ndarray) -> float:
        X, M, V, Y = data;
        N, L = Y.shape[: 2];

        self._X, self._M, self._V, self._Y = np.repeat(np.expand_dims(X, axis = 1), L, axis = 1), M, V, Y;

        kl_loss = M ** 2 + V ** 2 - 2 * np.log(V);
        kl_loss = float(np.sum(kl_loss)) / (2 * N);

        # reconstruction_loss = (Y * (1 - self._X)) + np.log(1 + np.exp(-Y));
        reconstruction_loss = softplus(Y) - self._X * Y;
        reconstruction_loss = float(np.sum(reconstruction_loss)) / (L * N);

        self._loss = kl_loss + reconstruction_loss;
        # print(f"kl_loss: {kl_loss}, reconstruction_loss: {reconstruction_loss}");

        return self._loss;


    def backward(self) -> Tuple[np.ndarray]:
        N, L = self._Y.shape[: 2];

        dX = np.sum(-self._Y, axis = 1) / (N * L);
        dM = self._M / N;
        dV = (self._V - 1 / self._V) / N;
        dY = (sigmoid(self._Y) - self._X) / (N * L);

        return dX, dM, dV, dY;
