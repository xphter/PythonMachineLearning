# -*- coding: utf-8 -*-
#
# Copyright (C) 2024 XphteR, Inc. All Rights Reserved
#
# @Time    : 2025-09-09
# @Author  : Du Peng
# @Email   : 278770518@qq.com
# @File    : NN.py
######################################################


import abc;
import copy;
import math;
import time;
import datetime;
import functools;
import collections;

import matplotlib.pyplot as plt;

from typing import Union, List, Tuple, Callable, Any, Optional, Iterable, Generator
from Functions import *;


class INetContext(metaclass = abc.ABCMeta):
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
    def trainingIterations(self) -> int:
        pass;


    @trainingIterations.setter
    @abc.abstractmethod
    def trainingIterations(self, value: int):
        pass;


    @abc.abstractmethod
    def clean(self):
        pass;


class INetLoss(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass;


    @property
    @abc.abstractmethod
    def loss(self) -> float:
        pass;


    @abc.abstractmethod
    def forward(self, *data: np.ndarray) -> float:
        pass;


    @abc.abstractmethod
    def backward(self) -> Tuple[np.ndarray, ...]:
        pass;


class INetParam(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def value(self) -> np.ndarray:
        pass;


    @property
    @abc.abstractmethod
    def grad(self) -> np.ndarray:
        pass;


class INetParamHandler(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def onPreUpdate(self, param: INetParam, lr : float):
        pass;


    @abc.abstractmethod
    def onPostUpdate(self, param: INetParam, lr : float):
        pass;


class INetParamDefinition(INetParam):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass;


    @property
    @abc.abstractmethod
    def canDecay(self) -> bool:
        pass;


    @property
    @abc.abstractmethod
    def handler(self) -> Optional[INetParamHandler]:
        pass;


    @abc.abstractmethod
    def copy(self, share : bool) -> "INetParamDefinition":
        pass;


class INetState(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def value(self) -> Any:
        pass;


    @abc.abstractmethod
    def copy(self) -> "INetState":
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
    def epochStep(self, epoch : int):
        pass;


    # return the shadow parameters
    @abc.abstractmethod
    def updateStep(self, params : List[INetParamDefinition], context : INetContext) -> Optional[List[INetParamDefinition]]:
        pass;


class INetLrScheduler(INetOptimizer):
    @property
    @abc.abstractmethod
    def minEpoch(self) -> int:
        pass;


    @property
    @abc.abstractmethod
    def maxEpoch(self) -> Optional[int]:
        pass;


    @abc.abstractmethod
    def isAvailable(self, epoch : int) -> bool:
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
    def accuracy(self) -> float:
        pass;


    @abc.abstractmethod
    def fromLoss(self, lossValues : Optional[List[float]] = None) -> bool:
        pass;


    @abc.abstractmethod
    def update(self, loss : float, *data : np.ndarray):
        pass;


    @abc.abstractmethod
    def reset(self):
        pass;


class INetModule(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def context(self) -> INetContext:
        pass;


    @context.setter
    @abc.abstractmethod
    def context(self, value : INetContext):
        pass;


    @property
    @abc.abstractmethod
    def params(self) -> List[INetParamDefinition]:
        pass;


    @params.setter
    @abc.abstractmethod
    def params(self, value : List[INetParamDefinition]):
        pass;


    @property
    @abc.abstractmethod
    def states(self) -> List[INetState]:
        pass;


    @states.setter
    @abc.abstractmethod
    def states(self, value: List[INetState]):
        pass;


    @abc.abstractmethod
    def reset(self):
        pass;


    @abc.abstractmethod
    def clean(self):
        pass;


    @abc.abstractmethod
    def copy(self, shareParams : bool = False) -> "INetModule":
        pass;


    @abc.abstractmethod
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        pass;


    @abc.abstractmethod
    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        pass;


    @abc.abstractmethod
    def clearGrads(self):
        pass;


class NetFitCheckpointInfo:
    def __init__(self, epoch : int, params : List[INetParamDefinition], states : List[INetState]):
        self._epoch = epoch;
        self._params = params;
        self._states = states;
    

    def __repr__(self) -> str:
        return self.__str__();


    def __str__(self) -> str:
        return f"epoch {self._epoch}";


    @property
    def epoch(self) -> int:
        return self._epoch;


    @property
    def params(self) -> List[INetParamDefinition]:
        return self._params;


    @property
    def states(self) -> List[INetState]:
        return self._states;


class INetFitResult(metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def trainingIterationLoss(self) -> List[float]:
        pass;
    

    @property
    @abc.abstractmethod
    def trainingEpochLoss(self) -> List[float]:
        pass;


    @property
    @abc.abstractmethod
    def trainingEpochAccuracy(self) -> List[float]:
        pass;


    @property
    @abc.abstractmethod
    def testEpochLoss(self) -> List[float]:
        pass;


    @property
    @abc.abstractmethod
    def testEpochAccuracy(self) -> List[float]:
        pass;


    @property
    @abc.abstractmethod
    def finalTrainingLoss(self) -> Optional[float]:
        pass;


    @property
    @abc.abstractmethod
    def finalTrainingAccuracy(self) -> Optional[float]:
        pass;


    @property
    @abc.abstractmethod
    def finalTestLoss(self) -> Optional[float]:
        pass;


    @property
    @abc.abstractmethod
    def finalTestAccuracy(self) -> Optional[float]:
        pass;


    @property
    @abc.abstractmethod
    def accuracyName(self) -> Optional[str]:
        pass;
    

    @property
    @abc.abstractmethod
    def lossName(self) -> Optional[str]:
        pass;


    @property
    @abc.abstractmethod
    def checkpoints(self) -> List[NetFitCheckpointInfo]:
        pass;


class INetModel(INetModule, metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def getFinalTag(self, T : np.ndarray) -> Optional[np.ndarray]:
        pass;


    @abc.abstractmethod
    def eval(self, lossFunc: INetLoss, evaluator: INetAccuracyEvaluator, lossValues: Optional[List[float]] = None, iterator: Optional[Iterable] = None) -> Tuple[float, float]:
        pass;


    @abc.abstractmethod
    def fit(self, trainingIterator : IDataIterator, lossFunc: INetLoss, optimizer: INetOptimizer, maxEpoch : int, testIterator : Optional[IDataIterator] = None,
            evaluator: Optional[INetAccuracyEvaluator] = None, evalEpoch : bool = True, evalIterations : Optional[int] = None, evalTrainingData : bool = False, evalTestData : bool = True,
            checkpoints : Optional[List[int]] = None, minEpoch : Optional[int] = None, plot = False) -> INetFitResult:
        pass;


    @abc.abstractmethod
    def predict(self, iterator : IDataIterator) -> Iterable:
        pass;


    @abc.abstractmethod
    def predictOne(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        pass;


class INetAttentionModule(INetModule, metaclass = abc.ABCMeta):
    @property
    @abc.abstractmethod
    def attentionWeight(self) -> Optional[np.ndarray]:
        pass;


class NetUtility:
    @staticmethod
    def plotFitResult(result : INetFitResult):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8))

        ax0.set_xlabel("iteration");
        ax0.set_ylabel(result.lossName);
        ax0.set_title(f"total {len(result.trainingIterationLoss)} iterations");
        ax0.plot(result.trainingIterationLoss, "-k", label = "training iteration loss");

        ax1.set_xlabel("epoch");
        ax1.set_ylabel(result.lossName);
        ax1.set_title(f"total {len(result.trainingEpochLoss)} epochs");
        if result.trainingEpochLoss is not None and len(result.trainingEpochLoss) > 0:
            ax1.plot(result.trainingEpochLoss, "o-g", label = "training epoch loss");
        if result.testEpochLoss is not None and len(result.testEpochLoss) > 0:
            ax1.plot(result.testEpochLoss, "o-b", label = "test epoch loss");

        ax2 = ax1.twinx();
        ax2.set_ylabel(result.accuracyName);
        if result.trainingEpochAccuracy is not None and  len(result.trainingEpochAccuracy) > 0:
            ax2.plot(result.trainingEpochAccuracy, "D-m", label = f"training epoch accuracy");
        if result.testEpochAccuracy is not None and len(result.testEpochAccuracy) > 0:
            ax2.plot(result.testEpochAccuracy, "D-r", label = f"test epoch accuracy");

        fig.legend(loc = "upper right", bbox_to_anchor = (1, 1), bbox_transform = ax0.transAxes);
        fig.tight_layout();
        plt.show(block = True);
        plt.close();


class NetContext(INetContext):
    def __init__(self):
        self._isTrainingMode = False;
        self._trainingEpoch = 0;
        self._trainingIterations = 0;


    @property
    def isTrainingMode(self) -> bool:
        return self._isTrainingMode;


    @isTrainingMode.setter
    def isTrainingMode(self, value: bool):
        self._isTrainingMode = value;


    @property
    def trainingEpoch(self) -> int:
        return self._trainingEpoch;


    @trainingEpoch.setter
    def trainingEpoch(self, value: int):
        self._trainingEpoch = value;


    @property
    def trainingIterations(self) -> int:
        return self._trainingIterations;


    @trainingIterations.setter
    def trainingIterations(self, value: int):
        self._trainingIterations = value;


    def clean(self):
        self._trainingEpoch = 0;
        self._trainingIterations = 0;


class NetParam(INetParam):
    def __init__(self, value : np.ndarray, grad : Optional[np.ndarray] = None):
        self._value = value;
        self._grad = np.zeros_like(value, dtype = value.dtype) if grad is None else grad;


    @property
    def value(self) -> np.ndarray:
        return self._value;


    @property
    def grad(self) -> np.ndarray:
        return self._grad;


class NetParamDefinition(NetParam, INetParamDefinition):
    def __init__(self, name : str, value : np.ndarray, handler: Optional[INetParamHandler] = None, grad : Optional[np.ndarray] = None, canDecay : bool = True):
        super().__init__(value, grad);

        self._name = name;
        self._handler = handler;
        self._canDecay = canDecay;
    

    def __repr__(self) -> str:
        return self.__str__();


    def __str__(self) -> str:
        return f"{self._name}({'*'.join([str(item) for item in self._value.shape])})";


    @property
    def name(self) -> str:
        return self._name;


    @property
    def canDecay(self) -> bool:
        return self._canDecay;


    @property
    def handler(self) -> Optional[INetParamHandler]:
        return self._handler;


    def copy(self, share : bool) -> INetParamDefinition:
        if share:
            return NetParamDefinition(self._name, self._value, self._handler, canDecay = self._canDecay);
        else:
            return NetParamDefinition(self._name, np.copy(self._value), self._handler, canDecay = self._canDecay);


class NetValueState(INetState):
    def __init__(self, value = None):
        self._value = value;


    @property
    def value(self) -> Any:
        return self._value;


    def copy(self) -> INetState:
        value = self._value;
        if value is not None:
            if isinstance(value, np.ndarray):
                value = np.copy(value);

            value = copy.deepcopy(value);

        return NetValueState(value);


class NetModuleBase(INetModule, metaclass = abc.ABCMeta):
    def __init__(self):
        self._name = "";
        self._context = NetContext();
        self._params : List[INetParamDefinition] = [];
        self._states: List[INetState] = [];


    def __repr__(self) -> str:
        return self.__str__();


    def __str__(self) -> str:
        return self._name;


    @property
    def context(self) -> INetContext:
        return self._context;


    @context.setter
    def context(self, value: INetContext):
        self._context = value;
        self._setContext(value);


    @property
    def params(self) -> List[INetParamDefinition]:
        return self._params;


    @params.setter
    def params(self, value: List[INetParamDefinition]):
        self._params = value;
        self._setParams(value);


    @property
    def states(self) -> List[INetState]:
        return self._states;


    @states.setter
    def states(self, value: List[INetState]):
        self._states = value;
        self._setStates(value);


    def _setContext(self, context : INetContext):
        pass;


    def _setParams(self, params: List[INetParamDefinition]):
        pass;


    def _setStates(self, states: List[INetState]):
        pass;


    def _reset(self):
        pass;


    def _clean(self):
        pass;


    def _copyMembers(self, module : INetModule, shareParams : bool):
        pass;


    def _copyParams(self, module : INetModule, shareParams : bool):
        module.params = [p.copy(shareParams) for p in self.params];


    def _copyStates(self, module : INetModule):
        module.states = [s.copy() for s in self.states];


    def reset(self):
        self._reset();


    def clean(self):
        self.reset();

        self._clean();


    def copy(self, shareParams : bool = False) -> INetModule:
        # has already copied the net context object
        module = copy.copy(self);

        self._copyMembers(module, shareParams);
        self._copyParams(module, shareParams);
        self._copyStates(module);

        return module;


    def clearGrads(self):
        for p in self.params:
            p.grad[...] = 0;


class AggregateNetModule(NetModuleBase):
    def __init__(self, *modules : INetModule):
        super().__init__();

        self._modules = modules;
        for m in modules:
            m.context = self._context;
            self._params.extend(m.params);
            self._states.extend(m.states);

        self._selfParamsNum, self._selfStatesNum = 0, 0;
        self._name = "  -->  ".join([str(m) for m in modules]);


    @property
    def modules(self) -> Tuple[INetModule, ...]:
        return self._modules;


    def _initSelfParams(self, params : Optional[List[INetParamDefinition]]):
        if params is None:
            params = [];

        modulesParamsNum = sum([len(m.params) for m in self._modules]);

        self._selfParamsNum = len(params);
        self._params = params + self._params[len(self._params) - modulesParamsNum: ];


    def _initSelfStates(self, states : Optional[List[INetState]]):
        if states is None:
            states = [];

        modulesStatesNum = sum([len(m.states) for m in self._modules]);

        self._selfStatesNum = len(states);
        self._states = states + self._states[len(self._states) - modulesStatesNum:];


    def _setContext(self, context : INetContext):
        for m in self._modules:
            m.context = context;


    def _setSelfParams(self, params: List[INetParamDefinition]):
        pass;


    def _setParams(self, params: List[INetParamDefinition]):
        i = 0;

        if self._selfParamsNum > 0:
            self._setSelfParams(params[i : i + self._selfParamsNum]);
            i += self._selfParamsNum;

        for m in self._modules:
            if (n := len(m.params)) == 0:
                continue;

            m.params = params[i: i + n];
            i += n;


    def _setSelfStates(self, states: List[INetState]):
        pass;


    def _setStates(self, states: List[INetState]):
        i = 0;

        if self._selfStatesNum > 0:
            self._setSelfStates(states[i : i + self._selfStatesNum]);
            i += self._selfStatesNum;

        for m in self._modules:
            if (n := len(m.states)) == 0:
                continue;

            m.states = states[i: i + n];
            i += n;


    def _reset(self):
        for m in self._modules:
            m.reset();


    def _clean(self):
        for m in self._modules:
            m.clean();


    def _copyMembers(self, module : INetModule, shareParams : bool):
        module._modules = tuple([m.copy(shareParams) for m in self.modules]);


    def _copyParams(self, module : INetModule, shareParams : bool):
        module._params = [];

        if self._selfParamsNum > 0:
            module.params.extend([p.copy(shareParams) for p in self.params[: self._selfParamsNum]]);

        for m in module.modules:
            module.params.extend(m.params);


    def _copyStates(self, module : INetModule):
        module._states = [];

        if self._selfStatesNum > 0:
            module.states.extend([s.copy() for s in self.states[: self._selfStatesNum]]);

        for m in module.modules:
            module.states.extend(m.states);


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        Y = data;

        for m in self._modules:
            Y = m.forward(*Y);

        return Y;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dX = dout;

        for m in reversed(self._modules):
            dX = m.backward(*dX);

        return dX;


class NetFitResult(INetFitResult):
    def __init__(self, trainingIterationLoss : List[float], trainingEpochLoss : List[float], trainingEpochAccuracy : List[float],
                 testEpochLoss : List[float], testEpochAccuracy : List[float],
                 finalTrainingLoss : Optional[float] = None, finalTrainingAccuracy : Optional[float] = None,
                 finalTestLoss : Optional[float] = None, finalTestAccuracy : Optional[float] = None,
                 accuracyName : Optional[str] = None, lossName : Optional[str] = None, checkpoints : Optional[List[NetFitCheckpointInfo]] = None):
        self._trainingIterationLoss = trainingIterationLoss;
        self._trainingEpochLoss = trainingEpochLoss;
        self._trainingEpochAccuracy = trainingEpochAccuracy;
        self._testEpochLoss = testEpochLoss;
        self._testEpochAccuracy = testEpochAccuracy;
        self._finalTrainingLoss = finalTrainingLoss;
        self._finalTrainingAccuracy = finalTrainingAccuracy;
        self._finalTestLoss = finalTestLoss;
        self._finalTestAccuracy = finalTestAccuracy;
        self._accuracyName = accuracyName;
        self._lossName = lossName;
        self._checkpoints = checkpoints if checkpoints is not None else [];
    

    @property
    def trainingIterationLoss(self) -> List[float]:
        return self._trainingIterationLoss;


    @property
    def trainingEpochLoss(self) -> List[float]:
        return self._trainingEpochLoss;


    @property
    def trainingEpochAccuracy(self) -> List[float]:
        return self._trainingEpochAccuracy;


    @property
    def testEpochLoss(self) -> List[float]:
        return self._testEpochLoss;


    @property
    def testEpochAccuracy(self) -> List[float]:
        return self._testEpochAccuracy;


    @property
    def finalTrainingLoss(self) -> Optional[float]:
        return self._finalTrainingLoss;


    @property
    def finalTrainingAccuracy(self) -> Optional[float]:
        return self._finalTrainingAccuracy;


    @property
    def finalTestLoss(self) -> Optional[float]:
        return self._finalTestLoss;


    @property
    def finalTestAccuracy(self) -> Optional[float]:
        return self._finalTestAccuracy;


    @property
    def accuracyName(self) -> Optional[str]:
        return self._accuracyName;


    @property
    def lossName(self) -> Optional[str]:
        return self._lossName;


    @property
    def checkpoints(self) -> List[NetFitCheckpointInfo]:
        return self._checkpoints;


class NetModelBase(AggregateNetModule, INetModel, metaclass = abc.ABCMeta):
    def __init__(self, *modules : INetModule):
        super().__init__(*modules);
        
        self._shadowParams : Optional[List[INetParamDefinition]] = None;


    def getFinalTag(self, T : np.ndarray) -> Optional[np.ndarray]:
        return T;


    def eval(self, lossFunc : INetLoss, evaluator : INetAccuracyEvaluator, lossValues : Optional[List[float]] = None, iterator : Optional[Iterable] = None) -> Tuple[float, float]:
        self.context.isTrainingMode = False;

        backupParams : Optional[List[INetParamDefinition]] = None;
        if self._shadowParams is not None:
            backupParams = self.params;
            self.params = self._shadowParams;

        try:
            evaluator.reset();

            if not (lossValues is not None and evaluator.fromLoss(lossValues)) and iterator is not None:
                lossValues = [];

                for data in iterator:
                    Y = self.forward(*data);
                    T = self.getFinalTag(data[-1]);
                    loss = lossFunc.forward(*Y, T) if T is not None else lossFunc.forward(*Y);

                    lossValues.append(loss);
                    evaluator.update(loss, *Y, T) if T is not None else evaluator.update(loss, *Y);

                evaluator.fromLoss(lossValues);
        finally:
            if backupParams is not None:
                self.params = backupParams;

            self.reset();
            self.context.isTrainingMode = True;

        return sum(lossValues) / len(lossValues) if lossValues is not None else 0.0, evaluator.accuracy;


    def fit(self, trainingIterator: IDataIterator, lossFunc: INetLoss, optimizer: INetOptimizer, maxEpoch: int, testIterator: Optional[IDataIterator] = None,
            evaluator: Optional[INetAccuracyEvaluator] = None, evalEpoch: bool = True, evalIterations: Optional[int] = None, evalTrainingData: bool = False, evalTestData: bool = True,
            checkpoints : Optional[List[int]] = None, minEpoch : Optional[int] = None, plot = False) -> INetFitResult:
        lossValues = [];
        trainingIterationLoss = [];
        trainingEpochLoss = [];
        trainingEpochAccuracy = [];
        testEpochLoss = [];
        testEpochAccuracy : List[float] = [];
        finalTrainingLoss, finalTrainingAccuracy = None, None;
        finalTestLoss, finalTestAccuracy = None, None;
        checkpointData : List[Optional[NetFitCheckpointInfo]] = [None] * maxEpoch;

        startTime = time.time();
        self._shadowParams = None;
        print(f"[{datetime.datetime.now()}] start to train model {self}");

        self.clean();
        self.context.clean();

        for epoch in range(maxEpoch):
            lossValues.clear();
            if evaluator is not None:
                evaluator.reset();

            self.context.isTrainingMode = True;
            self.context.trainingEpoch = epoch;

            optimizer.epochStep(epoch);

            for data in trainingIterator:
                self.context.trainingIterations += 1;

                Y = self.forward(*data);
                T = self.getFinalTag(data[-1]);
                loss = lossFunc.forward(*Y, T) if T is not None else lossFunc.forward(*Y);

                if not math.isfinite(loss):
                    raise OverflowError("training fail: the return value of loss function is NaN or infinity");

                lossValues.append(loss);
                trainingIterationLoss.append(loss);
                if evaluator is not None:
                    evaluator.update(loss, *Y, T) if T is not None else evaluator.update(loss, *Y);

                self.backward(*lossFunc.backward());
                self._shadowParams = optimizer.updateStep(self.params, self.context);
                self.clearGrads();

                if evaluator is not None and evalIterations is not None and len(lossValues) % evalIterations == 0:
                    loss, accuracy = self.eval(lossFunc, evaluator, lossValues[-evalIterations:], None);
                    if accuracy is not None:
                        print(f"epoch {epoch}, iterations: {len(lossValues)} / {trainingIterator.totalIterations}, training loss: {loss}, training {evaluator.name}: {accuracy}, elapsed time: {int(time.time() - startTime)}s");

            self.reset();
            trainingEpochLoss.append(sum(lossValues) / len(lossValues));
            if evaluator is not None:
                evaluator.fromLoss(lossValues);
                trainingEpochAccuracy.append(evaluator.accuracy);

            if evaluator is not None and evalEpoch:
                if evalTrainingData:
                    print("evaluating training data...");
                    loss, accuracy = self.eval(lossFunc, evaluator, lossValues, trainingIterator);
                    trainingEpochLoss[-1] = loss;
                    trainingEpochAccuracy[-1] = accuracy;
                if testIterator is not None and evalTestData:
                    print("evaluating test data...");
                    loss, accuracy = self.eval(lossFunc, evaluator, None, testIterator);
                    testEpochLoss.append(loss);
                    testEpochAccuracy.append(accuracy);
            
            if checkpoints is not None and epoch in checkpoints or minEpoch is not None and len(testEpochAccuracy) > 0:
                checkpointData[epoch] = NetFitCheckpointInfo(
                    epoch,
                    [p.copy(False) for p in (self._shadowParams if self._shadowParams is not None else self.params)],
                    [s.copy() for s in self.states]);

            trainingMessage = f", training {evaluator.name}: {trainingEpochAccuracy[-1]}" if len(trainingEpochAccuracy) > 0 and evaluator is not None else "";
            testMessage = f", test loss: {testEpochLoss[-1]}, test {evaluator.name}: {testEpochAccuracy[-1]}" if len(testEpochAccuracy) > 0 and evaluator is not None else "";
            print(f"epoch {epoch}, training loss: {trainingEpochLoss[-1]}{trainingMessage}{testMessage}, elapsed time: {int(time.time() - startTime)}s");

        if minEpoch is not None and len(testEpochAccuracy) > 0 and evaluator is not None:
            index = np.argmax(np.array(testEpochAccuracy[minEpoch:])) if evaluator.high else np.argmin(np.array(testEpochAccuracy[minEpoch:]));
            print(f"the final params were training on epoch {minEpoch + int(index)}");

            checkpointInfo : NetFitCheckpointInfo = checkpointData[minEpoch + int(index)]; # type: ignore
            self.params = checkpointInfo.params;
            self.states = checkpointInfo.states;
        elif self._shadowParams is not None:
            self.params = self._shadowParams;

        if evaluator is not None:
            print("evaluating final training data...");
            finalTrainingLoss, finalTrainingAccuracy = self.eval(lossFunc, evaluator, None, trainingIterator);
            print(f"the final training accuracy, loss: {finalTrainingLoss}, {evaluator.name}: {finalTrainingAccuracy}");

            if testIterator is not None:
                print("evaluating final test data...");
                finalTestLoss, finalTestAccuracy = self.eval(lossFunc, evaluator, None, testIterator);
                print(f"the final test accuracy, loss: {finalTestLoss}, {evaluator.name}: {finalTestAccuracy}");

        self.context.isTrainingMode = False;
        print(f"[{datetime.datetime.now()}] complete to train model, elapsed time: {int(time.time() - startTime)}s");

        result = NetFitResult(trainingIterationLoss, trainingEpochLoss, trainingEpochAccuracy, testEpochLoss, testEpochAccuracy,
                              finalTrainingLoss, finalTrainingAccuracy, finalTestLoss, finalTestAccuracy,
                              evaluator.name if evaluator is not None else None, lossFunc.name,
                              [item for item in checkpointData if item is not None and item.epoch in checkpoints] if checkpoints is not None else None);

        if plot:
            NetUtility.plotFitResult(result);

        return result;


    def predict(self, iterator : IDataIterator) -> Iterable:
        for data in iterator:
            yield self.forward(*data);


    def predictOne(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        return self.forward(*data);


class NetLossBase(INetLoss, metaclass = abc.ABCMeta):
    def __init__(self):
        self._loss = 0.0;


    @property
    def loss(self) -> float:
        return self._loss;


class NetOptimizerBase(INetOptimizer, metaclass = abc.ABCMeta):
    def __init__(self, lr : float, weightDecay : float = 0.0, decoupledDecay : bool = False):
        self._lr = lr;

        self._weightDecay = weightDecay;
        self._decoupledDecay = decoupledDecay;
        self._canDecay = weightDecay > 0.0;


    @property
    def learningRate(self) -> float:
        return self._lr;


    @learningRate.setter
    def learningRate(self, value : float):
        self._lr = value;
    

    @property
    def weightDecay(self) -> float:
        return self._weightDecay;


    @weightDecay.setter
    def weightDecay(self, value : float):
        self._weightDecay = value;
    

    @property
    def decoupledDecay(self) -> bool:
        return self._decoupledDecay;


    @decoupledDecay.setter
    def decoupledDecay(self, value : bool):
        self._decoupledDecay = value;


    def _onPreUpdate(self, params : List[INetParamDefinition]):
        handler: Optional[INetParamHandler] = None;

        for p in params:
            if self._canDecay and p.canDecay:
                value = p.value;
                if self._decoupledDecay:
                    value -= self._lr * self._weightDecay * value;
                else:
                    grad = p.grad;
                    grad += self._weightDecay * value;

            if (handler := p.handler) is not None:
                handler.onPreUpdate(p, self.learningRate);


    @abc.abstractmethod
    def _onUpdate(self, params : List[INetParamDefinition]):
        pass;


    def _onPostUpdate(self, params : List[INetParamDefinition]):
        handler: Optional[INetParamHandler] = None;

        for p in params:
            if (handler := p.handler) is None:
                continue;

            handler.onPostUpdate(p, self.learningRate);


    def epochStep(self, epoch : int):
        pass;


    def updateStep(self, params : List[INetParamDefinition], context : INetContext) -> Optional[List[INetParamDefinition]]:
        self._onPreUpdate(params);
        self._onUpdate(params);
        self._onPostUpdate(params);

        return None;


class NetLrSchedulerBase(INetLrScheduler, metaclass = abc.ABCMeta):
    def __init__(self, baseLr : float, minEpoch : int = 0, maxEpoch : Optional[int] = None):
        self._baseLr = baseLr;
        self._currentLr = baseLr;
        self._minEpoch = minEpoch;
        self._maxEpoch = maxEpoch;


    @property
    def learningRate(self) -> float:
        return self._currentLr;


    @learningRate.setter
    def learningRate(self, value: float):
        raise ValueError("not support to set lr");


    @property
    def minEpoch(self) -> int:
        return self._minEpoch;


    @property
    def maxEpoch(self) -> Optional[int]:
        return self._maxEpoch;


    def isAvailable(self, epoch : int) -> bool:
        return (self._minEpoch <= epoch <= self._maxEpoch) if self._maxEpoch is not None else (self._minEpoch <= epoch);


    def updateStep(self, params: List[INetParamDefinition], context: INetContext) -> Optional[List[INetParamDefinition]]:
        return None;


class FunctionalNetModule(NetModuleBase):
    def __init__(self, name : str, forwardFunc : Callable[[np.ndarray], np.ndarray], backwardFunc : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
        super().__init__();

        self._name = name;
        self._forwardFunc = forwardFunc;
        self._backwardFunc = backwardFunc;
        self._X, self._Y = (), ();


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        self._X = data;
        self._Y = tuple(self._forwardFunc(X) for X in data);

        return self._Y;


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dX = tuple(self._backwardFunc(X, Y, dY) for X, Y, dY in zip(self._X, self._Y, dout));

        return dX;


class ReluLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._mask = None;
        self._name = "ReLU";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._mask = (X > 0).astype(X.dtype);

        return relu(X), ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dX = dY * self._mask;

        return dX, ;


class PReluLayer(NetModuleBase):
    def __init__(self, outputSize : Optional[int] = None, beta : Optional[Union[float, np.ndarray]] = None):
        super().__init__();

        self._X = np.empty(0);
        self._name = "PReLU";

        if beta is not None:
            if isinstance(beta, np.ndarray):
                self._beta = beta;
            else:
                self._beta = np.array([float(beta)]);
        else:
            self._beta = sigmoid(np.random.randn(outputSize if outputSize is not None else 1).astype(defaultDType));

        self._params.append(NetParamDefinition("slope", self._beta, canDecay = False));


    @property
    def beta(self) -> np.ndarray:
        return self._beta;


    def _setParams(self, params: List[INetParamDefinition]):
        self._beta = params[0].value;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        self._X = data[0];
        Y = prelu(self._X, self._beta);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dX, dBeta = preluGradient(self._X, self._beta);

        dX *= dY;
        dBeta *= dY;
        if len(self._beta) == 1:
            dBeta = np.sum(dBeta);
        else:
            dBeta = np.sum(dBeta, axis = tuple(range(len(dBeta.shape) - 1)));

        self._params[0].grad[...] += dBeta;

        return dX, ;


class PReluNDLayer(PReluLayer):
    def __init__(self, dimensionNum : int, channelNum : int, beta : Optional[Union[float, np.ndarray]] = None):
        self._channelAxis = -(max(1, int(dimensionNum)) + 1);

        super().__init__(outputSize = channelNum, beta = beta);

        self._name = f"PRelu{dimensionNum}D {channelNum}";
    

    def _checkInput(self, X : np.ndarray):
        pass;


    # 1D X shape: (batch_size, input_size) or (batch_size, channel_num, sequence_length)
    # 2D X shape: (batch_size, channel_num, image_height, image_width)
    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];

        self._checkInput(X);
        
        if X.ndim > 2:
            Y, = super().forward(X.swapaxes(self._channelAxis, -1));
            Y = Y.swapaxes(self._channelAxis, -1);
        else:
            Y, = super().forward(X);
        
        return Y, ;
    

    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        if dY.ndim > 2:
            dX, = super().backward(dY.swapaxes(self._channelAxis, -1));
            dX = dX.swapaxes(self._channelAxis, -1);
        else:
            dX, = super().backward(dY);
        
        return dX, ;


class PRelu1DLayer(PReluNDLayer):
    def __init__(self, channelNum : int, beta : Optional[Union[float, np.ndarray]] = None):
        super().__init__(1, channelNum = channelNum, beta = beta);


class PRelu2DLayer(PReluNDLayer):
    def __init__(self, channelNum : int, beta : Optional[Union[float, np.ndarray]] = None):
        super().__init__(2, channelNum = channelNum, beta = beta);
    

    def _checkInput(self, X : np.ndarray):
        if X.ndim < 4:
            raise ValueError("the input shape of PRelu2D should be 4 at least");


class SoftplusLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._X = np.empty(0);
        self._name = "Softplus";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        self._X = data[0];
        Y = softplus(self._X);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dX = dY * softplusGradient(self._X);

        return dX, ;


class SwishLayer(NetModuleBase):
    def __init__(self, outputSize : Optional[int] = None, beta : Optional[Union[float, np.ndarray]] = None):
        super().__init__();

        self._X = np.empty(0);
        self._Y = np.empty(0);
        self._S  = np.empty(0);
        self._name = "Swish";

        if beta is not None:
            if isinstance(beta, np.ndarray):
                self._beta = beta;
            else:
                self._beta = np.array([float(beta)]);
        else:
            self._beta = sigmoid(np.random.randn(outputSize if outputSize is not None else 1).astype(defaultDType));

        self._params.append(NetParamDefinition("slop", self._beta, canDecay = False));


    @property
    def beta(self) -> np.ndarray:
        return self._beta;


    def _setParams(self, params: List[INetParamDefinition]):
        self._beta = params[0].value;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        self._X = data[0];
        self._S, self._Y = swish(self._X, self._beta);

        return self._Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dX, dBeta = swishGradient(self._Y, self._S, self._X, self._beta);

        dX *= dY;
        dBeta *= dY;
        if len(self._beta) == 1:
            dBeta = np.sum(dBeta);
        else:
            dBeta = np.sum(dBeta, axis = tuple(range(len(dBeta.shape) - 1)));

        self._params[0].grad[...] += dBeta;

        return dX, ;


class SwishNDLayer(SwishLayer):
    def __init__(self, dimensionNum : int, channelNum : int, beta : Optional[Union[float, np.ndarray]] = None):
        self._channelAxis = -(max(1, int(dimensionNum)) + 1);

        super().__init__(outputSize = channelNum, beta = beta);

        self._name = f"Swish{dimensionNum}D {channelNum}";
    

    def _checkInput(self, X : np.ndarray):
        pass;


    # 1D X shape: (batch_size, input_size) or (batch_size, channel_num, sequence_length)
    # 2D X shape: (batch_size, channel_num, image_height, image_width)
    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];

        self._checkInput(X);
        
        if X.ndim > 2:
            Y, = super().forward(X.swapaxes(self._channelAxis, -1));
            Y = Y.swapaxes(self._channelAxis, -1);
        else:
            Y, = super().forward(X);
        
        return Y, ;
    

    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        if dY.ndim > 2:
            dX, = super().backward(dY.swapaxes(self._channelAxis, -1));
            dX = dX.swapaxes(self._channelAxis, -1);
        else:
            dX, = super().backward(dY);
        
        return dX, ;


class Swish1DLayer(SwishNDLayer):
    def __init__(self, channelNum : int, beta : Optional[Union[float, np.ndarray]] = None):
        super().__init__(1, channelNum = channelNum, beta = beta);


class Swish2DLayer(SwishNDLayer):
    def __init__(self, channelNum : int, beta : Optional[Union[float, np.ndarray]] = None):
        super().__init__(2, channelNum = channelNum, beta = beta);
    

    def _checkInput(self, X : np.ndarray):
        if X.ndim < 4:
            raise ValueError("the input shape of Swish2D should be 4 at least");


class SiluLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._X = np.empty(0);
        self._S = np.empty(0);
        self._name = "SiLU";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        self._X = data[0];
        self._S = sigmoid(self._X);

        return self._X * self._S, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dX = dY * self._S * (1 + self._X * (1 - self._S));

        return dX, ;


# GELU(x) = 0.5 * x * ( 1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ) )
class GeluLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._X = np.empty(0);
        self._G, self._F = np.empty(0), np.empty(0);
        self._alpha, self._beta = math.sqrt(2.0 / math.pi), 0.044715;
        self._3beta = 3 * self._beta;
        self._name = "GELU";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        self._X = data[0];
        self._F = self._alpha * (self._X + self._beta * np.power(self._X, 3));
        self._G = 1 + np.tanh(self._F);

        return 0.5 * self._X * self._G, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dG = self._alpha * (1 + self._3beta * np.power(self._X, 2)) / np.square(np.cosh(self._F));
        dX = dY * 0.5 * (self._G + self._X * dG);

        return dX, ;


class MaxoutLayer(NetModuleBase):
    def __init__(self, k : int = 2):
        super().__init__();

        self._M = None;
        self._K = max(2, int(k));
        self._name = f"Maxout {self._K}";


    @property
    def K(self) -> int:
        return self._K;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        if X.shape[-1] % self._K != 0:
            raise ValueError(f"{self._name}: the last dimension of input shape {X.shape} is not a multiple of group size {self._K}");

        H = int(X.shape[-1] // self._K);
        Z = np.reshape(X, X.shape[: -1] + (H, self._K));

        if self.context.isTrainingMode:
            E = np.zeros_like(Z, dtype = np.int32) + np.arange(self._K, dtype = np.int32);
            M = E == np.argmax(Z, axis = -1, keepdims = True);
            Y = np.reshape(Z[M], X.shape[: -1] + (H, ));
            self._M = (M + 0).astype(defaultDType);
        else:
            Y = np.amax(Z, axis = -1);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dY1 = np.reshape(dY, dY.shape + (1, )) * self._M;
        dX = np.reshape(dY1, dY.shape[: -1] + (-1, ));

        return dX, ;


class SigmoidLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._Y = np.empty(0);
        self._name = "Sigmoid";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._Y = sigmoid(X);

        return self._Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dX = dY * sigmoidGradient(self._Y);

        return dX, ;


class TanhLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._Y = np.empty(0);
        self._name = "Tanh";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._Y = tanh(X);

        return self._Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dX = dY * tanhGradient(self._Y);

        return dX, ;


class DropoutLayer(NetModuleBase):
    def __init__(self, dropoutRatio = 0.5, reuseMask : bool = False):
        super().__init__();

        self._mask = None;
        self._reuseMask = reuseMask;
        self._dropoutRatio = max(0.0, min(1.0, float(dropoutRatio)));
        self._name = f"Dropout {dropoutRatio}";
    

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask;
    

    def _checkInput(self, X : np.ndarray) -> bool:
        return True;


    def _getMask(self, shape : Tuple[int, ...], dtype) -> np.ndarray:
        return (np.random.rand(*shape) > self._dropoutRatio).astype(dtype) / (1.0 - self._dropoutRatio);


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];

        if not self._checkInput(X):
            raise ValueError(f"the input shape {X.shape} is invalid");

        if self.context.isTrainingMode:
            if self._dropoutRatio == 0.0:
                return X, ;
            if self._dropoutRatio == 1.0:
                return np.zeros_like(X, dtype = X.dtype), ;

            if self._mask is None or not self._reuseMask:
                self._mask = self._getMask(X.shape, X.dtype);

            return X * self._mask, ;
        else:
            return X, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        if self._dropoutRatio == 0.0:
            return dY, ;
        if self._dropoutRatio == 1.0:
            return np.zeros_like(dY, dtype = dY.dtype), ;

        dX = dY * self._mask;
        return dX, ;


    def clearMask(self):
        self._mask = None;


# the index of time dimension is 0
class VariationalDropoutLayer(DropoutLayer):
    def __init__(self, dropoutRatio = 0.5):
        super().__init__(dropoutRatio);

        self._name = f"VariationalDropout {dropoutRatio}";


    def _getMask(self, shape : Tuple[int, ...], dtype) -> np.ndarray:
        if len(shape) <= 2:
            return super()._getMask(shape, dtype);

        mask = super()._getMask(shape[1: ], dtype);
        mask = np.repeat(np.expand_dims(mask, axis = 0), shape[0], axis = 0);

        return mask;


# the dimension of feature map is self._ndim
class DropoutNDLayer(DropoutLayer):
    def __init__(self, dimensionNum : int, dropoutRatio = 0.5):
        super().__init__(dropoutRatio = dropoutRatio, reuseMask = False);

        self._ndim = max(1, int(dimensionNum));
        self._name = f"Dropout{self._ndim}D {self._dropoutRatio}";
    

    def _checkInput(self, X: np.ndarray) -> bool:
        return X.ndim > self._ndim;
    

    def _getMask(self, shape: Tuple[int, ...], dtype) -> np.ndarray:
        return super()._getMask(shape[: -self._ndim] + (1, ) * self._ndim, dtype);


class Dropout1DLayer(DropoutNDLayer):
    def __init__(self, dropoutRatio = 0.5):
        super().__init__(1, dropoutRatio = dropoutRatio);


class Dropout2DLayer(DropoutNDLayer):
    def __init__(self, dropoutRatio = 0.5):
        super().__init__(2, dropoutRatio = dropoutRatio);


class Dropout3DLayer(DropoutNDLayer):
    def __init__(self, dropoutRatio = 0.5):
        super().__init__(3, dropoutRatio = dropoutRatio);


class ReshapeLayer(NetModuleBase):
    def __init__(self, *shapeSelector : Union[Tuple, Callable[[np.ndarray], Tuple]]):
        super().__init__();

        self._originalShapes = [];
        self._shapeSelector = shapeSelector;
        self._name = "Reshape";


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        self._originalShapes.clear();
        output : List[np.ndarray] = [];

        for X, selector in zip(data, self._shapeSelector):
            self._originalShapes.append(X.shape);
            output.append(X.reshape(selector if isinstance(selector, tuple) else selector(X)));

        return tuple(output);


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dX : List[np.ndarray] = [];

        for dY, shape in zip(dout, self._originalShapes):
            dX.append(dY.reshape(shape));

        return tuple(dX);


class FlattenLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._shapes = [];
        self._name = "Flatten";


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        self._shapes.clear();
        output : List[np.ndarray] = [];

        for X in data:
            self._shapes.append(X.shape);

            if X.ndim > 1:
                output.append(np.reshape(X, (len(X), -1)));
            else:
                output.append(X);

        return tuple(output);


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dX : List[np.ndarray] = [];

        for dY, shape in zip(dout, self._shapes):
            dX.append(np.reshape(dY, shape));

        return tuple(dX);


class AffineLayer(NetModuleBase):
    def __init__(self, inputSize : int, outputSize : int, includeBias : bool = True, W : Optional[np.ndarray] = None, b : Optional[np.ndarray] = None, weightHandler : Optional[INetParamHandler] = None):
        super().__init__();

        self._X = np.empty(0);
        self._shape = None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._includeBias = includeBias;
        self._name = f"Affine {inputSize}*{outputSize}";

        self._weight = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, outputSize).astype(defaultDType) if W is None else W;
        self._bias = (np.zeros(outputSize, dtype = defaultDType) if b is None else b) if includeBias else None;

        self._params.append(NetParamDefinition("weight", self._weight, weightHandler));
        if self._bias is not None:
            self._params.append(NetParamDefinition("bias", self._bias, canDecay = False));


    def _setParams(self, params: List[INetParamDefinition]):
        self._weight, self._bias = params[0].value, params[1].value if self._includeBias else None;


    @property
    def weight(self) -> np.ndarray:
        return self._weight;


    @property
    def bias(self) -> Optional[np.ndarray]:
        return self._bias;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
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


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        if self._shape is not None:
            dY = np.reshape(dY, (-1, dY.shape[-1]));

        dW = self._X.T @ dY;
        self._params[0].grad[...] += dW;

        if self._bias is not None:
            db = np.sum(dY, 0);
            self._params[1].grad[...] += db;

        dX = dY @ self._weight.T;
        if self._shape is not None:
            dX = np.reshape(dX, self._shape);

        return dX, ;


class BatchNormalizationLayer(NetModuleBase):
    def __init__(self, inputSize : int, gamma : Optional[np.ndarray] = None, beta : Optional[np.ndarray] = None, epsilon = 1e-8, momentum : Optional[float] = 0.1):
        super().__init__();

        self._epsilon = epsilon;
        self._momentum = max(0.0, min(1.0, float(momentum))) if momentum is not None else None;
        self._evalWeight = None;
        self._evalBias = None;
        self._name = f"BatchNormalization {inputSize}";

        self._n = 0;
        self._XC = np.empty(0);
        self._std = np.empty(0);
        self._XHat = np.empty(0);

        self._gamma = np.ones(inputSize, dtype = defaultDType) if gamma is None else gamma;
        self._beta = np.zeros(inputSize, dtype = defaultDType) if beta is None else beta;

        self._params.append(NetParamDefinition("weight", self._gamma));
        self._params.append(NetParamDefinition("bias", self._beta, canDecay = False));

        self._evalMean = np.zeros_like(self._gamma);
        self._evalVar = np.ones_like(self._gamma);

        self._states.append(NetValueState(self._evalMean));
        self._states.append(NetValueState(self._evalVar));


    def _setParams(self, params: List[INetParamDefinition]):
        self._gamma, self._beta = params[0].value, params[1].value;


    def _setStates(self, states: List[INetState]):
        self._evalMean, self._evalVar = states[0].value, states[1].value;
        self._evalWeight, self._evalBias = None, None;


    # def _reset(self):
    #     self._evalWeight, self._evalBias = None, None;


    def _clean(self):
        self._evalMean[...] = 0.0;
        self._evalVar[...] = 1.0;
        self._evalWeight, self._evalBias = None, None;


    @property
    def weight(self) -> np.ndarray:
        return self._gamma;


    @property
    def bias(self) -> np.ndarray:
        return self._beta;


    @property
    def evalMean(self) -> np.ndarray:
        return self._evalMean;


    @property
    def evalVar(self) -> np.ndarray:
        return self._evalVar;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];

        shape = None;
        if X.ndim > 2:
            shape = X.shape;
            X = X.reshape(-1, shape[-1]);

        if self.context.isTrainingMode:
            mu = X.mean(axis = 0);
            self._n = len(X);
            self._XC = X - mu;
            var = np.square(self._XC).mean(axis = 0);
            s2 = self._n / (self._n - 1) * var if self._n > 1 else var;

            if self._momentum is not None:
                self._evalMean[...] = (1.0 - self._momentum) * self._evalMean + self._momentum * mu;
                self._evalVar[...] = (1.0 - self._momentum) * self._evalVar + self._momentum * s2;
            else:
                self._evalMean[...] = (self._evalMean * (self.context.trainingIterations - 1) + mu) / self.context.trainingIterations;
                self._evalVar[...] = (self._evalVar * (self.context.trainingIterations - 1) + s2) / self.context.trainingIterations;
            self._evalWeight, self._evalBias = None, None;

            self._std = np.sqrt(var + self._epsilon);
            self._XHat = self._XC / self._std;
            Y = self._gamma * self._XHat + self._beta;
        else:
            if self._evalWeight is None or self._evalBias is None:
                evalStd = np.sqrt(self._evalVar + self._epsilon);
                self._evalWeight = self._gamma / evalStd;
                self._evalBias = self._beta  - self._gamma * self._evalMean / evalStd;

            Y = self._evalWeight * X + self._evalBias;

        if shape is not None:
            Y = Y.reshape(*shape);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        shape = None;
        if dY.ndim > 2:
            shape = dY.shape;
            dY = dY.reshape(-1, shape[-1]);

        dXHat = dY * self._gamma;
        dXC = dXHat / self._std - np.sum(dXHat * self._XC, axis = 0) / np.power(self._std, 3) * self._XC / self._n;

        dGamma = np.sum(dY * self._XHat, axis = 0);
        dBeta = np.sum(dY, axis = 0);
        dX = dXC - np.sum(dXC, axis = 0) / self._n;

        self._params[0].grad[...] += dGamma;
        self._params[1].grad[...] += dBeta;

        if shape is not None:
            dX = dX.reshape(*shape);

        return dX, ;


class BatchNormalization1DLayer(BatchNormalizationLayer):
    def __init__(self, inputSize : int, gamma : Optional[np.ndarray] = None, beta : Optional[np.ndarray] = None, epsilon = 1e-8, momentum : Optional[float] = 0.1):
        super().__init__(inputSize, gamma = gamma, beta = beta, epsilon = epsilon, momentum = momentum);

        self._name = f"BatchNormalization1D {inputSize}";


    # X shape: (batch_size, input_size) or (batch_size, channel_num, sequence_length)
    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        
        if X.ndim > 2:
            Y, = super().forward(X.swapaxes(-2, -1));
            Y = Y.swapaxes(-2, -1);
        else:
            Y, = super().forward(X);
        
        return Y, ;
    

    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        if dY.ndim > 2:
            dX, = super().backward(dY.swapaxes(-2, -1));
            dX = dX.swapaxes(-2, -1);
        else:
            dX, = super().backward(dY);
        
        return dX, ;


class BatchNormalization2DLayer(BatchNormalizationLayer):
    def __init__(self, channelNum : int, gamma : Optional[np.ndarray] = None, beta : Optional[np.ndarray] = None, epsilon = 1e-8, momentum : Optional[float] = 0.1):
        super().__init__(channelNum, gamma = gamma, beta = beta, epsilon = epsilon, momentum = momentum);

        self._name = f"BatchNormalization2D {channelNum}";


    # X shape: (batch_size, channel_num, image_height, image_width)
    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];

        if X.ndim < 4:
            raise ValueError("the input shape of BatchNorm2D should be 4 at least");
        
        Y, = super().forward(X.swapaxes(-3, -1));
        Y = Y.swapaxes(-3, -1);
        
        return Y, ;
    

    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dX, = super().backward(dY.swapaxes(-3, -1));
        dX = dX.swapaxes(-3, -1);
        
        return dX, ;


class LayerNormalizationLayer(NetModuleBase):
    def __init__(self, layerShape : Union[int, Tuple[int, ...]], gamma : Optional[np.ndarray] = None, beta : Optional[np.ndarray] = None, epsilon = 1e-8):
        super().__init__();

        self._epsilon = epsilon;
        if isinstance(layerShape, int):
            layerShape = (layerShape, );
        self._name = "LayerNormalization";

        self._XC = np.empty(0);
        self._std = np.empty(0);
        self._XHat = np.empty(0);
        self._layerNdim = len(layerShape);
        self._layerAxis = tuple(-i for i in range(1, self._layerNdim + 1));
        self._layerSize = functools.reduce(lambda i, j: i * j, layerShape, 1);

        self._gamma = np.ones(layerShape, dtype = defaultDType) if gamma is None else gamma;
        self._beta = np.zeros(layerShape, dtype = defaultDType) if beta is None else beta;

        self._params.append(NetParamDefinition("weight", self._gamma));
        self._params.append(NetParamDefinition("bias", self._beta, canDecay = False));


    def _setParams(self, params: List[INetParamDefinition]):
        self._gamma, self._beta = params[0].value, params[1].value;


    @property
    def weight(self) -> np.ndarray:
        return self._gamma;


    @property
    def bias(self) -> np.ndarray:
        return self._beta;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];

        mu = X.mean(axis = self._layerAxis, keepdims = True);
        self._XC = X - mu;
        var = np.square(self._XC).mean(axis = self._layerAxis, keepdims = True);

        self._std = np.sqrt(var + self._epsilon);
        self._XHat = self._XC / self._std;
        Y = self._gamma * self._XHat + self._beta;

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        batchShape = tuple(range(len(dY.shape) - self._layerNdim));

        dXHat = dY * self._gamma;
        dXC = dXHat / self._std - np.sum(dXHat * self._XC, axis = self._layerAxis, keepdims = True) / np.power(self._std, 3) * self._XC / self._layerSize;

        dGamma = np.sum(dY * self._XHat, axis = batchShape);
        dBeta = np.sum(dY, axis = batchShape);
        dX = dXC - np.sum(dXC, axis = self._layerAxis, keepdims = True) / self._layerSize;

        self._params[0].grad[...] += dGamma;
        self._params[1].grad[...] += dBeta;

        return dX, ;


class MinMaxLayer(NetModuleBase):
    def __init__(self, minValue : Optional[float] = None, maxValue : Optional[float] = None):
        super().__init__();

        self._minValue = minValue;
        self._maxValue = maxValue;
        self._M = [];
        self._name = f"MinMax({minValue if minValue is not None else '-∞'} ≤ x ≤ {maxValue if maxValue is not None else '+∞'})";


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        if self._minValue is None and self._maxValue is None:
            return data;

        self._M, Y = [], [];
        for X in data:
            L = X < self._minValue if self._minValue is not None else np.array(False);
            H = X > self._maxValue if self._maxValue is not None else np.array(False);
            M = ~L * ~H;

            self._M.append(M);
            Y.append(((L * self._minValue).astype(X.dtype) if self._minValue is not None else 0) +
                     M * X +
                     ((H * self._maxValue).astype(X.dtype) if self._maxValue is not None else 0));

        return tuple(Y);


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        if self._minValue is None and self._maxValue is None:
            return dout;

        return tuple([M * dL for M, dL in zip(self._M, dout)]);


class Convolution1DLayer(NetModuleBase):
    def __init__(self, inputChannel : int, outputChannel : int, kernelSize : int, stride : int = 1, padding : Union[Tuple[int, int], int, str] = 0, dilation : int = 1, W : Optional[np.ndarray] = None, b : Optional[np.ndarray] = None):
        super().__init__();

        self._stride = stride;
        self._lastLength = 0;
        self._isSame = str(padding).lower() == "same";
        self._isCausal = str(padding).lower() == "causal";
        self._padding = (padding, padding) if isinstance(padding, int) else (padding if isinstance(padding, tuple) else 0);
        self._dilation = max(1, int(dilation));
        self._shape = tuple();
        self._colX = np.empty(0);
        self._colW = np.empty(0);
        self._name = f"Convolution1D {outputChannel}*{inputChannel}*{kernelSize}";

        self._weight = math.sqrt(2.0 / (inputChannel * kernelSize)) * np.random.randn(outputChannel, inputChannel, kernelSize).astype(defaultDType) if W is None else W;
        self._bias = np.zeros(outputChannel, dtype = defaultDType) if b is None else b;

        self._params.append(NetParamDefinition("weight", self._weight));
        self._params.append(NetParamDefinition("bias", self._bias, canDecay = False));


    def _setParams(self, params: List[INetParamDefinition]):
        self._weight, self._bias = params[0].value, params[1].value;


    @property
    def weight(self) -> np.ndarray:
        return self._weight;


    @property
    def bias(self) -> np.ndarray:
        return self._bias;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._shape = X.shape;

        N, C, T = X.shape;
        FN, C, FW = self._weight.shape;

        if self._isSame or self._isCausal:
            if T != self._lastLength:
                self._padding = getConvSamePadding(T, FW, self._stride, self._dilation) if self._isSame else getConvCausalPadding(T, FW, self._stride, self._dilation);
                self._lastLength = T;

        self._colX, OT = seq2col(X, FW, self._stride, self._padding, dilation = self._dilation);
        self._colW = self._weight.reshape(FN, -1).T;
        Y = self._colX @ self._colW + self._bias;
        Y = Y.reshape(N, OT, FN).transpose(0, 2, 1);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        FN, C, FW = self._weight.shape;

        colDY = dY.transpose(0, 2, 1).reshape(-1, FN);
        dW = self._colX.T @ colDY;
        db = np.sum(colDY, axis = 0);
        dX = colDY @ self._colW.T;
        dX = col2seq(dX, self._shape, FW, self._stride, self._padding, dilation = self._dilation, inDiff = True);

        self._params[0].grad[...] += dW.T.reshape(FN, C, FW);
        self._params[1].grad[...] += db;

        return dX, ;


class MaxPooling1DLayer(NetModuleBase):
    def __init__(self, poolingSize : int, stride : Optional[int] = None, padding : Union[Tuple[int, int], int] = 0):
        super().__init__();

        self._PW = poolingSize;
        self._stride = stride if stride is not None else self._PW;
        self._padding = padding;
        self._shape = tuple();
        self._M = None;
        self._name = f"MaxPooling1D {self._PW}";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._shape = X.shape;

        N, C, T = X.shape;

        col, OT = seq2col(X, self._PW, self._stride, self._padding);
        col = col.reshape(-1, self._PW);
        Y = np.amax(col, axis = -1).reshape(N, OT, C).transpose(0, 2, 1);

        if self.context.isTrainingMode:
            E = np.zeros_like(col, dtype = np.int32) + np.arange(col.shape[-1], dtype = np.int32);
            M = E == np.argmax(col, axis = -1, keepdims = True);
            self._M = (M + 0).astype(defaultDType);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        N, C, OT = dY.shape;

        colDY = dY.transpose(0, 2, 1).reshape(-1, 1);
        colDY = colDY * self._M;
        colDY = colDY.reshape(-1, C * self._PW);
        dX = col2seq(colDY, self._shape, self._PW, self._stride, self._padding, inDiff = True);

        return dX, ;


class AvgPooling1DLayer(NetModuleBase):
    def __init__(self, poolingSize : int, stride : Optional[int] = None, padding : Union[Tuple[int, int], int] = 0):
        super().__init__();

        self._PW = poolingSize;
        self._stride = stride if stride is not None else self._PW;
        self._padding = padding;
        self._shape = tuple();
        self._M = 1.0 / self._PW * np.ones(self._PW, dtype = defaultDType);
        self._name = f"AvgPooling1D {self._PW}";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._shape = X.shape;

        N, C, T = X.shape;

        col, OT = seq2col(X, self._PW, self._stride, self._padding);
        col = col.reshape(-1, self._PW);
        Y = np.mean(col, axis = -1).reshape(N, OT, C).transpose(0, 2, 1);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        N, C, OT = dY.shape;

        colDY = dY.transpose(0, 2, 1).reshape(-1, 1);
        colDY = colDY * self._M;
        colDY = colDY.reshape(-1, C * self._PW);
        dX = col2seq(colDY, self._shape, self._PW, self._stride, self._padding, inDiff =True);

        return dX, ;


class TcnLayer(Convolution1DLayer):
    def __init__(self, inputChannel : int, outputChannel : int, kernelSize : int, layerIndex : int = 0, dilation : Optional[int] = None, W : Optional[np.ndarray] = None, b : Optional[np.ndarray] = None):
        realDilation = dilation if dilation is not None else 2 ** layerIndex;
        padding = (realDilation * (kernelSize - 1), 0);

        super().__init__(inputChannel, outputChannel, kernelSize, stride = 1, padding = padding, dilation = realDilation, W = W, b = b);

        self._name = f"TCN {outputChannel}*{inputChannel}*{kernelSize}*{realDilation}";


class Convolution2DLayer(NetModuleBase):
    def __init__(self, inputChannel : int, outputChannel : int, kernelSize : Union[Tuple[int, int], int], stride : Union[Tuple[int, int], int] = 1, padding : Union[Tuple[int, ...], int, str] = 0, dilation : int = 1, W : Optional[np.ndarray] = None, b : Optional[np.ndarray] = None):
        super().__init__();

        FH, FW = kernelSize[: 2] if isinstance(kernelSize, tuple) else (kernelSize, kernelSize);
        self._stride = stride;
        self._strideHeight, self._strideWidth = parseStride2D(stride);
        self._lastHeight, self._lastWidth = 0, 0;
        self._isSame = str(padding).lower() == "same";
        self._isCausal = str(padding).lower() == "causal";
        self._padding = (padding, padding, padding, padding) if isinstance(padding, int) else (padding if isinstance(padding, tuple) else 0);
        self._dilation = max(1, int(dilation));
        self._shape = tuple();
        self._colX = np.empty(0);
        self._colW = np.empty(0);
        self._name = f"Convolution2D {outputChannel}*{inputChannel}*{FH}*{FW}";

        self._weight = math.sqrt(2.0 / (inputChannel * FH * FW)) * np.random.randn(outputChannel, inputChannel, FH, FW).astype(defaultDType) if W is None else W;
        self._bias = np.zeros(outputChannel, dtype = defaultDType) if b is None else b;

        self._params.append(NetParamDefinition("weight", self._weight));
        self._params.append(NetParamDefinition("bias", self._bias, canDecay = False));


    def _setParams(self, params: List[INetParamDefinition]):
        self._weight, self._bias = params[0].value, params[1].value;


    @property
    def weight(self) -> np.ndarray:
        return self._weight;


    @property
    def bias(self) -> np.ndarray:
        return self._bias;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._shape = X.shape;

        N, C, H, W = X.shape;
        FN, C, FH, FW = self._weight.shape;

        if self._isSame or self._isCausal:
            if H != self._lastHeight or W != self._lastWidth:
                if self._isSame:
                    self._padding = getConvSamePadding(H, FH, self._strideHeight, self._dilation) + getConvSamePadding(W, FW, self._strideWidth, self._dilation);
                else:
                    self._padding = getConvCausalPadding(H, FH, self._strideHeight, self._dilation) + getConvCausalPadding(W, FW, self._strideWidth, self._dilation);

                self._lastHeight, self._lastWidth = H, W;

        self._colX, OH, OW = im2col(X, FH, FW, self._stride, self._padding, dilation = self._dilation);
        self._colW = self._weight.reshape(FN, -1).T;
        Y = self._colX @ self._colW + self._bias;
        Y = Y.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        FN, C, FH, FW = self._weight.shape;

        colDY = dY.transpose(0, 2, 3, 1).reshape(-1, FN);
        dW = self._colX.T @ colDY;
        db = np.sum(colDY, axis = 0);
        dX = colDY @ self._colW.T;
        dX = col2im(dX, self._shape, FH, FW, self._stride, self._padding, dilation = self._dilation, inDiff = True);

        self._params[0].grad[...] += dW.T.reshape(FN, C, FH, FW);
        self._params[1].grad[...] += db;

        return dX, ;


class MaxPooling2DLayer(NetModuleBase):
    def __init__(self, poolingSize : Union[Tuple[int, int], int], stride : Optional[Union[Tuple[int, int], int]] = None, padding : Union[Tuple[int, ...], int] = 0):
        super().__init__();

        self._PH, self._PW = poolingSize[: 2] if isinstance(poolingSize, tuple) else (poolingSize, poolingSize);
        self._stride = stride if stride is not None else (self._PH, self._PW);
        self._padding = padding;
        self._shape = tuple();
        self._M = None;
        self._name = f"MaxPooling2D {self._PH}*{self._PW}";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._shape = X.shape;

        N, C, H, W = X.shape;

        col, OH, OW = im2col(X, self._PH, self._PW, self._stride, self._padding);
        col = col.reshape(-1, self._PH * self._PW);
        Y = np.amax(col, axis = -1).reshape(N, OH, OW, C).transpose(0, 3, 1, 2);

        if self.context.isTrainingMode:
            E = np.zeros_like(col, dtype = np.int32) + np.arange(col.shape[-1], dtype = np.int32);
            M = E == np.argmax(col, axis = -1, keepdims = True);
            self._M = (M + 0).astype(defaultDType);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        N, C, OH, OW = dY.shape;

        colDY = dY.transpose(0, 2, 3, 1).reshape(-1, 1);
        colDY = colDY * self._M;
        colDY = colDY.reshape(-1, C * self._PH * self._PW);
        dX = col2im(colDY, self._shape, self._PH, self._PW, self._stride, self._padding, inDiff = True);

        return dX, ;


class AvgPooling2DLayer(NetModuleBase):
    def __init__(self, poolingSize : Union[Tuple[int, int], int], stride : Optional[Union[Tuple[int, int], int]] = None, padding : Union[Tuple[int, ...], int] = 0):
        super().__init__();

        self._PH, self._PW = poolingSize[: 2] if isinstance(poolingSize, tuple) else (poolingSize, poolingSize);
        self._stride = stride if stride is not None else (self._PH, self._PW);
        self._padding = padding;
        self._shape = tuple();
        self._M = 1.0 / (self._PH * self._PW) * np.ones(self._PH * self._PW, dtype = defaultDType);
        self._name = f"AvgPooling2D {self._PH}*{self._PW}";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._shape = X.shape;

        N, C, H, W = X.shape;

        col, OH, OW = im2col(X, self._PH, self._PW, self._stride, self._padding);
        col = col.reshape(-1, self._PH * self._PW);
        Y = np.mean(col, axis = -1).reshape(N, OH, OW, C).transpose(0, 3, 1, 2);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        N, C, OH, OW = dY.shape;

        colDY = dY.transpose(0, 2, 3, 1).reshape(-1, 1);
        colDY = colDY * self._M;
        colDY = colDY.reshape(-1, C * self._PH * self._PW);
        dX = col2im(colDY, self._shape, self._PH, self._PW, self._stride, self._padding, inDiff =True);

        return dX, ;


class AdditiveResidualBlock(AggregateNetModule):
    def __init__(self, mainModule : INetModule, adaptiveModule : Optional[INetModule] = None, activationModule : Optional[INetModule] = None):
        self._mainModule = mainModule;
        self._adaptiveModule = adaptiveModule;
        self._activationModule = activationModule;

        modules : List[INetModule] = [self._mainModule];
        if self._adaptiveModule is not None:
            modules.append(self._adaptiveModule);
        if self._activationModule is not None:
            modules.append(self._activationModule);

        super().__init__(*tuple(modules));

        if self._activationModule is not None:
            self._name = f"AdditiveResidualBlock(({str(self._adaptiveModule) if self._adaptiveModule is not None else 'X'} + ({self._mainModule})) --> {self._activationModule})";
        else:
            self._name = f"AdditiveResidualBlock({str(self._adaptiveModule) if self._adaptiveModule is not None else 'X'} + ({self._mainModule}))";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data;

        Y = self._mainModule.forward(*X);
        if self._adaptiveModule is not None:
            X = self._adaptiveModule.forward(*X);
        Z = tuple([x + y for x, y in zip(X, Y)]);

        if self._activationModule is not None:
            return self._activationModule.forward(*Z);
        else:
            return Z;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dZ = self._activationModule.backward(*dout) if self._activationModule is not None else dout;

        dX1 = self._mainModule.backward(*dZ);
        if self._adaptiveModule is not None:
            dX2 = self._adaptiveModule.backward(*dZ);
        else:
            dX2 = dZ;

        return tuple([dx1 + dx2 for dx1, dx2 in zip(dX1, dX2)]);


class EmbeddingLayer(NetModuleBase):
    def __init__(self, embeddingNum : int, embeddingSize : int, W : Optional[np.ndarray] = None):
        super().__init__();

        self._index = None;
        self._shape = tuple();
        self._embeddingNum = embeddingNum;
        self._embeddingSize = embeddingSize;
        self._name = f"Embedding {embeddingNum}*{embeddingSize}";

        self._weight = math.sqrt(2.0 / embeddingSize) * np.random.randn(embeddingNum, embeddingSize).astype(defaultDType) if W is None else W;
        self._params.append(NetParamDefinition("weight", self._weight));


    def _setParams(self, params: List[INetParamDefinition]):
        self._weight = params[0].value;


    @property
    def weight(self) -> np.ndarray:
        return self._weight;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._shape = X.shape;
        self._index = X.flatten();
        Y = self._weight[self._index].reshape(X.shape + (self._embeddingSize,));

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dY = dY.reshape(-1, self._embeddingSize);

        dW = self._params[0].grad;
        npAddAt(dW, self._index, dY);

        # the dX is always zero!!!
        return np.zeros(self._shape, dtype = dY.dtype), ;


class EmbeddingWithDotLayer(NetModuleBase):
    def __init__(self, embeddingNum : int, embeddingSize : int, W : Optional[np.ndarray] = None):
        super().__init__();

        self._X = None;
        self._W = None;
        self._name = f"EmbeddingWithDot {embeddingNum}*{embeddingSize}";
        self._embeddingLayer = EmbeddingLayer(embeddingNum, embeddingSize, W = W);


    @property
    def params(self) -> List[INetParamDefinition]:
        return self._embeddingLayer.params;


    @params.setter
    def params(self, value: List[INetParamDefinition]):
        self._embeddingLayer.params = value;


    @property
    def weight(self) -> np.ndarray:
        return self._embeddingLayer.weight;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X, T = data;
        self._X = X;
        self._W = self._embeddingLayer.forward(T)[0];
        Y = np.sum(self._X * self._W, axis = -1);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dY = np.expand_dims(dY, axis = -1);

        dW = dY * self._X;
        dX = dY * self._W;
        self._embeddingLayer.backward(dW);

        return dX, ;


class RnnCellBase(NetModuleBase, metaclass = abc.ABCMeta):
    def __init__(self, inputSize : int, hiddenSize : int, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, bx : Optional[np.ndarray] = None, bh : Optional[np.ndarray] = None):
        super().__init__();

        self._inputSize = inputSize;
        self._hiddenSize = hiddenSize;

        self._weightX, self._weightH, self._biasX, self._biasH = self._initParams(inputSize, hiddenSize, Wx, Wh, bx, bh);
        self._params.append(NetParamDefinition("weightX", self._weightX));
        self._params.append(NetParamDefinition("weightH", self._weightH));
        self._params.append(NetParamDefinition("biasX", self._biasX, canDecay = False));
        self._params.append(NetParamDefinition("biasH", self._biasH, canDecay = False));


    def _setParams(self, params: List[INetParamDefinition]):
        self._weightX, self._weightH, self._biasX, self._biasH = params[0].value, params[1].value, params[2].value, params[3].value;


    @abc.abstractmethod
    def _initParams(self, inputSize: int, hiddenSize: int, Wx: Optional[np.ndarray], Wh: Optional[np.ndarray], bx: Optional[np.ndarray], bh: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass;


    @property
    def weightX(self) -> np.ndarray:
        return self._weightX;


    @property
    def weightH(self) -> np.ndarray:
        return self._weightH;


    @property
    def biasX(self) -> np.ndarray:
        return self._biasX;


    @property
    def biasH(self) -> np.ndarray:
        return self._biasH;


class RnnCell(RnnCellBase):
    def __init__(self, inputSize : int, hiddenSize : int, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, bx : Optional[np.ndarray] = None, bh : Optional[np.ndarray] = None, activationFunc : Optional[INetModule] = None):
        if activationFunc is not None and len(activationFunc.params) > 0:
            raise ValueError("not supports activation function with parameters");

        super().__init__(inputSize, hiddenSize, Wx, Wh, bx, bh);

        self._X, self._W = np.empty(0), np.empty(0);
        self._xi = [inputSize];
        self._activationFunc = activationFunc if activationFunc is not None else TanhLayer();
        self._name = f"RNN Cell {inputSize}*{hiddenSize}";


    def _initParams(self, inputSize : int, hiddenSize : int, Wx : Optional[np.ndarray], Wh : Optional[np.ndarray], bx : Optional[np.ndarray], bh : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weightX = math.sqrt(2.0 / (inputSize + hiddenSize)) * np.random.randn(inputSize, hiddenSize).astype(defaultDType) if Wx is None else Wx;
        weightH = math.sqrt(1.0 / hiddenSize) * np.random.randn(hiddenSize, hiddenSize).astype(defaultDType) if Wh is None else Wh;
        biasX = np.zeros(hiddenSize, dtype = defaultDType) if bx is None else bx;
        biasH = np.zeros(hiddenSize, dtype = defaultDType) if bh is None else bh;

        return weightX, weightH, biasX, biasH;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X, H = data;
        self._X = np.concatenate((X, H), axis = -1);
        self._W = np.concatenate((self._weightX, self._weightH), axis = 0);
        A, = self._activationFunc.forward(self._X @ self._W + self._biasX + self._biasH);

        return A, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY, = self._activationFunc.backward(*dout);
        dX = dY @ self._W.T;
        dW = self._X.T @ dY;
        db = np.sum(dY, axis = 0);

        dX, dH = tuple(np.split(dX, self._xi, axis = -1));
        dWx, dWh = tuple(np.split(dW, self._xi, axis = 0));

        self._params[0].grad[...] += dWx;
        self._params[1].grad[...] += dWh;
        self._params[2].grad[...] += db;
        self._params[3].grad[...] += db;

        return dX, dH;


class GruCell(RnnCellBase):
    def __init__(self, inputSize : int, hiddenSize : int, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, bx : Optional[np.ndarray] = None, bh : Optional[np.ndarray] = None):
        super().__init__(inputSize, hiddenSize, Wx, Wh, bx, bh);

        self._Xg, self._Xa = np.empty(0), np.empty(0);
        self._Wg, self._Wa = np.empty(0), np.empty(0);
        self._H, self._G, self._R, self._Z, self._A = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0);
        self._gi, self._ri, self._xi = [2 * hiddenSize], [hiddenSize], [inputSize];
        self._name = f"GRU Cell {inputSize}*{hiddenSize}";


    def _initParams(self, inputSize: int, hiddenSize: int, Wx: Optional[np.ndarray], Wh: Optional[np.ndarray], bx: Optional[np.ndarray], bh: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weightX = math.sqrt(2.0 / (inputSize + hiddenSize)) * np.random.randn(inputSize, 3 * hiddenSize).astype(defaultDType) if Wx is None else Wx;
        weightH = math.sqrt(1.0 / hiddenSize) * np.random.randn(hiddenSize, 3 * hiddenSize).astype(defaultDType) if Wh is None else Wh;
        biasX = np.zeros(3 * hiddenSize, dtype = defaultDType) if bx is None else bx;
        biasH = np.zeros(3 * hiddenSize, dtype = defaultDType) if bh is None else bh;

        return weightX, weightH, biasX, biasH;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X, self._H = data;

        W = np.concatenate((self._weightX, self._weightH), axis = 0);
        b = self._biasX + self._biasH;

        self._Wg, self._Wa = tuple(np.split(W, self._gi, axis = -1));
        bg, ba = tuple(np.split(b, self._gi, axis = -1));

        self._Xg = np.concatenate((X, self._H), axis = -1);
        self._G = sigmoid(self._Xg @ self._Wg + bg);
        self._R, self._Z = tuple(np.split(self._G, self._ri, axis = -1));

        self._Xa = np.concatenate((X, self._R * self._H), axis = -1);
        self._A = tanh(self._Xa @ self._Wa + ba);

        Y = self._Z * (self._H - self._A) + self._A;

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dZ = dY * (self._H - self._A);
        dH = dY * self._Z;
        dA = dY * (1 - self._Z) * tanhGradient(self._A);

        dXa = dA @ self._Wa.T;
        dWa = self._Xa.T @ dA;
        dba = np.sum(dA, axis = 0);

        dX, dRH = tuple(np.split(dXa, self._xi, axis = -1));
        dR = dRH * self._H;
        dH += dRH * self._R;

        dG = np.concatenate((dR, dZ), axis = -1) * sigmoidGradient(self._G);
        dXg = dG @ self._Wg.T;
        dWg = self._Xg.T @ dG;
        dbg = np.sum(dG, axis = 0);

        dXgx, dXgh = tuple(np.split(dXg, self._xi, axis = -1));
        dX += dXgx;
        dH += dXgh;

        dW = np.concatenate((dWg, dWa), axis = -1);
        db = np.concatenate((dbg, dba), axis = -1);
        dWx, dWh = tuple(np.split(dW, self._xi, axis = 0));

        self._params[0].grad[...] += dWx;
        self._params[1].grad[...] += dWh;
        self._params[2].grad[...] += db;
        self._params[3].grad[...] += db;

        return dX, dH;


class LstmCell(RnnCellBase):
    def __init__(self, inputSize : int, hiddenSize : int, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, bx : Optional[np.ndarray] = None, bh : Optional[np.ndarray] = None):
        super().__init__(inputSize, hiddenSize, Wx, Wh, bx, bh);

        self._X, self._W, self._C = np.empty(0), np.empty(0), np.empty(0);
        self._F, self._I, self._O, self._G, self._S, self._tanhYC = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0);
        self._xi, self._si, self._gi = [self._inputSize], [3 * self._hiddenSize], [self._hiddenSize, 2 * self._hiddenSize];
        self._name = f"LSTM Cell {inputSize}*{hiddenSize}";


    def _initParams(self, inputSize : int, hiddenSize : int, Wx : Optional[np.ndarray], Wh : Optional[np.ndarray], bx : Optional[np.ndarray], bh : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weightX = math.sqrt(2.0 / (inputSize + hiddenSize)) * np.random.randn(inputSize, 4 * hiddenSize).astype(defaultDType) if Wx is None else Wx;
        weightH = math.sqrt(1.0 / hiddenSize) * np.random.randn(hiddenSize, 4 * hiddenSize).astype(defaultDType) if Wh is None else Wh;
        biasX = np.zeros(4 * hiddenSize, dtype = defaultDType) if bx is None else bx;
        biasH = np.zeros(4 * hiddenSize, dtype = defaultDType) if bh is None else bh;

        return weightX, weightH, biasX, biasH;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X, H, self._C  = data;

        self._X = np.concatenate((X, H), axis = -1);
        self._W = np.concatenate((self._weightX, self._weightH), axis = 0);
        b = self._biasX + self._biasH;

        G, S = np.split(self._X @ self._W + b, self._si, axis = -1);
        self._G, self._S = sigmoid(G), tanh(S);
        self._F, self._I, self._O = np.split(self._G, self._gi, axis = -1);

        YC = self._F * self._C + self._I * self._S;
        self._tanhYC = tanh(YC);
        YH = self._O * self._tanhYC;

        return YH, YC;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dYH, dYC = dout;

        dO = dYH * self._tanhYC;
        dYC = dYC + dYH * self._O * tanhGradient(self._tanhYC);

        dF = dYC * self._C;
        dC = dYC * self._F;
        dI = dYC * self._S;
        dS = dYC * self._I * tanhGradient(self._S);
        dG = np.concatenate((dF, dI, dO), axis = -1);
        dG *= sigmoidGradient(self._G);
        dA = np.concatenate((dG, dS), axis = -1);

        dX = dA @ self._W.T;
        dW = self._X.T @ dA;
        db = np.sum(dA, axis = 0);

        dX, dH = np.split(dX, self._xi, axis = -1);
        dWx, dWh = np.split(dW, self._xi, axis = 0);
        self._params[0].grad[...] += dWx;
        self._params[1].grad[...] += dWh;
        self._params[2].grad[...] += db;
        self._params[3].grad[...] += db;

        return dX, dH, dC;


class RnnLayerBase(NetModuleBase, metaclass = abc.ABCMeta):
    def __init__(self, inputSize : int, hiddenSize : int, stateful : bool = True, returnSequence : bool = True, returnState : bool = False, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, bx : Optional[np.ndarray] = None, bh : Optional[np.ndarray] = None):
        super().__init__();

        if not returnSequence and not returnState:
            raise ValueError("returnSequence and returnState are both false");

        self._inputSize = inputSize;
        self._hiddenSize = hiddenSize;
        self._stateful = stateful;
        self._returnSequence = returnSequence;
        self._returnState = returnState;

        self._weightX, self._weightH, self._biasX, self._biasH = self._initParams(inputSize, hiddenSize, Wx, Wh, bx, bh);
        self._params.append(NetParamDefinition("weightX", self._weightX));
        self._params.append(NetParamDefinition("weightH", self._weightH));
        self._params.append(NetParamDefinition("biasX", self._biasX, canDecay = False));
        self._params.append(NetParamDefinition("biasH", self._biasH, canDecay = False));


    def _setParams(self, params: List[INetParamDefinition]):
        self._weightX, self._weightH, self._biasX, self._biasH = params[0].value, params[1].value, params[2].value, params[3].value;


    @abc.abstractmethod
    def _initParams(self, inputSize : int, hiddenSize : int, Wx : Optional[np.ndarray], Wh : Optional[np.ndarray], bx : Optional[np.ndarray], bh : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass;


    @property
    def weightX(self) -> np.ndarray:
        return self._weightX;


    @property
    def weightH(self) -> np.ndarray:
        return self._weightH;


    @property
    def biasX(self) -> np.ndarray:
        return self._biasX;


    @property
    def biasH(self) -> np.ndarray:
        return self._biasH;


class RnnLayer(RnnLayerBase):
    def __init__(self, inputSize : int, hiddenSize : int, stateful : bool = True, returnSequence : bool = True, returnState : bool = False, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, bx : Optional[np.ndarray] = None, bh : Optional[np.ndarray] = None, activationFuncSelector : Optional[Callable[[int], INetModel]] = None):
        super().__init__(inputSize, hiddenSize, stateful, returnSequence, returnState, Wx, Wh, bx, bh);

        self._H = None;
        self._dH = None;
        self._sequenceLength = 0;
        self._foreignState = False;
        self._activationFuncSelector : Callable[[int], INetModel] = activationFuncSelector if activationFuncSelector is not None else (lambda size: TanhLayer()); # type: ignore
        self._name = f"RNN {inputSize}*{hiddenSize}";

        self._Xs, self._W = np.empty(0), np.empty(0);
        self._activationFuncs : List[INetModule] = [];


    def _setContext(self, context : INetContext):
        for af in self._activationFuncs:
            af.context = context;


    def _initParams(self, inputSize : int, hiddenSize : int, Wx : Optional[np.ndarray], Wh : Optional[np.ndarray], bx : Optional[np.ndarray], bh : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weightX = math.sqrt(2.0 / (inputSize + hiddenSize)) * np.random.randn(inputSize, hiddenSize).astype(defaultDType) if Wx is None else Wx;
        weightH = math.sqrt(1.0 / hiddenSize) * np.random.randn(hiddenSize, hiddenSize).astype(defaultDType) if Wh is None else Wh;
        biasX = np.zeros(hiddenSize, dtype = defaultDType) if bx is None else bx;
        biasH = np.zeros(hiddenSize, dtype = defaultDType) if bh is None else bh;

        return weightX, weightH, biasX, biasH;


    @property
    def dH(self) -> Optional[np.ndarray]:
        return self._dH;


    def _reset(self):
        self._H = None;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]: # type: ignore
        Xs = data[0];
        self._foreignState = len(data) > 1;
        self._sequenceLength, N = Xs.shape[: 2];

        if len(self._activationFuncs) != self._sequenceLength:
            self._activationFuncs = [self._activationFuncSelector(self._hiddenSize) for _ in range(self._sequenceLength)];

        if self._foreignState:
            self._H = data[1];
        else:
            if not self._stateful:
                self._H = None;
        
        if self._H is None:
            self._H = np.zeros((N, self._hiddenSize), Xs.dtype);

        self._Xs, Ys = [], [];
        self._W = np.concatenate((self._weightX, self._weightH), axis = 0);
        for t in range(self._sequenceLength):
            X, af = Xs[t], self._activationFuncs[t];

            X = np.concatenate((X, self._H), axis = -1);
            self._Xs.append(X);

            self._H, = af.forward(X @ self._W + self._biasX + self._biasH);
            Ys.append(self._H);

        Ys = np.array(Ys);

        if self._returnSequence and self._returnState:
            return Ys, self._H;
        if self._returnSequence:
            return Ys, ;
        if self._returnState:
            return self._H, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        if self._returnSequence and self._returnState:
            dYs, dH = dout;
        elif self._returnSequence:
            dYs = dout[0];

            # truncated BPTT
            dH = np.zeros_like(self._H);
        else:
            dYs, dH = [0] * self._sequenceLength, dout[0];

        dXs = [];
        dW = np.zeros_like(self._W);
        db = np.zeros_like(self._biasH);
        WT = self._W.T;

        for t in reversed(range(self._sequenceLength)):
            af, X, dY = self._activationFuncs[t], self._Xs[t], dYs[t];

            dA, = af.backward(dY + dH);
            dW += X.T @ dA;
            db += np.sum(dA, axis = 0);
            dX, dH = tuple(np.split(dA @ WT, [self._inputSize], axis = -1));

            dXs.append(dX);

        dXs.reverse();
        dWx, dWh = tuple(np.split(dW, [self._inputSize], axis = 0));

        self._dH = dH;
        dXs = np.array(dXs);
        self._params[0].grad[...] += dWx;
        self._params[1].grad[...] += dWh;
        self._params[2].grad[...] += db;
        self._params[3].grad[...] += db;

        return (dXs, self._dH) if self._foreignState else (dXs, );


    def setState(self, H : np.ndarray):
        self._H = H;


class GruLayer(RnnLayerBase):
    def __init__(self, inputSize : int, hiddenSize : int, stateful : bool = True, returnSequence : bool = True, returnState : bool = False, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, bx : Optional[np.ndarray] = None, bh : Optional[np.ndarray] = None):
        super().__init__(inputSize, hiddenSize, stateful, returnSequence, returnState, Wx, Wh, bx, bh);

        self._H = None;
        self._dH = None;
        self._sequenceLength = 0;
        self._foreignState = False;
        self._name = f"GRU {inputSize}*{hiddenSize}";

        self._Xgs, self._Xas = [], [];
        self._Wg, self._Wa = np.empty(0), np.empty(0);
        self._Hs, self._Gs, self._Rs, self._Zs, self._As = [], [], [], [], [];
        self._gi, self._ri, self._xi = [2 * hiddenSize], [hiddenSize], [inputSize];


    def _initParams(self, inputSize : int, hiddenSize : int, Wx : Optional[np.ndarray], Wh : Optional[np.ndarray], bx : Optional[np.ndarray], bh : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weightX = math.sqrt(2.0 / (inputSize + hiddenSize)) * np.random.randn(inputSize, 3 * hiddenSize).astype(defaultDType) if Wx is None else Wx;
        weightH = math.sqrt(1.0 / hiddenSize) * np.random.randn(hiddenSize, 3 * hiddenSize).astype(defaultDType) if Wh is None else Wh;
        biasX = np.zeros(3 * hiddenSize, dtype = defaultDType) if bx is None else bx;
        biasH = np.zeros(3 * hiddenSize, dtype = defaultDType) if bh is None else bh;

        return weightX, weightH, biasX, biasH;


    @property
    def dH(self) -> Optional[np.ndarray]:
        return self._dH;


    def _reset(self):
        self._H = None;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]: # type: ignore
        Xs = data[0];
        self._foreignState = len(data) > 1;
        self._sequenceLength, N = Xs.shape[: 2];

        if self._foreignState:
            self._H = data[1];
        else:
            if not self._stateful:
                self._H = None;
        
        if self._H is None:
            self._H = np.zeros((N, self._hiddenSize), Xs.dtype);

        W = np.concatenate((self._weightX, self._weightH), axis = 0);
        b = self._biasX + self._biasH;

        self._Wg, self._Wa = tuple(np.split(W, self._gi, axis = -1));
        bg, ba = tuple(np.split(b, self._gi, axis = -1));

        Ys = [];
        self._Hs.clear();
        self._Xgs.clear();
        self._Gs.clear();
        self._Rs.clear();
        self._Zs.clear();
        self._Xas.clear();
        self._As.clear();

        for X in Xs:
            self._Hs.append(self._H);

            Xg = np.concatenate((X, self._H), axis = -1);

            G = sigmoid(Xg @ self._Wg + bg);
            R, Z = tuple(np.split(G, self._ri, axis = -1));

            Xa = np.concatenate((X, R * self._H), axis = -1);
            A = tanh(Xa @ self._Wa + ba);

            self._H = Z * (self._H - A) + A;

            self._Xgs.append(Xg);
            self._Gs.append(G);
            self._Rs.append(R);
            self._Zs.append(Z);
            self._Xas.append(Xa);
            self._As.append(A);
            Ys.append(self._H);

        Ys = np.array(Ys);

        if self._returnSequence and self._returnState:
            return Ys, self._H;
        if self._returnSequence:
            return Ys, ;
        if self._returnState:
            return self._H, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        if self._returnSequence and self._returnState:
            dYs, dH = dout;
        elif self._returnSequence:
            dYs = dout[0];

            # truncated BPTT
            dH = np.zeros_like(self._H);
        else:
            dYs, dH = [0] * self._sequenceLength, dout[0];

        dXs = [];
        dWg = np.zeros_like(self._Wg);
        dWa = np.zeros_like(self._Wa);
        dbg = np.zeros(2 * self._hiddenSize, dtype = self._H.dtype); # type: ignore
        dba = np.zeros(self._hiddenSize, dtype = self._H.dtype); # type: ignore
        WgT, WaT = self._Wg.T, self._Wa.T;

        for t in reversed(range(self._sequenceLength)):
            dY = dYs[t];
            H, Xg, G, R, Z, Xa, A = self._Hs[t], self._Xgs[t], self._Gs[t], self._Rs[t], self._Zs[t], self._Xas[t], self._As[t];

            dY = dY + dH;

            dZ = dY * (H - A);
            dH = dY * Z;
            dA = dY * (1 - Z) * tanhGradient(A);

            dXa = dA @ WaT;
            dWa += Xa.T @ dA;
            dba += np.sum(dA, axis = 0);

            dX, dRH = tuple(np.split(dXa, self._xi, axis = -1));
            dR = dRH * H;
            dH += dRH * R;

            dG = np.concatenate((dR, dZ), axis = -1) * sigmoidGradient(G);
            dXg = dG @ WgT;
            dWg += Xg.T @ dG;
            dbg += np.sum(dG, axis = 0);

            dXgx, dXgh = tuple(np.split(dXg, self._xi, axis = -1));
            dX += dXgx;
            dH += dXgh;

            dXs.append(dX);

        dXs.reverse();
        dW = np.concatenate((dWg, dWa), axis = -1);
        db = np.concatenate((dbg, dba), axis = -1);
        dWx, dWh = tuple(np.split(dW, self._xi, axis = 0));

        self._dH = dH;
        dXs = np.array(dXs);
        self._params[0].grad[...] += dWx;
        self._params[1].grad[...] += dWh;
        self._params[2].grad[...] += db;
        self._params[3].grad[...] += db;

        return (dXs, self._dH) if self._foreignState else (dXs, );


    def setState(self, H : np.ndarray):
        self._H = H;


class LstmLayer(RnnLayerBase):
    def __init__(self, inputSize : int, hiddenSize : int, stateful : bool = True, returnSequence : bool = True, returnState : bool = False, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, bx : Optional[np.ndarray] = None, bh : Optional[np.ndarray] = None):
        super().__init__(inputSize, hiddenSize, stateful, returnSequence, returnState, Wx, Wh, bx, bh);

        self._H, self._C = None, None;
        self._dH, self._dC = None, None;
        self._sequenceLength = 0;
        self._foreignState = False;
        self._name = f"LSTM {inputSize}*{hiddenSize}";

        self._Xs, self._W, self._Cs = [], np.empty(0), [];
        self._Fs, self._Is, self._Os, self._Gs, self._Ss, self._tanhYCs = [], [], [], [], [], [];
        self._xi, self._si, self._gi = [self._inputSize], [3 * self._hiddenSize], [self._hiddenSize, 2 * self._hiddenSize];


    def _initParams(self, inputSize : int, hiddenSize : int, Wx : Optional[np.ndarray], Wh : Optional[np.ndarray], bx : Optional[np.ndarray], bh : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weightX = math.sqrt(2.0 / (inputSize + hiddenSize)) * np.random.randn(inputSize, 4 * hiddenSize).astype(defaultDType) if Wx is None else Wx;
        weightH = math.sqrt(1.0 / hiddenSize) * np.random.randn(hiddenSize, 4 * hiddenSize).astype(defaultDType) if Wh is None else Wh;
        biasX = np.zeros(4 * hiddenSize, dtype = defaultDType) if bx is None else bx;
        biasH = np.zeros(4 * hiddenSize, dtype = defaultDType) if bh is None else bh;

        return weightX, weightH, biasX, biasH;


    @property
    def dH(self) -> Optional[np.ndarray]:
        return self._dH;


    @property
    def dC(self) -> Optional[np.ndarray]:
        return self._dC;


    def _reset(self):
        self._H, self._C = None, None;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]: # type: ignore
        Xs = data[0];
        self._foreignState = len(data) > 1;
        self._sequenceLength, N = Xs.shape[: 2];

        if self._foreignState:
            self._H, self._C = data[1], data[2];
        else:
            if not self._stateful:
                self._H, self._C = None, None;
        
        if self._H is None:
            self._H = np.zeros((N, self._hiddenSize), Xs.dtype);
        if self._C is None:
            self._C = np.zeros((N, self._hiddenSize), Xs.dtype);

        self._W = np.concatenate((self._weightX, self._weightH), axis = 0);
        b = self._biasX + self._biasH;

        YHs = [];
        self._Xs.clear();
        self._Cs.clear();
        self._Fs.clear();
        self._Is.clear();
        self._Os.clear();
        self._Gs.clear();
        self._Ss.clear();
        self._tanhYCs.clear();

        for X in Xs:
            self._Cs.append(self._C);

            X = np.concatenate((X, self._H), axis = -1);
            G, S = np.split(X @ self._W + b, self._si, axis = -1);
            G, S = sigmoid(G), tanh(S);
            F, I, O = np.split(G, self._gi, axis = -1);

            self._C = F * self._C + I * S;
            tanhYC = tanh(self._C);
            self._H = O * tanhYC;

            self._Xs.append(X);
            self._Fs.append(F);
            self._Is.append(I);
            self._Os.append(O);
            self._Gs.append(G);
            self._Ss.append(S);
            self._tanhYCs.append(tanhYC);
            YHs.append(self._H);

        YHs = np.array(YHs);

        if self._returnSequence and self._returnState:
            return YHs, self._H, self._C;
        if self._returnSequence:
            return YHs, ;
        if self._returnState:
            return self._H, self._C;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        if self._returnSequence and self._returnState:
            dYs, dH, dC = dout;
        elif self._returnSequence:
            dYs = dout[0];

            # truncated BPTT
            dH = np.zeros_like(self._H);
            dC = np.zeros_like(self._C);
        else:
            dYs, dH, dC = [0] * self._sequenceLength, dout[0], dout[1];

        dXs = [];
        dW = np.zeros_like(self._W);
        db = np.zeros_like(self._biasX);
        WT = self._W.T;

        for t in reversed(range(self._sequenceLength)):
            dY = dYs[t];
            C, X, F, I, O, G, S, tanhYC = self._Cs[t], self._Xs[t], self._Fs[t], self._Is[t], self._Os[t], self._Gs[t], self._Ss[t], self._tanhYCs[t];

            dYH = dY + dH;
            dYC = dC;

            dO = dYH * tanhYC;
            dYC = dYC + dYH * O * tanhGradient(tanhYC);

            dF = dYC * C;
            dC = dYC * F;
            dI = dYC * S;
            dS = dYC * I * tanhGradient(S);
            dG = np.concatenate((dF, dI, dO), axis = -1);
            dG *= sigmoidGradient(G);
            dA = np.concatenate((dG, dS), axis = -1);

            dX = dA @ WT;
            dW += X.T @ dA;
            db += np.sum(dA, axis = 0);

            dX, dH = np.split(dX, self._xi, axis = -1);
            dXs.append(dX);

        dXs.reverse();
        dWx, dWh = np.split(dW, self._xi, axis = 0);

        self._dH = dH;
        self._dC = dC;
        dXs = np.array(dXs);
        self._params[0].grad[...] += dWx;
        self._params[1].grad[...] += dWh;
        self._params[2].grad[...] += db;
        self._params[3].grad[...] += db;

        return (dXs, self._dH, self._dC) if self._foreignState else (dXs, );


    def setState(self, H : np.ndarray, C : np.ndarray):
        self._H, self._C = H, C;


# '''
# dropout mechanism: https://arxiv.org/abs/1603.05118 <Recurrent Dropout without Memory Loss>
# '''
# class LstmCell2(NetModuleBase):
#     def __init__(self, inputSize : int, outputSize : int, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, b : Optional[np.ndarray] = None, inputDropout : float = 0, recurrentDropout : float = 0):
#         super().__init__();

#         self._X, self._H, self._C = None, None, None;
#         self._F, self._G, self._I, self._O = None, None, None, None;
#         self._YC, self._tanhYC, self._YH = None, None, None;
#         self._inputSize = inputSize;
#         self._outputSize = outputSize;
#         self._name = f"LSTM Cell {inputSize}*{outputSize}";

#         self._weightX = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, 4 * outputSize).astype(defaultDType) if Wx is None else Wx;
#         self._weightH = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, 4 * outputSize).astype(defaultDType) if Wh is None else Wh;
#         self._bias = np.zeros(4 * outputSize, dtype = defaultDType) if b is None else b;
#         self._inputDropout = inputDropout;
#         self._recurrentDropout = recurrentDropout;
#         self._inputDropoutMask = None;
#         self._recurrentDropoutMask = None;

#         self._params.append(NetParamDefinition("weightX", self._weightX));
#         self._params.append(NetParamDefinition("weightH", self._weightH));
#         self._params.append(NetParamDefinition("bias", self._bias, canDecay = False));


#     def _setParams(self, value: List[np.ndarray]):
#         self._weightX, self._weightH, self._bias = value[0], value[1], value[2];


#     def _reset(self):
#         self.setInputDropoutMask();
#         self.setRecurrentDropoutMask();


#     @property
#     def weightX(self) -> np.ndarray:
#         return self._weightX;


#     @property
#     def weightH(self) -> np.ndarray:
#         return self._weightH;


#     @property
#     def bias(self) -> np.ndarray:
#         return self._bias;


#     def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
#         self._X, self._H, self._C = data;

#         if self._inputDropoutMask is None:
#             self._inputDropoutMask = getDropoutMask(self._H, self._inputDropout);
#         if self._recurrentDropoutMask is None:
#             self._recurrentDropoutMask = getDropoutMask(self._C, self._recurrentDropout);

#         if self.context.isTrainingMode:
#             A = self._X @ self._weightX + (self._inputDropoutMask * self._H) @ self._weightH + self._bias;
#         else:
#             A = self._X @ self._weightX + ((1 - self._inputDropout) * self._H) @ self._weightH + self._bias;
#         self._F, self._I, self._O, self._G = tuple(np.hsplit(A, 4));
#         self._F, self._I, self._O, self._G = sigmoid(self._F), sigmoid(self._I), sigmoid(self._O), tanh(self._G);

#         if self.context.isTrainingMode:
#             self._YC = self._C * self._F + self._recurrentDropoutMask * self._G * self._I;
#         else:
#             self._YC = self._C * self._F + (1 - self._recurrentDropout) * self._G * self._I;
#         self._tanhYC = tanh(self._YC);
#         self._YH = self._tanhYC * self._O;

#         return self._YH, self._YC;


#     def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
#         dYH, dYC = dout;

#         dYC += dYH * self._O * tanhGradient(self._tanhYC);
#         dF, dG, dI, dO = dYC * self._C, dYC * self._I * self._recurrentDropoutMask, dYC * self._G * self._recurrentDropoutMask, dYH * self._tanhYC;
#         dF *= sigmoidGradient(self._F);
#         dG *= tanhGradient(self._G);
#         dI *= sigmoidGradient(self._I);
#         dO *= sigmoidGradient(self._O);
#         dA = np.hstack((dF, dI, dO, dG));

#         dWx = self._X.T @ dA;
#         dWh = (self._inputDropoutMask * self._H).T @ dA;
#         db = np.sum(dA, axis = 0);

#         dX = dA @ self._weightX.T;
#         dH = (dA @ self._weightH.T) * self._inputDropoutMask;
#         dC = dYC * self._F;

#         self._params[0].grad[...] = dWx;
#         self._params[1].grad[...] = dWh;
#         self._params[2].grad[...] = db;

#         return dX, dH, dC;


#     def setInputDropoutMask(self, mask : Optional[np.ndarray] = None):
#         self._inputDropoutMask = mask;


#     def setRecurrentDropoutMask(self, mask : Optional[np.ndarray] = None):
#         self._recurrentDropoutMask = mask;


# class LstmLayer2(NetModuleBase):
#     def __init__(self, inputSize : int, outputSize : int, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, b : Optional[np.ndarray] = None, returnSequences : bool = False, returnState : bool = False, stateful : bool = False, stepwise = False, inputDropout : float = 0, recurrentDropout : float = 0):
#         super().__init__();

#         self._T = 0;
#         self._H, self._C = None, None;
#         self._dH, self._dC = None, None;
#         self._returnSequences = returnSequences;
#         self._returnState = returnState;
#         self._inputState = False;
#         self._stateful = stateful;
#         self._stepwise = stepwise;
#         self._stepIndex = 0;
#         self._inputSize = inputSize;
#         self._outputSize = outputSize;
#         self._name = f"LSTM {inputSize}*{outputSize}";

#         self._weightX = math.sqrt(2.0 / inputSize) * np.random.randn(inputSize, 4 * outputSize).astype(defaultDType) if Wx is None else Wx;
#         self._weightH = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, 4 * outputSize).astype(defaultDType) if Wh is None else Wh;
#         self._bias = np.zeros(4 * outputSize, dtype = defaultDType) if b is None else b;
#         self._inputDropout = inputDropout;
#         self._recurrentDropout = recurrentDropout;
#         self._inputDropoutMask = None;
#         self._recurrentDropoutMask = None;
#         self._lstmModules : List[LstmCell] = [];

#         self._params.append(NetParamDefinition("weightX", self._weightX));
#         self._params.append(NetParamDefinition("weightH", self._weightH));
#         self._params.append(NetParamDefinition("bias", self._bias, canDecay = False));


#     def _setContext(self, context : INetContext):
#         for cell in self._lstmModules:
#             cell.context = value;


#     def _setParams(self, value: List[np.ndarray]):
#         self._weightX, self._weightH, self._bias = value[0], value[1], value[2];
#         for cell in self._lstmModules:
#             cell.params = value;


#     def _reset(self):
#         self._H, self._C = None, None;
#         self.resetStepState();

#         for cell in self._lstmModules:
#             cell.reset();


#     def _getInputState(self, *data: np.ndarray):
#         if len(data) > 1:
#             return True, data[1], data[2];
#         else:
#             return False, None, None;


#     def _createCell(self) -> LstmCell:
#         cell = LstmCell(self._inputSize, self._outputSize, self._weightX, self._weightH, self._bias, inputDropout = self._inputDropout, recurrentDropout = self._recurrentDropout);
#         cell.context = self.context;
#         cell.setInputDropoutMask(self._inputDropoutMask);
#         cell.setRecurrentDropoutMask(self._recurrentDropoutMask);
#         return cell;


#     def _forwardStep(self, t : int, *data : np.ndarray):
#         X = data[0];
#         self._H, self._C = self._lstmModules[t].forward(X, self._H, self._C);


#     def _backwardStep(self, t : int, *dout: np.ndarray) -> Tuple[np.ndarray]:
#         dY = dout[0];
#         lstm = self._lstmModules[t];
#         dX, self._dH, self._dC = lstm.backward(dY + self._dH, self._dC);

#         for i in range(len(lstm.grads)):
#             self._grads[i] += lstm.grads[i];

#         return dX, ;


#     def _forwardAll(self, *data : np.ndarray) -> np.ndarray:
#         X = data[0];
#         N, T = X.shape[: 2];

#         Y = np.zeros((N, T, self._outputSize), dtype = X.dtype);
#         for t in range(T):
#             self._forwardStep(t, X[:, t]);
#             Y[:, t] = self._H;

#         return Y;


#     def _backwardAll(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         dY = dout[0];
#         N, T = len(dY), self._T;
#         dX = np.zeros((N, T, self._inputSize), dtype = dY.dtype);

#         for t in reversed(range(T)):
#             dX[:, t], = self._backwardStep(t, dY[:, t]);

#         return dX, ;


#     @property
#     def weightX(self) -> np.ndarray:
#         return self._weightX;


#     @property
#     def weightH(self) -> np.ndarray:
#         return self._weightH;


#     @property
#     def bias(self) -> np.ndarray:
#         return self._bias;


#     @property
#     def dH(self) -> np.ndarray:
#         return self._dH;


#     @property
#     def dC(self) -> np.ndarray:
#         return self._dC;


#     # def _reset(self):
#     #     self._H, self._C = None, None;
#     #     self.resetStepState();


#     # input: X, H, C
#     def forward(self, *data: np.ndarray) -> Tuple[np.ndarray]:
#         X = data[0];
#         N = len(X);

#         if not self._stepwise or self._stepIndex == 0:
#             if not self._stateful or self._H is None:
#                 self._H = np.zeros((N, self._outputSize), dtype = X.dtype);
#             if not self._stateful or self._C is None:
#                 self._C = np.zeros((N, self._outputSize), dtype = X.dtype);

#             # only used in unit test!
#             # if self._inputDropoutMask is None:
#             #     self._inputDropoutMask = getDropoutMask(self._H, self._inputDropout);
#             # if self._recurrentDropoutMask is None:
#             #     self._recurrentDropoutMask = getDropoutMask(self._C, self._recurrentDropout);

#             self._inputDropoutMask = getDropoutMask(self._H, self._inputDropout);
#             self._recurrentDropoutMask = getDropoutMask(self._C, self._recurrentDropout);

#             for cell in self._lstmModules:
#                 cell.setInputDropoutMask(self._inputDropoutMask);
#                 cell.setRecurrentDropoutMask(self._recurrentDropoutMask);

#         self._inputState, H, C = self._getInputState(*data);
#         if self._inputState:
#             self._H = H if H is not None else np.zeros((N, self._outputSize), dtype = X.dtype);
#             self._C = C if C is not None else np.zeros((N, self._outputSize), dtype = X.dtype);

#         if not self._stepwise:
#             self._T = T = X.shape[1];

#             if len(self._lstmModules) < T:
#                 self._lstmModules.extend([self._createCell() for _ in range(T - len(self._lstmModules))]);

#             Y = self._forwardAll(*data);
#             if not self._returnSequences:
#                 Y = Y[:, -1];
#         else:
#             while len(self._lstmModules) < self._stepIndex + 1:
#                 self._lstmModules.append(self._createCell());

#             self._forwardStep(self._stepIndex, *data);
#             self._stepIndex += 1;

#             Y = self._H;

#         return (Y, self._H, self._C) if self._returnState else (Y, );


#     # input: dY, dH, dC
#     def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         dY = dout[0];

#         if not self._stepwise or self._stepIndex == len(self._lstmModules):
#             # truncated BPTT
#             self._dH = np.zeros_like(self._H);
#             self._dC = np.zeros_like(self._H);
#             # for i in range(len(self._grads)):
#             #     self._grads[i][...] = 0;

#         if self._returnState:
#             self._dH += dout[1];
#             self._dC += dout[2];

#         if not self._stepwise:
#             N, T = len(dY), self._T;

#             if not self._returnSequences:
#                 dH = dY;
#                 dY = np.zeros((N, T, dY.shape[-1]), dtype = dY.dtype);
#                 dY[:, -1] = dH;

#             din = self._backwardAll(*((dY, ) + dout[1:]));
#         else:
#             self._stepIndex -= 1;
#             din = self._backwardStep(self._stepIndex, *((dY, ) + dout[1:]));

#         if self._inputState:
#             if not self._stepwise:
#                 din += (self._dH, self._dC);
#             else:
#                 din += (np.copy(self._dH), np.copy(self._dC));
#                 self._dH[...] = 0;
#                 self._dC[...] = 0;

#         return din;


#     def setState(self, H : np.ndarray, C : Optional[np.ndarray] = None):
#         self._H, self._C = H, C;


#     def resetStepState(self):
#         if self._stepwise:
#             self._stepIndex = 0;


#     # only used in unit test!
#     def setInputDropoutMask(self, mask : Optional[np.ndarray] = None):
#         self._inputDropoutMask = mask;

#         for cell in self._lstmModules:
#             cell.setInputDropoutMask(self._inputDropoutMask);


#     # only used in unit test!
#     def setRecurrentDropoutMask(self, mask : Optional[np.ndarray] = None):
#         self._recurrentDropoutMask = mask;

#         for cell in self._lstmModules:
#             cell.setRecurrentDropoutMask(self._recurrentDropoutMask);


# class BahdanauAttentionLstmLayer(LstmLayer):
#     def __init__(self, inputSize : int, outputSize : int, Wx : Optional[np.ndarray] = None, Wh : Optional[np.ndarray] = None, b : Optional[np.ndarray] = None, Wq : Optional[np.ndarray] = None, Wk : Optional[np.ndarray] = None, wv : Optional[np.ndarray] = None, returnSequences : bool = False, returnState : bool = False, stateful : bool = False, stepwise = False, inputDropout : float = 0, recurrentDropout : float = 0):
#         super().__init__(outputSize + inputSize, outputSize, Wx, Wh, b, returnSequences, returnState, stateful, stepwise, inputDropout, recurrentDropout);

#         self._shapeK = None;
#         self._attentionWeight = None;
#         self._name = f"BahdanauAttentionLSTM {inputSize}*{outputSize}";

#         self._weightQ = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, outputSize).astype(defaultDType) if Wq is None else Wq;
#         self._weightK = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, outputSize).astype(defaultDType) if Wk is None else Wk;
#         self._weightV = math.sqrt(2.0 / outputSize) * np.random.randn(outputSize, 1).astype(defaultDType) if wv is None else wv;
#         self._attentionModules : List[QKVAttentionLayer] = [];

#         weights = [self._weightQ, self._weightK, self._weightV];
#         self._params.extend(weights);
#         self._grads.extend([np.zeros_like(w) for w in weights]);


#     def _setContext(self, context : INetContext):
#         super()._setContext(context);

#         for m in self._attentionModules:
#             m.context = value;


#     def _setParams(self, value: List[np.ndarray]):
#         self._weightX, self._weightH, self._bias = value[0], value[1], value[2];
#         self._weightQ, self._weightK, self._weightV = value[3], value[4], value[5];

#         for cell in self._lstmModules:
#             cell.params = (self._weightX, self._weightH, self._bias);

#         for m in self._attentionModules:
#             m.params = (self._weightQ, self._weightK, self._weightV);


#     def _reset(self):
#         super()._reset();

#         for m in self._attentionModules:
#             m.reset();


#     def _getInputState(self, *data: np.ndarray):
#         if len(data) > 2:
#             return True, data[2], data[3];
#         else:
#             return False, None, None;


#     def _createAttention(self):
#         layer = QKVAttentionLayer(AdditiveAttentionWeight1TModule(self._outputSize, self._outputSize, self._outputSize, Wq = self._weightQ, Wk = self._weightK, wv = self._weightV), SelectByWeight1TModule());
#         layer.context = self.context;
#         return layer;


#     def _forwardStep(self, t : int, *data : np.ndarray):
#         X, K = data[: 2];

#         lstm = self._lstmModules[t];
#         attention = self._attentionModules[t];

#         context = attention.forward(self._H, K, K)[0];
#         self._attentionWeight.append(attention.attentionWeight);
#         self._H, self._C = lstm.forward(np.concatenate((context, X), axis = -1), self._H, self._C);


#     def _backwardStep(self, t : int, *dout: np.ndarray) -> Tuple[np.ndarray]:
#         dY = dout[0];

#         lstm = self._lstmModules[t];
#         attention = self._attentionModules[t];

#         dX, self._dH, self._dC = lstm.backward(dY + self._dH, self._dC);
#         dContext = dX[:, : self._outputSize];
#         dQ, dK, dV = attention.backward(dContext);
#         self._dH += dQ;

#         grads = lstm.grads + attention.grads;
#         for i in range(len(grads)):
#             self._grads[i] += grads[i];

#         return dX[:, self._outputSize:], dK + dV;


#     def _forwardAll(self, *data : np.ndarray) -> np.ndarray:
#         X, K = data[: 2];
#         N, T = X.shape[: 2];
#         Y = np.zeros((N, T, self._outputSize), dtype = X.dtype);

#         for t in range(T):
#             self._forwardStep(t, X[:, t], K);
#             Y[:, t] = self._H;

#         return Y;


#     def _backwardAll(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         dY = dout[0];
#         N, T = len(dY), self._T;
#         dX = np.zeros((N, T, self._inputSize - self._outputSize), dtype = dY.dtype);
#         dK = np.zeros(self._shapeK, dtype = dY.dtype);

#         for t in reversed(range(T)):
#             dX[:, t], dKS = self._backwardStep(t, dY[:, t]);
#             dK += dKS;

#         return dX, dK;


#     def forward(self, *data: np.ndarray) -> Tuple[np.ndarray]:
#         X, K = data[: 2];
#         self._shapeK = K.shape;

#         if not self._stepwise or self._stepIndex == 0:
#             self._attentionWeight = [];

#         if not self._stepwise:
#             T = X.shape[1];

#             if len(self._attentionModules) < T:
#                 self._attentionModules = [self._createAttention() for _ in range(T - len(self._attentionModules))];
#         else:
#             while len(self._attentionModules) < self._stepIndex + 1:
#                 self._attentionModules.append(self._createAttention());

#         return super().forward(*data);


#     @property
#     def attentionWeight(self) -> np.ndarray:
#         return np.array(self._attentionWeight).transpose(1, 0, 2);


# rnnSelector(inputSize, hiddenSize, stateful, returnSequence, returnState) -> RnnLayerBase
class StackRnnLayer(AggregateNetModule):
    def __init__(self, inputSize : int, hiddenSize : int, rnnSelector : Callable[[int, int, bool, bool, bool], RnnLayerBase], normalSelector : Optional[Callable[[int], INetModule]] = None, layersNum : int = 2, dropoutRatio : float = 0.0, stateful : bool = True, returnSequence : bool = True, returnState : bool = False, bidirectional : bool = False):
        if not returnSequence and not returnState:
            raise ValueError("returnSequence and returnState are both false");

        modules : List[INetModule] = [];
        self._rnnModules : List[INetModule] = [];
        self._normalModules : List[INetModule] = [];
        self._dropoutModules : List[INetModule] = [];

        for l in range(layersNum):
            if bidirectional:
                modules.append(BiRnnLayer(inputSize if l == 0 else 2 * hiddenSize, hiddenSize, rnnSelector, stateful, True, True));
            else:
                modules.append(rnnSelector(inputSize if l == 0 else hiddenSize, hiddenSize, stateful, True, True));
            self._rnnModules.append(modules[-1]);

            if normalSelector is not None and l < layersNum - 1:
                modules.append(normalSelector(2 * hiddenSize if bidirectional else hiddenSize));
                self._normalModules.append(modules[-1]);

            if dropoutRatio > 0 and l < layersNum - 1:
                modules.append(VariationalDropoutLayer(dropoutRatio));
                self._dropoutModules.append(modules[-1]);

        super().__init__(*tuple(modules));

        self._inputSize = inputSize;
        self._hiddenSize = hiddenSize;
        self._layersNum = layersNum;
        self._normalsNum = len(self._normalModules);
        self._dropoutsNum = len(self._dropoutModules);
        self._stateful = stateful;
        self._returnSequence = returnSequence;
        self._returnState = returnState;

        self._Ys = None;
        self._hiddenStates = ();
        self._dStates = None;
        self._foreignState = False;

        if bidirectional:
            # self._name = f"StackBiRnnLayer {inputSize}*{'*'.join([str(2 * hiddenSize)] * layersNum)}";
            self._name = f"StackBiRnnLayer({self._name})";
        else:
            # self._name = f"StackRnnLayer {inputSize}*{'*'.join([str(hiddenSize)] * layersNum)}";
            self._name = f"StackRnnLayer({self._name})";


    # input X, states: [L, N, H]
    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]: # type: ignore
        Xs = data[0];
        self._foreignState = len(data) > 1;

        outputStates = [];
        inputStates = data[1: ] if self._foreignState else None;

        for l in range(self._layersNum):
            layer = self._rnnModules[l];

            if inputStates is not None:
                outputs = layer.forward(*((Xs, ) + tuple(item[l] for item in inputStates)));
            else:
                outputs = layer.forward(Xs);

            Xs = outputs[0];
            outputStates.extend(outputs[1: ]);

            if l < self._normalsNum:
                Xs, = self._normalModules[l].forward(Xs);

            if l < self._dropoutsNum:
                Xs, = self._dropoutModules[l].forward(Xs);

        self._Ys = Xs;
        statesCount = len(outputStates) // self._layersNum;
        self._hiddenStates = tuple(np.array(outputStates[i::statesCount]) for i in range(statesCount));

        if self._returnSequence and self._returnState:
            return Xs, *self._hiddenStates;
        if self._returnSequence:
            return Xs, ;
        if self._returnState:
            return self._hiddenStates ;


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        if self._returnSequence and self._returnState:
            dXs, dOutputStates = dout[0], dout[1: ];
        elif self._returnSequence:
            dXs = dout[0];

            # truncated BPTT
            dOutputStates = tuple(np.zeros_like(item) for item in self._hiddenStates);
        else:
            dXs, dOutputStates = np.zeros_like(self._Ys), dout;

        dInputStates = [];
        for l in reversed(range(self._layersNum)):
            layer = self._rnnModules[l];

            if l < self._dropoutsNum:
                dXs, = self._dropoutModules[l].backward(dXs);
            
            if l < self._normalsNum:
                dXs, = self._normalModules[l].backward(dXs);

            dInputs = layer.backward(*((dXs, ) + tuple(item[l] for item in dOutputStates)));
            dXs = dInputs[0];
            dInputStates.extend(dInputs[1: ]);

        statesCount = len(self._hiddenStates);

        if self._foreignState:
            self._dStates = tuple(np.array(dInputStates[-statesCount + i::-statesCount]) for i in range(statesCount));
        
            return (dXs, ) + self._dStates;
        else:
            self._dStates = None;

            return (dXs, );


# rnnSelector(inputSize, hiddenSize, stateful, returnSequence, returnState) -> RnnLayerBase
class BiRnnLayer(AggregateNetModule):
    def __init__(self, inputSize : int, hiddenSize : int, rnnSelector : Callable[[int, int, bool, bool, bool], RnnLayerBase], stateful : bool = True, returnSequence : bool = True, returnState : bool = False):
        self._forwardRnn : INetModule = rnnSelector(inputSize, hiddenSize, stateful, True, True);
        self._backwardRnn : INetModule = rnnSelector(inputSize, hiddenSize, stateful, True, True);

        super().__init__(self._forwardRnn, self._backwardRnn);

        self._inputSize = inputSize;
        self._hiddenSize = hiddenSize;
        self._stateful = stateful;
        self._returnSequence = returnSequence;
        self._returnState = returnState;

        self._Ys = None;
        self._hiddenStates = ();
        self._dStates = None;
        self._foreignState = False;
        self._name = f"BiRNN {inputSize}*{2 * hiddenSize}";


    @property
    def forwardRnn(self) -> INetModule:
        return self._forwardRnn;


    @property
    def backwardRnn(self) -> INetModule:
        return self._backwardRnn;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]: # type: ignore
        Xs = data[0];
        self._foreignState = len(data) > 1;
        inputStates = data[1:] if self._foreignState else None;

        if inputStates is not None:
            forwardOutputs = self._forwardRnn.forward(*((Xs, ) + tuple(item[..., : self._hiddenSize] for item in inputStates)));
            backwardOutputs = self._backwardRnn.forward(*((Xs[::-1], ) + tuple(item[..., self._hiddenSize: ] for item in inputStates)));
        else:
            forwardOutputs = self._forwardRnn.forward(Xs);
            backwardOutputs = self._backwardRnn.forward(Xs[::-1]);

        self._Ys = np.concatenate((forwardOutputs[0], backwardOutputs[0][::-1]), axis = -1);
        self._hiddenStates = tuple(np.concatenate((fs, bs), axis = -1) for fs, bs in zip(forwardOutputs[1: ], backwardOutputs[1: ]));

        if self._returnSequence and self._returnState:
            return self._Ys, *self._hiddenStates;
        if self._returnSequence:
            return self._Ys, ;
        if self._returnState:
            return self._hiddenStates;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        if self._returnSequence and self._returnState:
            dYs, dOutputStates = dout[0], dout[1:];
        elif self._returnSequence:
            dYs = dout[0];

            # truncated BPTT
            dOutputStates = tuple(np.zeros_like(item) for item in self._hiddenStates);
        else:
            dYs, dOutputStates = np.zeros_like(self._Ys), dout;

        dForwardInputs = self._forwardRnn.backward(dYs[..., : self._hiddenSize], *tuple(item[..., : self._hiddenSize] for item in dOutputStates));
        dBackwardInputs = self._backwardRnn.backward(dYs[..., self._hiddenSize: ][::-1], *tuple(item[..., self._hiddenSize: ] for item in dOutputStates));

        dXs = dForwardInputs[0] + dBackwardInputs[0][::-1];

        if self._foreignState:
            self._dStates = tuple(np.concatenate((fs, bs), axis = -1) for fs, bs in zip(dForwardInputs[1: ], dBackwardInputs[1: ]));
        
            return (dXs, ) + self._dStates;
        else:
            self._dStates = None;
            
            return (dXs, );


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


    def _findSample(self, sampleSize : int, exceptIndex = None) -> np.ndarray:
        p = self._probability;
        if exceptIndex is not None:
            p = p.copy();
            p[exceptIndex] = 0;
            p /= np.sum(p);

        return np.random.choice(self._vocab, sampleSize, replace = DeviceConfig.enableGPU or sampleSize >= self._vocabSize, p = p);


    # return: negative samples, final tags
    def getSample(self, T : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, windowSize : int, vocabSize : int, hiddenSize : int, negativeSampler : CorpusNegativeSampler, inW : Optional[np.ndarray] = None, outW : Optional[np.ndarray] = None):
        self._finalTag = None;
        self._windowSize = windowSize;
        self._vocabSize = vocabSize;
        self._hiddenSize = hiddenSize;
        self._negativeSampler = negativeSampler;

        self._embeddingLayer = EmbeddingLayer(vocabSize, hiddenSize, W = inW);
        self._outputLayer = EmbeddingWithDotLayer(vocabSize, hiddenSize, W = outW);

        super().__init__(self._embeddingLayer, self._outputLayer);
        self._name = "CBOW";


    @property
    def wordVector(self) -> np.ndarray:
        return self._embeddingLayer.weight;


    @property
    def weights(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._embeddingLayer.weight, self._outputLayer.weight;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X, T = data;
        N, C = X.shape;
        S = self._negativeSampler.sampleSize;

        H = self._embeddingLayer.forward(X)[0];
        H = np.sum(H, axis = -2) / C;

        H = np.expand_dims(H, axis = -2);
        H = np.repeat(H, S + 1, axis = -2);
        T, self._finalTag = self._negativeSampler.getSample(T);

        Y = self._outputLayer.forward(H, T)[0];

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        N, C = len(dY), 2 * self._windowSize;

        dH = self._outputLayer.backward(dY)[0];
        dH = np.sum(dH, axis = -2) / C;
        dH = np.expand_dims(dH, axis = -2);
        dH = np.repeat(dH, C, axis = -2);
        dX = self._embeddingLayer.backward(dH)[0];

        return dX, ;


    def getFinalTag(self, T : np.ndarray) -> Optional[np.ndarray]:
        return self._finalTag;


class SkipGramModel(NetModelBase):
    def __init__(self, windowSize : int, vocabSize : int, hiddenSize : int, negativeSampler : CorpusNegativeSampler, inW : Optional[np.ndarray] = None, outW : Optional[np.ndarray] = None):
        self._finalTag = None;
        self._windowSize = windowSize;
        self._vocabSize = vocabSize;
        self._hiddenSize = hiddenSize;
        self._negativeSampler = negativeSampler;

        self._embeddingLayer = EmbeddingLayer(vocabSize, hiddenSize, W = inW);
        self._outputLayer = EmbeddingWithDotLayer(vocabSize, hiddenSize, W = outW);

        super().__init__(self._embeddingLayer, self._outputLayer);
        self._name = "SkipGram";


    @property
    def wordVector(self) -> np.ndarray:
        return self._embeddingLayer.weight;


    @property
    def weights(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._embeddingLayer.weight, self._outputLayer.weight;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
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


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dH = self._outputLayer.backward(dY)[0];
        dH = np.sum(dH, axis = (-2, -3));
        dX = self._embeddingLayer.backward(dH)[0];

        return dX, ;


    def getFinalTag(self, T : np.ndarray) -> Optional[np.ndarray]:
        return self._finalTag;


class SoftmaxLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._Y = np.empty(0);
        self._M = None;
        self._name = "Softmax";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        self._M = data[1] if len(data) > 1 else None;  # softmax mask

        self._Y = softmax(X, self._M);

        return self._Y, ;


    # dX = Y * (dY - ∑(dY * Y))
    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dX = softmaxGradient(self._Y, dY);

        return dX, ;


class ConcatenationLayer(NetModuleBase):
    def __init__(self):
        super().__init__();

        self._index = np.empty(0);
        self._name = "Concatenation";


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        index = [0];
        for item in data:
            index.append(index[-1] + item.shape[-1]);
        self._index = np.array(index[1: -1]);

        return np.concatenate(data, axis = -1), ;


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        return tuple(np.split(dY, self._index, axis = -1));


class GatedAdditiveLayer(NetModuleBase):
    def __init__(self, inputSize : int, W : Optional[np.ndarray] = None, b : Optional[np.ndarray] = None):
        super().__init__();

        self._X1, self._X2 = np.empty(0), np.empty(0);
        self._X, self._A, self._G = np.empty(0), np.empty(0), np.empty(0);

        self._weight = math.sqrt(2.0 / inputSize) * np.random.randn(2 * inputSize, inputSize).astype(defaultDType) if W is None else W;
        self._bias = np.zeros(inputSize, dtype = defaultDType) if b is None else b;
        self._name = f"GatedAdditive {inputSize}";

        self._params.append(NetParamDefinition("weight", self._weight));
        self._params.append(NetParamDefinition("bias", self._bias, canDecay = False));
    

    @property
    def weight(self) -> np.ndarray:
        return self._weight;
    

    @property
    def bias(self) -> np.ndarray:
        return self._bias;
    

    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X1, X2 = data[: 2];
        X = np.concatenate((X1, X2), axis = -1);
        A = X @ self._weight + self._bias;
        G = sigmoid(A);
        Y = X1 * G + X2 * (1 - G);

        self._X1, self._X2 = X1, X2;
        self._X, self._A, self._G = X, A, G;

        return Y, ;
    

    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dX1 = dY * self._G;
        dX2 = dY * (1 - self._G);
        dG = dY * (self._X1 - self._X2);
        dA = dG * sigmoidGradient(self._G);

        dX = dA @ self._weight.T;
        dW = np.swapaxes(self._X, -2, -1) @ dA;
        if dW.ndim > 2:
            dW = np.sum(dW, axis = tuple(range(dW.ndim - 2)));
        db = np.sum(dA, axis = tuple(range(dA.ndim - 1)));
        dX1_, dX2_ = np.split(dX, 2, axis = -1);
        dX1 += dX1_;
        dX2 += dX2_;

        self._params[0].grad[...] += dW;
        self._params[1].grad[...] += db;

        return dX1, dX2;


class CrossEntropyLoss(NetLossBase):
    def __init__(self, reductionType : LossReductionType = LossReductionType.Mean):
        super().__init__();

        self._Y = np.empty(0);
        self._T = np.empty(0);
        self._M = None;
        self._reductionType = reductionType;
    

    @property
    def name(self) -> str:
        return "Cross Entropy Loss";


    def forward(self, *data: np.ndarray) -> float:
        if len(data) > 2:
            self._Y, self._M, self._T = data[: 3];
        else:
            self._Y, self._T = data[: 2]; # type: ignore
            self._M = None;

        self._loss = crossEntropyError(self._Y, self._T, self._M, self._reductionType);

        return self._loss; # type: ignore


    def backward(self) -> Tuple[np.ndarray, ...]:
        return crossEntropyErrorGradient(self._Y, self._T, self._M, self._reductionType), ;


class SoftmaxWithCrossEntropyLoss(NetLossBase):
    def __init__(self, reductionType : LossReductionType = LossReductionType.Mean):
        super().__init__();

        self._Y = np.empty(0);
        self._T = np.empty(0);
        self._M = None;
        self._reductionType = reductionType;
    

    @property
    def name(self) -> str:
        return "Cross Entropy Loss";


    def forward(self, *data: np.ndarray) -> float:
        if len(data) > 2:
            X, self._M, self._T = data[: 3];
        else:
            X, self._T = data[: 2]; # type: ignore
            self._M = None;

        self._Y = softmax(X);
        self._loss = crossEntropyError(self._Y, self._T, self._M, self._reductionType);

        return self._loss; # type: ignore


    def backward(self) -> Tuple[np.ndarray, ...]:
        return softmaxWithCrossEntropyErrorGradient(self._Y, self._T, self._M, self._reductionType), ;


class SoftmaxWithCrossEntropy1DLoss(NetLossBase):
    def __init__(self, reductionType : LossReductionType = LossReductionType.Mean):
        super().__init__();

        self._Y = np.empty(0);
        self._T = np.empty(0);
        self._M = None;
        self._reductionType = reductionType;
    

    @property
    def name(self) -> str:
        return "Cross Entropy Loss";


    def forward(self, *data: np.ndarray) -> float:
        if len(data) > 2:
            X, self._M, self._T = data[: 3];
        else:
            X, self._T = data[: 2]; # type: ignore
            self._M = None;

        self._Y = softmax(X);
        self._loss = crossEntropyError1D(self._Y, self._T, self._M, self._reductionType);

        return self._loss; # type: ignore


    def backward(self) -> Tuple[np.ndarray, ...]:
        return softmaxWithCrossEntropyErrorGradient1D(self._Y, self._T, self._M, self._reductionType), ;


class SequenceSoftmaxWithCrossEntropy1DLoss(SoftmaxWithCrossEntropy1DLoss):
    def __init__(self):
        super().__init__(reductionType = LossReductionType.No);

        self._n = 0;


    def forward(self, *data: np.ndarray) -> float:
        X, validLength, T = data;
        M = getLossMaskByValidLength(T.shape[-1], validLength) if validLength is not None else None;
        L = super().forward(X, M, T); # type: ignore

        n = np.sum(M) if M is not None else X.size // X.shape[-1];
        loss = float(np.sum(L) / n);

        self._n = n;
        return loss;


    def backward(self) -> Tuple[np.ndarray, ...]:
        dX, = super().backward();
        dX /= self._n;

        return dX, ;


class SigmoidWithCrossEntropyLoss(NetLossBase):
    def __init__(self, reductionType : LossReductionType = LossReductionType.Mean):
        super().__init__();

        self._Y = np.empty(0);
        self._T = np.empty(0);
        self._M = None;
        self._reductionType = reductionType;
    

    @property
    def name(self) -> str:
        return "Cross Entropy Loss";


    def forward(self, *data: np.ndarray) -> float:
        if len(data) > 2:
            X, self._M, self._T = data[: 3];
        else:
            X, self._T = data[: 2]; # type: ignore
            self._M = None;

        self._Y = sigmoid(X);

        Y2 = np.expand_dims(self._Y, axis = -1);
        T2 = np.expand_dims(self._T, axis = -1);
        self._loss = crossEntropyError(np.concatenate((Y2, 1 - Y2), axis = -1),
                                       np.concatenate((T2, 1 - T2), axis = -1), self._M, self._reductionType);

        return self._loss; # type: ignore


    def backward(self) -> Tuple[np.ndarray, ...]:
        return sigmoidWithCrossEntropyErrorGradient(self._Y, self._T, self._M, self._reductionType), ;


class IdentityWithMeanSquareLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._Y = np.empty(0);
        self._T = np.empty(0);
        self._W = np.empty(0);
    

    @property
    def name(self) -> str:
        return "MSE";


    def forward(self, *data: np.ndarray) -> float:
        if len(data) > 2:
            self._Y, self._W, self._T = data;
        else:
            self._Y, self._T = data; # type: ignore
            self._W = None;

        self._loss = meanSquareError(self._Y, self._T, W = self._W) / 2;

        return self._loss;


    def backward(self) -> Tuple[np.ndarray, ...]:
        if self._W is not None:
            dY = self._W * (self._Y - self._T) / float(np.sum(self._W));
        else:
            dY = (self._Y - self._T) / self._T.size;

        return dY, ;


class IdentityWithMeanAbsoluteLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._Y, self._T = np.empty(0), np.empty(0);
        self._W = np.empty(0);
    

    @property
    def name(self) -> str:
        return "MAE";


    def forward(self, *data: np.ndarray) -> float:
        if len(data) > 2:
            self._Y, self._W, self._T = data;
        else:
            self._Y, self._T = data; # type: ignore
            self._W = None;

        self._loss = meanAbsoluteError(self._Y, self._T, W = self._W);

        return self._loss;


    def backward(self) -> Tuple[np.ndarray, ...]:
        ML = self._Y < self._T;
        MH = self._Y > self._T;

        if self._W is not None:
            dY = self._W * (MH * 1 - ML * 1).astype(self._Y.dtype) / float(np.sum(self._W));
        else:
            dY = (MH * 1 - ML * 1).astype(self._Y.dtype) / self._T.size;

        return dY, ;


class IdentityWithMeanAbsolutePercentLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._Y = np.empty(0);
        self._T = np.empty(0);
        self._W = np.empty(0);
    

    @property
    def name(self) -> str:
        return "MAPE";


    def forward(self, *data: np.ndarray) -> float:
        if len(data) > 2:
            self._Y, self._W, self._T = data;
        else:
            self._Y, self._T = data; # type: ignore
            self._W = None;

        self._loss = meanAbsolutePercentError(self._Y, self._T, W = self._W);

        return self._loss;


    def backward(self) -> Tuple[np.ndarray, ...]:
        ML = self._Y < self._T;
        MH = self._Y > self._T;

        if self._W is not None:
            dY = self._W * (MH * 1 - ML * 1).astype(self._Y.dtype) / np.fabs(self._T) / float(np.sum(self._W));
        else:
            dY = (MH * 1 - ML * 1).astype(self._Y.dtype) / np.fabs(self._T) / self._T.size;

        return dY, ;


class IdentityWithHuberLoss(NetLossBase):
    def __init__(self, delta : float = 1.0):
        super().__init__();

        self._delta = np.array(max(0.0, float(delta)), dtype = defaultDType);
        self._Y, self._T = np.empty(0), np.empty(0);
        self._W = np.empty(0);

        self._ML, self._MM, self._MH = np.empty(0), np.empty(0), np.empty(0);
    

    @property
    def name(self) -> str:
        return "Huber Loss";


    def forward(self, *data: np.ndarray) -> float:
        if len(data) > 2:
            self._Y, self._W, self._T = data;
        else:
            self._Y, self._T = data; # type: ignore
            self._W = None;

        self._loss, self._ML, self._MM, self._MH = huberError(self._Y, self._T, self._delta, W = self._W);

        return self._loss;


    def backward(self) -> Tuple[np.ndarray, ...]:
        dY = self._ML * (-self._delta) + self._MM * (self._Y - self._T) + self._MH * self._delta;

        if self._W is not None:
            dY *= self._W / float(np.sum(self._W));
        else:
            dY /= self._T.size;

        return dY, ;


class SumWithMeanSquareLoss(NetLossBase):
    def __init__(self):
        super().__init__();

        self._X = np.empty(0);
        self._Y = np.empty(0);
        self._T = np.empty(0);
        # self._shape = None;

    
    @property
    def name(self) -> str:
        return "MSE";


    def forward(self, *data: np.ndarray) -> float:
        X, T = data;
        self._T = T;
        self._X = X;
        self._Y = np.sum(self._X, axis = -1, keepdims = True);
        self._loss = meanSquareError(self._Y, self._T) / 2;
        # self._shape = X.shape;

        return self._loss;


    def backward(self) -> Tuple[np.ndarray, ...]:
        dY = (self._Y - self._T) / self._T.size;
        dX = np.repeat(dY, self._X.shape[-1], axis = -1);

        return dX, ;


class L1Regularization(INetParamHandler):
    def __init__(self, decay : float = 0.01):
        self._decay = max(0.0, decay);


    def onPreUpdate(self, param: INetParam, lr: float):
        if self._decay == 0:
            return;

        grad = param.grad;
        grad += self._decay * np.sign(param.value).astype(grad.dtype);


    def onPostUpdate(self, param: INetParam, lr: float):
        pass;


class L1WeightDecay(INetParamHandler):
    def __init__(self, decay : float = 0.01):
        self._decay = max(0.0, decay);


    def onPreUpdate(self, param: INetParam, lr: float):
        if self._decay == 0:
            return;

        value = param.value;
        value -= lr * self._decay * np.sign(param.value).astype(value.dtype);


    def onPostUpdate(self, param: INetParam, lr: float):
        pass;


class L2Regularization(INetParamHandler):
    def __init__(self, decay : float = 0.01):
        self._decay = max(0.0, decay);


    def onPreUpdate(self, param: INetParam, lr: float):
        if self._decay == 0:
            return;

        grad = param.grad;
        grad += self._decay * param.value;


    def onPostUpdate(self, param: INetParam, lr: float):
        pass;


class L2WeightDecay(INetParamHandler):
    def __init__(self, decay : float = 0.01):
        self._decay = max(0.0, decay);


    def onPreUpdate(self, param: INetParam, lr: float):
        if self._decay == 0:
            return;

        value = param.value;
        value -= lr * self._decay * value;


    def onPostUpdate(self, param: INetParam, lr: float):
        pass;


class _ParametersShareInfo:
    def __init__(self, index : int, target : int, isTranspose : bool = False):
        self.index = index;
        self.target = target;
        self.isTranspose = isTranspose;


    def __repr__(self):
        return self.__str__();


    def __str__(self):
        return f"[{self.target}] += [{self.index}]{'.T' if self.isTranspose else ''}";


# class ParametersShare(INetOptimizer):
#     def __init__(self, optimizer : INetOptimizer):
#         self._optimizer = optimizer;
#         self._sharesInfo: Optional[List[_ParametersShareInfo]] = None;


#     @property
#     def learningRate(self) -> float:
#         return self._optimizer.learningRate;


#     @learningRate.setter
#     def learningRate(self, value: float):
#         self._optimizer.learningRate = value;


#     def _find(self, params : List[np.ndarray]):
#         L = len(params);
#         sharesInfo = [];
#         params = params[:];

#         for i in range(L - 1):
#             if (p1 := params[i]) is None:
#                 continue;

#             for j in range(i + 1, L):
#                 if (p2 := params[j]) is None:
#                     continue;

#                 if p1 is p2:
#                     # p1 == p2
#                     sharesInfo.append(_ParametersShareInfo(j, i));
#                     params[j] = None;
#                 elif p1.ndim == 2 and p2.ndim == 2 and (p1 is p2.base or p2 is p1.base):
#                     # p1 == p2.T or p1.T == p2
#                     s1, s2 = p1.shape, p2.shape;

#                     if s1[0] == s2[1] and s1[1] == s2[0]:
#                         if p1 is p2.base:
#                             sharesInfo.append(_ParametersShareInfo(j, i, isTranspose = True));
#                             params[j] = None;
#                         else:
#                             sharesInfo.append(_ParametersShareInfo(i, j, isTranspose = True));
#                             params[i] = None;
#                             break;

#         self._sharesInfo = sorted(sharesInfo, key = lambda item: item.index, reverse = True);


#     def updateStep(self, params : List[np.ndarray], grads : List[np.ndarray]):
#         L = len(params);
#         params, grads = params[:], grads[:];

#         if self._sharesInfo is None:
#             self._find(params);

#         for info in self._sharesInfo:
#             grads[info.target] += (grads[info.index].T if info.isTranspose else grads[info.index]);
#             params.pop(info.index);
#             grads.pop(info.index);

#         self._optimizer.updateStep(params, grads);


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
    

    def epochStep(self, epoch : int):
        pass;


    def updateStep(self, params : List[INetParamDefinition], context : INetContext) -> Optional[List[INetParamDefinition]]:
        totalL2 = sum([float(np.sum(p.grad ** 2)) for p in params]);
        ratio = self._maxL2 / math.sqrt(totalL2 + self._epsilon);

        if ratio < 1:
            for p in params:
                grad = p.grad;
                grad *= ratio;

        return self._optimizer.updateStep(params, context);


class SGD(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, weightDecay : float = 0.0, decoupledDecay : bool = False):
        super().__init__(lr, weightDecay = weightDecay, decoupledDecay = decoupledDecay);


    def _onUpdate(self, params : List[INetParamDefinition]):
        for p in params:
            paramValue = p.value;
            paramValue -= self._lr * p.grad;


class SGDM(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, beta : float = 0.9, weightDecay : float = 0.0, decoupledDecay : bool = False):
        super().__init__(lr, weightDecay = weightDecay, decoupledDecay = decoupledDecay);

        self._beta = beta;
        self._momentum : Optional[List[np.ndarray]] = None;


    def _onUpdate(self, params : List[INetParamDefinition]):
        if self._momentum is None:
            self._momentum = [np.zeros_like(p.value) for p in params];

        for m, p in zip(self._momentum, params):
            m[...] = self._beta * m + p.grad;

            paramValue = p.value;
            paramValue -= self._lr * m;


# class Nesterov(NetOptimizerBase):
#     def __init__(self, lr : float = 0.001, beta : float = 0.9):
#         super().__init__(lr);
#
#         self._m = None;
#         self._beta = beta;
#
#
#     def update(self, params: List[np.ndarray], grads: List[np.ndarray]):
#         if self._m is None:
#             self._m = [np.zeros_like(item) for item in params];
#
#         for i in range(len(params)):
#             m = self._m[i];
#             self._m[i] = self._beta * self._m[i] + self._lr * grads[i];
#             params[i] -= (1 + self._beta) * self._m[i] - self._beta * m;


class AdaGrad(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, epsilon : float = 1e-8, weightDecay : float = 0.0, decoupledDecay : bool = False):
        super().__init__(lr, weightDecay = weightDecay, decoupledDecay = decoupledDecay);

        self._v = None;
        self._epsilon = epsilon;


    def _onUpdate(self, params : List[INetParamDefinition]):
        if self._v is None:
            self._v = [np.zeros_like(p.value) for p in params];

        for v, p in zip(self._v, params):
            v += p.grad ** 2;

            paramValue = p.value;
            paramValue -= self._lr * p.grad / (np.sqrt(v) + self._epsilon);


class RMSProp(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, beta : float = 0.9, epsilon : float = 1e-8, weightDecay : float = 0.0, decoupledDecay : bool = False):
        super().__init__(lr, weightDecay = weightDecay, decoupledDecay = decoupledDecay);

        self._v = None;
        self._beta = beta;
        self._epsilon = epsilon;


    def _onUpdate(self, params : List[INetParamDefinition]):
        if self._v is None:
            self._v = [np.zeros_like(p.value) for p in params];

        for v, p in zip(self._v, params):
            v[...] = self._beta * v + (1 - self._beta) * p.grad ** 2;

            paramValue = p.value;
            paramValue -= self._lr * p.grad / (np.sqrt(v) + self._epsilon);


class AdaDelta(NetOptimizerBase):
    def __init__(self, lr : float = 1.0, beta : float = 0.9, epsilon : float = 1e-8, weightDecay : float = 0.0, decoupledDecay : bool = False):
        super().__init__(lr, weightDecay = weightDecay, decoupledDecay = decoupledDecay);

        self._v = None;
        self._d = None;
        self._beta = beta;
        self._epsilon = epsilon;


    def _onUpdate(self, params : List[INetParamDefinition]):
        if self._v is None:
            self._v = [np.zeros_like(p.value) for p in params];
        if self._d is None:
            self._d = [np.zeros_like(p.value) for p in params];

        for v, d, p in zip(self._v, self._d, params):
            v[...] = self._beta * v + (1 - self._beta) * p.grad ** 2;
            g = (np.sqrt(d + self._epsilon) / np.sqrt(v + self._epsilon)) * p.grad;
            d[...] = self._beta * d + (1 - self._beta) * g ** 2;

            paramValue = p.value;
            paramValue -= self._lr * g;


class Adam(NetOptimizerBase):
    def __init__(self, lr : float = 0.001, beta1 : float = 0.9, beta2 : float = 0.999, yogi : bool = False, epsilon : float = 1e-8, weightDecay : float = 0.0, decoupledDecay : bool = False):
        super().__init__(lr, weightDecay = weightDecay, decoupledDecay = decoupledDecay);

        self._m = None;
        self._v = None;
        self._beta1 = beta1;
        self._beta2 = beta2;
        self._yogi = yogi;
        self._epsilon = epsilon;
        self._t = 0;


    def _onUpdate(self, params : List[INetParamDefinition]):
        if self._m is None:
            self._m = [np.zeros_like(item.value) for item in params];
        if self._v is None:
            self._v = [np.zeros_like(item.value) for item in params];

        self._t += 1;

        for m, v, p in zip(self._m, self._v, params):
            g2 = p.grad ** 2;

            m[...] = self._beta1 * m + (1 - self._beta1) * p.grad;

            if self._yogi:
                v[...] = v + (1 - self._beta2) * g2 * np.sign(g2 - v);
            else:
                v[...] = self._beta2 * v + (1 - self._beta2) * g2;

            mc = m / (1 - self._beta1 ** self._t);
            vc = v / (1 - self._beta2 ** self._t);

            paramValue = p.value;
            paramValue -= self._lr * mc / (np.sqrt(vc) + self._epsilon);


class AdamW(Adam):
    def __init__(self, lr : float = 0.001, beta1 : float = 0.9, beta2 : float = 0.999, yogi : bool = False, epsilon : float = 1e-8, weightDecay : float = 0.01):
        super().__init__(lr, beta1, beta2, yogi, epsilon, weightDecay, True);


class AveragedWeightNetOptimizer(INetOptimizer):
    def __init__(self, optimizer : INetOptimizer, avgFunc : Optional[Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = None):
        if optimizer is None:
            raise ValueError("optimizer is None");

        self._avgFunc = avgFunc if avgFunc is not None else AveragedWeightNetOptimizer._defaultAvgFunc;
        self._shadowParams : List[INetParamDefinition] = [];
        self._optimizer = optimizer;
        self._averagedNum = 0;
        

    @staticmethod
    def _defaultAvgFunc(shadowValue : np.ndarray, modelValue : np.ndarray, averagedNum : int) -> np.ndarray:
        return (shadowValue * averagedNum + modelValue) / (averagedNum + 1);


    def _initParams(self, params : List[INetParamDefinition], context : INetContext):
        self._averagedNum = 0;
        self._shadowParams = [p.copy(False) for p in params];


    def _averageParams(self, params : List[INetParamDefinition], context : INetContext):
        for shadowParam, modelParam in zip(self._shadowParams, params):
            shadowParam.value[...] = self._avgFunc(shadowParam.value, modelParam.value, self._averagedNum);
        
        self._averagedNum += 1;


    @property
    def learningRate(self) -> float:
        return self._optimizer.learningRate;


    @learningRate.setter
    def learningRate(self, value : float):
        self._optimizer.learningRate = value;
    

    @property
    def shadowParams(self) -> List[INetParamDefinition]:
        return self._shadowParams;


    def epochStep(self, epoch : int):
        self._optimizer.epochStep(epoch);


    def updateStep(self, params : List[INetParamDefinition], context : INetContext) -> Optional[List[INetParamDefinition]]:
        self._optimizer.updateStep(params, context);

        if len(self._shadowParams) != len(params):
            self._initParams(params, context);

        self._averageParams(params, context);
        
        return self._shadowParams;


class ExponentialAvgWeightNetOptimizer(AveragedWeightNetOptimizer):
    def __init__(self, optimizer : INetOptimizer, decay : float = 0.99):
        super().__init__(optimizer);

        self._decay = max(0, min(1, float(decay)));
        self._1_Decay = 1 - self._decay;
        self._avgFunc = self._emaAvgFunc;


    def _emaAvgFunc(self, shadowValue : np.ndarray, modelValue : np.ndarray, averagedNum : int) -> np.ndarray:
        return self._decay * shadowValue + self._1_Decay * modelValue;


class NetOptimizerWithLrScheduler(INetOptimizer):
    def __init__(self, optimizer : INetOptimizer, scheduler : INetLrScheduler):
        if optimizer is None:
            raise ValueError("optimizer is None");
        if scheduler is None:
            raise ValueError("scheduler is None");

        self._optimizer = optimizer;
        self._scheduler = scheduler;


    @property
    def learningRate(self) -> float:
        return self._scheduler.learningRate;


    @learningRate.setter
    def learningRate(self, value: float):
        raise ValueError("not support to set lr");


    def epochStep(self, epoch : int):
        self._scheduler.epochStep(epoch);

        if self._optimizer.learningRate != self._scheduler.learningRate:
            self._optimizer.learningRate = self._scheduler.learningRate;

        print(f"the learning rate of epoch {epoch} is {self._scheduler.learningRate}");


    def updateStep(self, params: List[INetParamDefinition], context: INetContext) -> Optional[List[INetParamDefinition]]:
        return self._optimizer.updateStep(params, context);


class ConstantNetLrScheduler(NetLrSchedulerBase):
    def __init__(self, baseLr : float, minEpoch : int = 0, maxEpoch : Optional[int] = None):
        super().__init__(baseLr, minEpoch = minEpoch, maxEpoch = maxEpoch);


    def epochStep(self, epoch : int):
        pass;


class LinearNetLrScheduler(NetLrSchedulerBase):
    def __init__(self, baseLr : float, startFactor : float = 0.2, endFactor = 1.0, minEpoch : int = 0, maxEpoch : int = 5):
        if maxEpoch is None:
            raise Value("maxEpoch is None");

        super().__init__(baseLr, minEpoch = minEpoch, maxEpoch = maxEpoch);

        self._startLr = self._baseLr * startFactor;
        self._endLr = self._baseLr * endFactor;
        self._lrDelta = self._baseLr * (endFactor - startFactor) / (maxEpoch - minEpoch);


    def epochStep(self, epoch : int):
        if epoch < self._minEpoch:
            self._currentLr = self._startLr;
        elif epoch <= self._maxEpoch: # type: ignore
            self._currentLr = self._startLr + self._lrDelta * (epoch - self._minEpoch);
        else:
            self._currentLr = self._endLr;


class MultiStepNetLrScheduler(NetLrSchedulerBase):
    def __init__(self, baseLr : float, milestones : List[int], gamma : float = 0.5, minEpoch : int = 0, maxEpoch : Optional[int] = None):
        super().__init__(baseLr, minEpoch = minEpoch, maxEpoch = maxEpoch);

        self._milestones = milestones;
        self._gamma = gamma;
        self._startLr = baseLr;
        self._endLr = baseLr * gamma ** len(milestones);


    def epochStep(self, epoch : int):
        if epoch < self._minEpoch:
            self._currentLr = self._startLr;
        else:
            if self._maxEpoch is None or epoch <= self._maxEpoch:
                idx = [i for i, m in enumerate(self._milestones) if m > epoch];
                self._currentLr = self._baseLr * math.pow(self._gamma, idx[0] if len(idx) > 0 else len(self._milestones));
            else:
                self._currentLr = self._endLr;


class CosineNetLrScheduler(NetLrSchedulerBase):
    def __init__(self, baseLr : float, minLr : float, minEpoch : int = 0, maxEpoch : int = 5):
        if baseLr <= minLr:
            raise ValueError("baseLr <= minLr");
        if maxEpoch is None:
            raise ValueError("maxEpoch is None");

        super().__init__(baseLr, minEpoch = minEpoch, maxEpoch = maxEpoch);

        self._minLr = minLr;


    def epochStep(self, epoch : int):
        if epoch < self._minEpoch:
            self._currentLr = self._baseLr;
        elif epoch <= self._maxEpoch: # type: ignore
            self._currentLr = self._minLr + (self._baseLr - self._minLr) * 0.5 * (1 + math.cos(math.pi * (epoch - self._minEpoch) / (self._maxEpoch - self._minEpoch))); # type: ignore
        else:
            self._currentLr = self._minLr;


class CyclicNetLrScheduler(NetLrSchedulerBase):
    def __init__(self, scheduler : INetLrScheduler, cycleSize : int, baseLr : float = 0.0, minEpoch : int = 0, maxEpoch : Optional[int] = None):
        if scheduler is None:
            raise ValueError("scheduler is None");
        if cycleSize <= 0:
            raise ValueError("cycleSize <= 0");

        super().__init__(baseLr, minEpoch = minEpoch, maxEpoch = maxEpoch);

        self._scheduler = scheduler;
        self._cycleSize = cycleSize;


    def epochStep(self, epoch : int):
        if epoch < self._minEpoch:
            self._currentLr = self._baseLr;
        else:
            if self._maxEpoch is None or epoch <= self._maxEpoch:
                self._scheduler.epochStep((epoch - self._minEpoch) % self._cycleSize + self._scheduler.minEpoch);
                self._currentLr = self._scheduler.learningRate;
            else:
                self._currentLr = self._baseLr;


class AggregateNetLrScheduler(NetLrSchedulerBase):
    def __init__(self, schedulers : List[INetLrScheduler], baseLr : float = 0.0, minEpoch : int = 0, maxEpoch : Optional[int] = None):
        super().__init__(baseLr, minEpoch = minEpoch, maxEpoch = maxEpoch);

        self._schedulers = schedulers;


    def epochStep(self, epoch : int):
        scheduler = None;

        for item in self._schedulers:
            if item.isAvailable(epoch):
                scheduler = item;
                break;

        if scheduler is not None:
            scheduler.epochStep(epoch);
            self._currentLr = scheduler.learningRate;
        else:
            self._currentLr = self._baseLr;


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


    def fit(self, X : np.ndarray) -> np.ndarray:
        Y = X;
        for scaler in self._scalers:
            Y = scaler.fit(Y);
        return Y;


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
        self._ndim = 0;
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
        self._minValue = np.empty(0);
        self._maxValue = np.empty(0);
        self._valueDelta = np.empty(0);


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

    def __init__(self, takeMedian : bool = False):
        super().__init__();

        self._mu = np.empty(0);
        self._sigma = np.empty(0);
        self._takeMedian = takeMedian;


    def _getParams(self) -> List:
        return [self._mu, self._sigma];


    def _setParams(self, value: Tuple):
        self._mu, self._sigma = value;


    def _fit(self, X: np.ndarray):
        if not self._takeMedian:
            self._mu = np.mean(X, axis = 0);
            self._sigma = np.std(X, axis = 0);
        else:
            self._mu = np.median(X, axis = 0);
            self._sigma = np.median(np.abs(X - self._mu), axis = 0);
        return X;


    def _transform(self, X : np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sigma;


    def _inverse(self, Y : np.ndarray, *args, **kwargs) -> np.ndarray:
        index = np.array(kwargs[StandardScaler.INDEX_ARGUMENT_NAME]) if StandardScaler.INDEX_ARGUMENT_NAME in kwargs else None;
        return self._sigma * Y + self._mu if index is None else self._sigma[index] * Y + self._mu[index];


class RobustScaler(ScalerBase):
    def __init__(self):
        super().__init__();

        self._mu = np.empty(0);
        self._iqr = np.empty(0);


    def _getParams(self) -> List:
        return [self._mu, self._iqr];


    def _setParams(self, value: Tuple):
        self._mu, self._iqr = value;


    def _fit(self, X: np.ndarray):
        self._mu = np.median(X, axis = 0);
        self._iqr = np.quantile(X, 0.75, axis = 0) -np.quantile(X, 0.25, axis = 0);

        return X;


    def _transform(self, X : np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._iqr;


    def _inverse(self, Y : np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._iqr * Y + self._mu;


class DiffScaler(ScalerBase):
    INDEX_ARGUMENT_NAME = "index";

    def __init__(self, interval : int = 1):
        super().__init__();

        self._X = np.empty(0);
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


    def apply(self, func : Callable[[INetModule], Any]):
        for m in self._modules:
            func(m);


class AdditiveAttentionModule(AggregateNetModule, INetAttentionModule):
    def __init__(self, querySize : int, keySize : int, hiddenSize : int, dropoutRatio : float = 0.0, Wq : Optional[np.ndarray] = None, Wk : Optional[np.ndarray] = None, wv : Optional[np.ndarray] = None):
        self._dropoutLayer = DropoutLayer(dropoutRatio);
        super().__init__(self._dropoutLayer);

        self._Q, self._K, self._V, self._A = np.empty(0), np.empty(0), np.empty(0), np.empty(0);
        self._attentionWeight, self._dropoutWeight = np.empty(0), np.empty(0);
        self._name = f"AdditiveAttention {hiddenSize}" if dropoutRatio <= 0 else f"AdditiveAttention {hiddenSize}*{dropoutRatio}";

        self._weightQ = math.sqrt(2.0 / querySize) * np.random.randn(querySize, hiddenSize).astype(defaultDType) if Wq is None else Wq;
        self._weightK = math.sqrt(2.0 / keySize) * np.random.randn(keySize, hiddenSize).astype(defaultDType) if Wk is None else Wk;
        self._weightV = math.sqrt(2.0 / hiddenSize) * np.random.randn(hiddenSize, 1).astype(defaultDType) if wv is None else wv;

        self._initSelfParams([
            NetParamDefinition("weightQ", self._weightQ),
            NetParamDefinition("weightK", self._weightK),
            NetParamDefinition("weightV", self._weightV)
        ]);


    def _setSelfParams(self, params: List[INetParamDefinition]):
        self._weightQ, self._weightK, self._weightV = params[0].value, params[1].value, params[2].value;


    @property
    def attentionWeight(self) -> Optional[np.ndarray]:
        return self._attentionWeight;


    # Q shape: (batch_size, query_num, query_size)
    # K shape: (batch_size, pair_num, key_size)
    # V shape: (batch_size, pair_num, value_size)
    # M shape: (batch_size, query_num, pair_num)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        Q, K, V = data[: 3];
        M = data[3] if len(data) > 3 else None;  # softmax mask
        queryNum, keyNum =  Q.shape[-2], K.shape[-2];

        self._Q = np.repeat(np.expand_dims(Q, axis = -2), keyNum, axis = -2);
        self._K = np.repeat(np.expand_dims(K, axis = -3), queryNum, axis = -3);
        self._V = V;

        self._A = tanh(self._Q @ self._weightQ + self._K @ self._weightK);
        S = self._A @ self._weightV;

        self._attentionWeight = softmax(np.squeeze(S, axis = -1), M);
        self._dropoutWeight, = self._dropoutLayer.forward(self._attentionWeight);
        Y = self._dropoutWeight @ self._V;

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dDropoutWeight = dY @ np.swapaxes(self._V, -1, -2);
        dV = np.swapaxes(self._dropoutWeight, -1, -2) @ dY;
        dAttentionWeight, = self._dropoutLayer.backward(dDropoutWeight);
        dS = np.expand_dims(softmaxGradient(self._attentionWeight, dAttentionWeight), axis = -1);

        dA = dS @ self._weightV.T;
        dWeightV = np.sum(np.swapaxes(self._A, -1, -2) @ dS, axis = tuple(range(self._A.ndim - 2)));
        dA *= tanhGradient(self._A);
        dQ = dA @ self._weightQ.T;
        dWeightQ = np.sum(np.swapaxes(self._Q, -1, -2) @ dA, axis = tuple(range(self._Q.ndim - 2)));
        dK = dA @ self._weightK.T;
        dWeightK = np.sum(np.swapaxes(self._K, -1, -2) @ dA, axis = tuple(range(self._K.ndim - 2)));

        dQ = np.sum(dQ, axis = -2);
        dK = np.sum(dK, axis = -3);

        self._params[0].grad[...] += dWeightQ;
        self._params[1].grad[...] += dWeightK;
        self._params[2].grad[...] += dWeightV;

        return dQ, dK, dV;


class DotProductAttentionModule(AggregateNetModule, INetAttentionModule):
    def __init__(self, dropoutRatio : float = 0.0):
        self._dropoutLayer = DropoutLayer(dropoutRatio);
        super().__init__(self._dropoutLayer);

        self._Q, self._K, self._V, self._A, self._scale = None, None, np.empty(0), None, 1;
        self._attentionWeight, self._dropoutWeight = np.empty(0), np.empty(0);
        self._name = f"DotProductAttention" if dropoutRatio <= 0 else f"DotProductAttention {dropoutRatio}";


    @property
    def attentionWeight(self) -> np.ndarray:
        return self._attentionWeight;


    # Q shape: (batch_size, query_num, query_size)
    # K shape: (batch_size, pair_num, key_size)
    # V shape: (batch_size, pair_num, value_size)
    # M shape: (batch_size, query_num, pair_num)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        self._Q, self._K, self._V = data[: 3];
        M = data[3] if len(data) > 3 else None;  # softmax mask
        self._scale = 1.0 / math.sqrt(self._Q.shape[-1]);

        self._A = (self._Q @ np.swapaxes(self._K, -1, -2)) * self._scale;
        self._attentionWeight = softmax(self._A, M);
        self._dropoutWeight, = self._dropoutLayer.forward(self._attentionWeight);
        Y = self._dropoutWeight @ self._V;

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dDropoutWeight = dY @ np.swapaxes(self._V, -1, -2);
        dV = np.swapaxes(self._dropoutWeight, -1, -2) @ dY;
        dAttentionWeight, = self._dropoutLayer.backward(dDropoutWeight);
        dA = softmaxGradient(self._attentionWeight, dAttentionWeight);
        dA *= self._scale;
        dQ = dA @ self._K;
        dK = np.swapaxes(dA, -1, -2) @ self._Q;

        return dQ, dK, dV;


class MultiHeadAttentionModule(AggregateNetModule, INetAttentionModule):
    def __init__(self, attentionModule : INetAttentionModule, querySize : int, keySize : int, valueSize : int, hiddenSize : Union[int, Tuple[int, int, int, int]], headNum : int = 2, Wq : Optional[np.ndarray] = None, Wk : Optional[np.ndarray] = None, Wv : Optional[np.ndarray] = None, Wo : Optional[np.ndarray] = None):
        if headNum < 1:
            raise ValueError("headNum < 1");

        self._attentionModule = attentionModule;
        super().__init__(self._attentionModule);

        if isinstance(hiddenSize, tuple):
            queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize = hiddenSize;
        else:
            queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize = (hiddenSize, ) * 4;

        self._headNum = headNum;
        self._headShape = (headNum, -1);
        self._Q, self._K, self._V, self._C = np.empty(0), np.empty(0), np.empty(0), np.empty(0);
        self._name = f"MultiHeadAttention({headNum}*{hiddenSize}, {attentionModule})";

        self._weightQ = math.sqrt(2.0 / querySize) * np.random.randn(querySize, headNum * queryHiddenSize).astype(defaultDType) if Wq is None else Wq;
        self._weightK = math.sqrt(2.0 / keySize) * np.random.randn(keySize, headNum * keyHiddenSize).astype(defaultDType) if Wk is None else Wk;
        self._weightV = math.sqrt(2.0 / valueSize) * np.random.randn(valueSize, headNum * valueHiddenSize).astype(defaultDType) if Wv is None else Wv;
        self._weightO = math.sqrt(2.0 / (headNum * valueHiddenSize)) * np.random.randn(headNum * valueHiddenSize, outputHiddenSize).astype(defaultDType) if Wo is None else Wo;

        self._initSelfParams([NetParamDefinition("weightQ", self._weightQ),
                              NetParamDefinition("weightK", self._weightK),
                              NetParamDefinition("weightV", self._weightV),
                              NetParamDefinition("weightO", self._weightO)]);


    @property
    def attentionWeight(self) -> Optional[np.ndarray]:
        return self._attentionModule.attentionWeight;


    def _setSelfParams(self, params: List[INetParamDefinition]):
        self._weightQ, self._weightK, self._weightV, self._weightO = params[0].value, params[1].value, params[2].value , params[3].value;


    def _reshapeInput(self, X : np.ndarray) -> np.ndarray:
        Y = X.reshape(X.shape[: -1] + self._headShape);
        return np.swapaxes(Y, -2, -3);


    def _restoreInput(self, X : np.ndarray) -> np.ndarray:
        Y = np.swapaxes(X, -2, -3);
        return Y.reshape(Y.shape[: -2] + (-1, ));


    # Q shape: (batch_size, query_num, query_size)
    # K shape: (batch_size, pair_num, key_size)
    # V shape: (batch_size, pair_num, value_size)
    # M shape: (batch_size, query_num, pair_num)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        self._Q, self._K, self._V = data[: 3];
        M = np.repeat(np.expand_dims(data[3], axis = -3), self._headNum, axis = -3) if len(data) > 3 and data[3] is not None else None;  # softmax mask

        QH = self._Q @ self._weightQ;
        KH = self._K @ self._weightK;
        VH = self._V @ self._weightV;

        QH = self._reshapeInput(QH);
        KH = self._reshapeInput(KH);
        VH = self._reshapeInput(VH);

        self._C, = self._attentionModule.forward(QH, KH, VH, M); # type: ignore
        self._C = np.swapaxes(self._C, -2, -3);
        self._C = self._C.reshape(self._C.shape[: -2] + (-1, ));
        Y = self._C @ self._weightO;

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dC = dY @ self._weightO.T;
        dWeightO = np.sum(np.swapaxes(self._C, -1, -2) @ dY, axis = tuple(range(self._C.ndim - 2)));
        dC = np.swapaxes(dC.reshape(self._C.shape[: -1] + self._headShape), -2, -3);
        dQH, dKH, dVH = self._attentionModule.backward(dC);

        dQH = self._restoreInput(dQH);
        dKH = self._restoreInput(dKH);
        dVH = self._restoreInput(dVH);

        dQ = dQH @ self._weightQ.T;
        dWeightQ = np.sum(np.swapaxes(self._Q, -1, -2) @ dQH, axis = tuple(range(self._Q.ndim - 2)));
        dK = dKH @ self._weightK.T;
        dWeightK = np.sum(np.swapaxes(self._K, -1, -2) @ dKH, axis = tuple(range(self._K.ndim - 2)));
        dV = dVH @ self._weightV.T;
        dWeightV = np.sum(np.swapaxes(self._V, -1, -2) @ dVH, axis = tuple(range(self._V.ndim - 2)));

        self._params[0].grad[...] += dWeightQ;
        self._params[1].grad[...] += dWeightK;
        self._params[2].grad[...] += dWeightV;
        self._params[3].grad[...] += dWeightO;

        return dQ, dK, dV;


class SelfAttentionModule(AggregateNetModule, INetAttentionModule):
    def __init__(self, attentionModule : INetAttentionModule):
        self._attentionModule = attentionModule;
        super().__init__(self._attentionModule);

        self._name = f"SelfAttention({attentionModule})";


    @property
    def attentionWeight(self) -> Optional[np.ndarray]:
        return self._attentionModule.attentionWeight;


    # X shape: (batch_size, sequence_length, sequence_dimension)
    # M shape: (batch_size, sequence_length, sequence_length)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        M = data[1] if len(data) > 1 else None;  # softmax mask

        Y, = self._attentionModule.forward(X, X, X, M); # type: ignore

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dQ, dK, dV = self._attentionModule.backward(dY);
        dX = dQ + dK + dV;

        return dX, ;


class SinePositionalEncodingModule(AggregateNetModule):
    def __init__(self, dimensionSize : int, maxLength : int = 10000, dropoutRatio : float = 0.0):
        self._dropoutLayer = DropoutLayer(dropoutRatio);
        super().__init__(self._dropoutLayer);

        self._encoding = self._createEncoding(dimensionSize, maxLength);
        self._name = "SinePositionEncoding";


    @property
    def positionalEncoding(self) -> np.ndarray:
        return self._encoding;


    def _createEncoding(self, dimensionSize : int, maxLength : int) -> np.ndarray:
        m = dimensionSize if dimensionSize % 2 == 0 else dimensionSize + 1;
        idx = np.arange(maxLength, dtype = np.float32).reshape(-1, 1) / np.power(10000, np.arange(0, m, 2, dtype = np.float32) / dimensionSize).reshape(1, -1);
        E = np.zeros((maxLength, m));
        E[:, 0::2] = np.sin(idx);
        E[:, 1::2] = np.cos(idx);

        return E[:, : dimensionSize].astype(defaultDType);


    # the dimension -2 is position
    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        startIndex = data[1] if len(data) > 1 else 0;

        X = X + self._encoding[startIndex: startIndex + X.shape[-2], :];
        Y, = self._dropoutLayer.forward(X);

        return Y, ;


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dX, = self._dropoutLayer.backward(dY);

        return dX, ;


class TransformerAddNormalizationModule(AggregateNetModule):
    def __init__(self, normalizedShape : Union[int, Tuple[int, ...]], dropoutRatio : float = 0.0):
        self._dropoutLayer = DropoutLayer(dropoutRatio);
        self._normalLayer = LayerNormalizationLayer(normalizedShape);

        super().__init__(self._dropoutLayer, self._normalLayer);
        self._name = "TransformerAddNormalization";


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X, F = data[: 2];

        Y1, = self._dropoutLayer.forward(F);
        Y2 = Y1 + X;
        Y, = self._normalLayer.forward(Y2);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dY2, = self._normalLayer.backward(dY);
        dF, = self._dropoutLayer.backward(dY2);
        dX = dY2;

        return dX, dF;


class TransformerPositionwiseFFNModule(AggregateNetModule):
    def __init__(self, inputSize : int, hiddenSize : int, activationFuncSelector : Optional[Callable[[int], INetModule]] = None):
        self._activationFuncSelector = activationFuncSelector if activationFuncSelector is not None else (lambda size: ReluLayer());

        super().__init__(
            AffineLayer(inputSize, hiddenSize),
            self._activationFuncSelector(hiddenSize),
            AffineLayer(hiddenSize, inputSize),
        );
        self._name = f"TransformerPositionwiseFFN({self._name})";


class TransformerEncoderBlock(AggregateNetModule, INetAttentionModule):
    def __init__(self, inputSize : int, attentionHiddenSize : int, ffnHiddenSize : int, normalizedShape : Union[int, Tuple[int, ...]], headNum : int = 2, dropoutRatio : float = 0.0, ffnActivationFuncSelector : Optional[Callable[[int], INetModule]] = None):
        self._attentionModule = SelfAttentionModule(MultiHeadAttentionModule(DotProductAttentionModule(dropoutRatio = dropoutRatio), inputSize, inputSize, inputSize, (attentionHiddenSize, attentionHiddenSize, attentionHiddenSize, inputSize), headNum = headNum));
        self._addNormal1 = TransformerAddNormalizationModule(normalizedShape, dropoutRatio = dropoutRatio);
        self._positionwiseFFN = TransformerPositionwiseFFNModule(inputSize, ffnHiddenSize, activationFuncSelector = ffnActivationFuncSelector);
        self._addNormal2 = TransformerAddNormalizationModule(normalizedShape, dropoutRatio = dropoutRatio);

        super().__init__(self._attentionModule, self._addNormal1, self._positionwiseFFN, self._addNormal2);
        self._name = f"TransformerEncoderBlock({self._attentionModule}, {self._positionwiseFFN})";


    @property
    def attentionWeight(self) -> Optional[np.ndarray]:
        return self._attentionModule.attentionWeight;


    # X shape: (batch_size, sequence_length, sequence_dimension)
    # M shape: (batch_size, sequence_length, sequence_length)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        M = data[1] if len(data) > 1 else None;  # softmax mask

        F1, = self._attentionModule.forward(X, M); # type: ignore
        Y1, = self._addNormal1.forward(X, F1);
        F2, = self._positionwiseFFN.forward(Y1);
        Y, = self._addNormal2.forward(Y1, F2);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dY1, dF2 = self._addNormal2.backward(dY);
        dY1 += self._positionwiseFFN.backward(dF2)[0];
        dX, dF1 = self._addNormal1.backward(dY1);
        dX += self._attentionModule.backward(dF1)[0];

        return dX, ;


class TransformerEncoder(AggregateNetModule, INetAttentionModule):
    def __init__(self, inputSize: int, attentionHiddenSize: int, ffnHiddenSize: int, normalizedShape: Union[int, Tuple[int, ...]], headNum: int = 2, blockNum : int = 2, maxSequenceLength : int = 10000, dropoutRatio: float = 0.0, ffnActivationFuncSelector : Optional[Callable[[int], INetModule]] = None):
        if blockNum < 1:
            raise ValueError("blockNum < 1");
        
        self._attentionWeight = None;
        self._positionalEncoding = SinePositionalEncodingModule(inputSize, maxLength = maxSequenceLength, dropoutRatio = dropoutRatio);
        self._blocks = [TransformerEncoderBlock(inputSize, attentionHiddenSize, ffnHiddenSize, normalizedShape, headNum = headNum, dropoutRatio = dropoutRatio, ffnActivationFuncSelector = ffnActivationFuncSelector) for _ in range(blockNum)];

        super().__init__(*tuple([self._positionalEncoding] + self._blocks));
        self._name = f"TransformerEncoder({blockNum} * ({self._blocks[0]}))";


    @property
    def attentionWeight(self) -> Optional[np.ndarray]:
        return self._attentionWeight;


    # X shape: (batch_size, sequence_length, sequence_dimension)
    # M shape: (batch_size, sequence_length, sequence_length)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        M = data[1] if len(data) > 1 else None;  # softmax mask

        PX, = self._positionalEncoding.forward(X);

        Y = PX;
        for block in self._blocks:
            Y, = block.forward(Y, M); # type: ignore

        self._attentionWeight = np.array([item.attentionWeight for item in self._blocks]);
        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dPX, = super().backward(dY);
        dX, = self._positionalEncoding.backward(dPX);

        return dX, ;


class TransformerEmbeddingEncoder(AggregateNetModule, INetAttentionModule):
    def __init__(self, embeddingNum : int, embeddingSize : int, attentionHiddenSize: int, ffnHiddenSize: int, normalizedShape: Union[int, Tuple[int, ...]], headNum: int = 2, blockNum : int = 2, maxSequenceLength : int = 10000, dropoutRatio: float = 0.0, ffnActivationFuncSelector : Optional[Callable[[int], INetModule]] = None):
        self._embeddingScale = math.sqrt(embeddingSize);
        self._embedding = EmbeddingLayer(embeddingNum, embeddingSize);
        self._encoder = TransformerEncoder(embeddingSize, attentionHiddenSize, ffnHiddenSize, normalizedShape, headNum = headNum, blockNum = blockNum, maxSequenceLength = maxSequenceLength, dropoutRatio = dropoutRatio, ffnActivationFuncSelector = ffnActivationFuncSelector);

        super().__init__(self._embedding, self._encoder);
        self._name = f"TransformerEmbeddingEncoder({self._embedding}, {self._encoder})";


    @property
    def attentionWeight(self) -> Optional[np.ndarray]:
        return self._encoder.attentionWeight;


    # X shape: (batch_size, sequence_length)
    # validLength shape: (batch_size)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        validLength = data[1] if len(data) > 1 else None;

        sequenceLength = X.shape[-1];
        M = getAttentionMaskByValidLength(sequenceLength, sequenceLength, validLength, onlyBatch = True) if validLength is not None else None;

        EX, = self._embedding.forward(X);
        Y, = self._encoder.forward(EX * self._embeddingScale, M); # type: ignore

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dEX, = self._encoder.backward(dY);
        dEX *= self._embeddingScale;
        dX, = self._embedding.backward(dEX);

        return dX, ;


class TransformerDecoderBlock(AggregateNetModule, INetAttentionModule):
    def __init__(self, inputSize : int, encoderSize : int, attentionHiddenSize : int, ffnHiddenSize : int, normalizedShape : Union[int, Tuple[int, ...]], headNum : int = 2, dropoutRatio : float = 0.0, ffnActivationFuncSelector : Optional[Callable[[int], INetModule]] = None):
        self._innerAttentionModule = MultiHeadAttentionModule(DotProductAttentionModule(dropoutRatio = dropoutRatio), inputSize, inputSize, inputSize, (attentionHiddenSize, attentionHiddenSize, attentionHiddenSize, inputSize), headNum = headNum);
        self._addNormal1 = TransformerAddNormalizationModule(normalizedShape, dropoutRatio = dropoutRatio);
        self._crossAttentionModule = MultiHeadAttentionModule(DotProductAttentionModule(dropoutRatio = dropoutRatio), inputSize, encoderSize, encoderSize, (attentionHiddenSize, attentionHiddenSize, attentionHiddenSize, inputSize), headNum = headNum);
        self._addNormal2 = TransformerAddNormalizationModule(normalizedShape, dropoutRatio = dropoutRatio);
        self._positionwiseFFN = TransformerPositionwiseFFNModule(inputSize, ffnHiddenSize, activationFuncSelector = ffnActivationFuncSelector);
        self._addNormal3 = TransformerAddNormalizationModule(normalizedShape, dropoutRatio = dropoutRatio);

        super().__init__(self._innerAttentionModule, self._addNormal1, self._crossAttentionModule, self._addNormal2, self._positionwiseFFN, self._addNormal3);
        self._name = f"TransformerDecoderBlock({self._innerAttentionModule}, {self._crossAttentionModule}, {self._positionwiseFFN})";


    @property
    def attentionWeight(self) -> Optional[np.ndarray]:
        return self._crossAttentionModule.attentionWeight;


    # Q shape: (batch_size, query_num, query_size)
    # K shape: (batch_size, pair_num, key_size)
    # V shape: (batch_size, pair_num, value_size)
    # encoderY shape: (batch_size, sequence_length, sequence_dimension)
    # encoderM shape: (batch_size, query_num, sequence_length)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        Q, K, V, encoderY = data[: 4];
        encoderM = data[4] if len(data) > 4 else None;  # softmax mask

        if self.context.isTrainingMode:
            queryNum, keyNum = Q.shape[-2], K.shape[-2];
            validLength = np.tile(np.arange(1, queryNum + 1), Q.shape[: -2] + (1, ));
            decoderM = getAttentionMaskByValidLength(queryNum, keyNum, validLength);
        else:
            decoderM = None;

        F1, = self._innerAttentionModule.forward(Q, K, V, decoderM); # type: ignore
        Y1, = self._addNormal1.forward(Q, F1);
        F2, = self._crossAttentionModule.forward(Y1, encoderY, encoderY, encoderM); # type: ignore
        Y2, = self._addNormal2.forward(Y1, F2);
        F3, = self._positionwiseFFN.forward(Y2);
        Y, = self._addNormal3.forward(Y2, F3);

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dY2, dF3 = self._addNormal3.backward(dY);
        dY2 += self._positionwiseFFN.backward(dF3)[0];

        dY1, dF2 = self._addNormal2.backward(dY2);
        dY1_, dEncoderY, dEncoderY_ = self._crossAttentionModule.backward(dF2);
        dY1 += dY1_;
        dEncoderY += dEncoderY_;

        dQ, dF1 = self._addNormal1.backward(dY1);
        dQ_, dK, dV = self._innerAttentionModule.backward(dF1);
        dQ += dQ_;

        return dQ, dK, dV, dEncoderY;


class TransformerDecoder(AggregateNetModule, INetAttentionModule):
    def __init__(self, inputSize: int, encoderSize : int, attentionHiddenSize: int, ffnHiddenSize: int, normalizedShape: Union[int, Tuple[int, ...]], headNum: int = 2, blockNum : int = 2, maxSequenceLength : int = 10000, dropoutRatio: float = 0.0, ffnActivationFuncSelector : Optional[Callable[[int], INetModule]] = None):
        if blockNum < 1:
            raise ValueError("blockNum < 1");

        self._blockNum = blockNum;
        self._attentionWeight = None;
        self._positionalEncoding = SinePositionalEncodingModule(inputSize, maxLength = maxSequenceLength, dropoutRatio = dropoutRatio);
        self._blocks = [TransformerDecoderBlock(inputSize, encoderSize, attentionHiddenSize, ffnHiddenSize, normalizedShape, headNum = headNum, dropoutRatio = dropoutRatio, ffnActivationFuncSelector = ffnActivationFuncSelector) for _ in range(blockNum)];

        super().__init__(*tuple([self._positionalEncoding] + self._blocks));
        self._name = f"TransformerDecoder({blockNum} * ({self._blocks[0]}))";


    @property
    def attentionWeight(self) -> Optional[np.ndarray]:
        return self._attentionWeight;


    # X shape: (batch_size, sequence_length, sequence_dimension)
    # encoderM shape: (batch_size, sequence_length, encoder_length)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X, encoderY = data[: 2];
        encoderM = data[2] if len(data) > 2 else None;  # softmax mask

        Y, = self._positionalEncoding.forward(X);
        for block in self._blocks:
            Y, = block.forward(Y, Y, Y, encoderY, encoderM); # type: ignore

        self._attentionWeight = np.array([item.attentionWeight for item in self._blocks]);
        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dEncoderY = None;
        for block in reversed(self._blocks):
            dQ, dK, dV, dEncoderY_ = block.backward(dY);
            dY = dQ + dK + dV;
            if dEncoderY is None:
                dEncoderY = dEncoderY_;
            else:
                dEncoderY += dEncoderY_;

        dX, = self._positionalEncoding.backward(dY);

        return dX, dEncoderY; # type: ignore


    def predict(self, *data : np.ndarray, blockInputs : List[Optional[np.ndarray]]) -> Tuple[np.ndarray, ...]:
        X, encoderY = data[: 2];
        encoderM = data[2] if len(data) > 2 else None;  # softmax mask

        startIndex = np.array(blockInputs[0].shape[-2] if blockInputs[0] is not None else 0);
        Y, = self._positionalEncoding.forward(X, startIndex);

        for i in range(len(self._blocks)):
            K = blockInputs[i];
            if K is None:
                K = Y;
            else:
                K = np.concatenate((K, Y), axis = -2);
            blockInputs[i] = K;

            Y, = self._blocks[i].forward(Y, K, K, encoderY, encoderM); # type: ignore

        return Y, ;


class TransformerEmbeddingDecoder(AggregateNetModule, INetAttentionModule):
    def __init__(self, embeddingNum : int, embeddingSize : int, encoderSize : int, attentionHiddenSize: int, ffnHiddenSize: int, normalizedShape: Union[int, Tuple[int, ...]], headNum: int = 2, blockNum : int = 2, maxSequenceLength : int = 10000, dropoutRatio: float = 0.0, ffnActivationFuncSelector : Optional[Callable[[int], INetModule]] = None):
        self._embeddingScale = math.sqrt(embeddingSize);
        self._embedding = EmbeddingLayer(embeddingNum, embeddingSize);
        self._decoder = TransformerDecoder(embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, normalizedShape, headNum = headNum, blockNum = blockNum, maxSequenceLength = maxSequenceLength, dropoutRatio = dropoutRatio, ffnActivationFuncSelector = ffnActivationFuncSelector);

        super().__init__(self._embedding, self._decoder);
        self._name = f"TransformerEmbeddingDecoder({self._embedding}, {self._decoder})";


    @property
    def attentionWeight(self) -> Optional[np.ndarray]:
        return self._decoder.attentionWeight;


    # X shape: (batch_size, sequence_length)
    # encoderY shape: (batch_size, sequence_length, encoder_size)
    # encoderValidLength shape: (batch_size)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X, encoderY = data[: 2];
        encoderValidLength = data[2] if len(data) > 2 else None;

        decoderSequenceLength = X.shape[-1];
        encoderSequenceLength = encoderY.shape[-2];
        encoderM = getAttentionMaskByValidLength(decoderSequenceLength, encoderSequenceLength, encoderValidLength, onlyBatch = True) if encoderValidLength is not None else None;

        EX, = self._embedding.forward(X);
        Y, = self._decoder.forward(EX * self._embeddingScale, encoderY, encoderM); # type: ignore

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        dEX, dEncoderY = self._decoder.backward(dY);
        dEX *= self._embeddingScale;
        dX, = self._embedding.backward(dEX);

        return dX, dEncoderY;


    def predict(self, *data : np.ndarray, blockInputs : List[Optional[np.ndarray]]) -> Tuple[np.ndarray, ...]:
        X, encoderY = data[: 2];
        encoderValidLength = data[2] if len(data) > 2 else None;

        decoderSequenceLength = X.shape[-1];
        encoderSequenceLength = encoderY.shape[-2];
        encoderM = getAttentionMaskByValidLength(decoderSequenceLength, encoderSequenceLength, encoderValidLength, onlyBatch = True) if encoderValidLength is not None else None;

        EX, = self._embedding.forward(X);
        return self._decoder.predict(EX * self._embeddingScale, encoderY, encoderM, blockInputs = blockInputs); # type: ignore


# encode sequence to vector
class AttentionPoolingLayer(AggregateNetModule, INetAttentionModule):
    def __init__(self, inputSize : int, hiddenSize : Optional[Union[int, Tuple[int, ...], List[int]]] = None, activationFuncSelector : Optional[Callable[[int], INetModule]] = None):
        self._X = None;
        self._attentionWeight, self._innerWeight = np.empty(0), None;
        self._activationFuncSelector = activationFuncSelector if activationFuncSelector is not None else (lambda size: TanhLayer());

        modules = [];
        lastHiddenSize = inputSize;

        if hiddenSize is not None:
            if isinstance(hiddenSize, int):
                modules.append(AffineLayer(inputSize, hiddenSize));
                modules.append(self._activationFuncSelector(hiddenSize));
                lastHiddenSize = hiddenSize;
            else:
                preHiddenSize = inputSize;
                for nextSize in hiddenSize:
                    modules.append(AffineLayer(preHiddenSize, nextSize));
                    modules.append(self._activationFuncSelector(nextSize));
                    preHiddenSize = nextSize;
                lastHiddenSize = hiddenSize[-1];
        modules.append(AffineLayer(lastHiddenSize, 1, includeBias = False));

        super().__init__(*tuple(modules));

        if hiddenSize is None:
            self._name = f"AttentionPoolingLayer {inputSize}*1";
        elif isinstance(hiddenSize, int):
            self._name = f"AttentionPoolingLayer {inputSize}*{hiddenSize}*1";
        else:
            self._name = f"AttentionPoolingLayer {'*'.join([str(item) for item in [inputSize] + list(hiddenSize) + [1]])}";


    @property
    def attentionWeight(self) -> Optional[np.ndarray]:
        return self._attentionWeight;


    # X shape: (batch_size, sequence_length, sequence_dimension)
    # M shape: (batch_size, sequence_length)
    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        M = data[1] if len(data) > 1 else None;  # softmax mask

        attentionScore, = super().forward(X);
        attentionScore = np.squeeze(attentionScore, axis = -1);

        attentionWeight = softmax(attentionScore, M);
        innerWeight = np.expand_dims(attentionWeight, axis = -1);
        Y = np.sum(innerWeight * X, axis = -2);

        self._X = X;
        self._attentionWeight = attentionWeight;
        self._innerWeight = innerWeight;
        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dY = np.expand_dims(dY, axis = -2);

        dAttentionWeight = np.sum(dY * self._X, axis = -1);
        dAttentionScore = np.expand_dims(softmaxGradient(self._attentionWeight, dAttentionWeight), axis = -1);
        dX, = super().backward(dAttentionScore);
        dX += dY * self._innerWeight;

        return dX, ;


# # select value by weights for 1 time step
# class SelectByWeight1TModule(NetModuleBase):
#     def __init__(self):
#         super().__init__();

#         self._V = None;
#         self._W = None;
#         self._name = "SelectByWeight1T";


#     def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
#         # W: weights, N1 ... Nm * T1, V: values, N1 ...Nm * T1 * V
#         W, V = data;
#         self._V, self._W = V, np.expand_dims(W, axis = -1);
#         Y = np.sum(self._V * self._W, axis = -2);

#         return Y, ;


#     def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         dY = np.repeat(np.expand_dims(dout[0], axis = -2), self._V.shape[-2], axis = -2);
#         dV = dY * self._W;
#         dW = np.sum(dY * self._V, axis = -1);

#         return dW, dV;


# # select value by weights for N time step
# class SelectByWeightNTModule(SelectByWeight1TModule):
#     def __init__(self):
#         super().__init__();

#         self._name = "SelectByWeightNT";


#     def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
#         # W: weights, N1 ... Nm * T2 * T1, V: values, N1 ...Nm * T1 * D
#         W, V = data;
#         self._V, self._W = np.expand_dims(V, axis = -3), np.expand_dims(W, axis = -1);
#         Y = np.sum(self._V * self._W, axis = -2);

#         return Y, ;


#     def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         dY = np.repeat(np.expand_dims(dout[0], axis = -2), self._V.shape[-2], axis = -2);
#         dV = np.sum(dY * self._W, axis = -3);
#         dW = np.sum(dY * self._V, axis = -1);

#         return dW, dV;


# # additive attention weight for 1 time step
# class AdditiveAttentionWeight1TModule(AggregateNetModule):
#     def __init__(self, querySize : int, keySize : int, hiddenSize : int, Wq : Optional[np.ndarray] = None, Wk : Optional[np.ndarray] = None, wv : Optional[np.ndarray] = None):
#         self._qLayer = AffineLayer(querySize, hiddenSize, includeBias = False, W = Wq);
#         self._kLayer = AffineLayer(keySize, hiddenSize, includeBias = False, W = Wk);
#         self._vLayer = AffineLayer(hiddenSize, 1, includeBias = False, W = wv);
#         self._softmax = SoftmaxLayer();

#         super().__init__(self._qLayer, self._kLayer, self._vLayer, self._softmax);

#         self._H = None;
#         self._name = "AdditiveAttentionWeight1T";


#     def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
#         # Q: queries, N1 ... Nm * Q, K: keys, N1 ...Nm * T1 * K
#         Q, K = data;

#         QY = self._qLayer.forward(Q)[0];
#         KY = self._kLayer.forward(K)[0];

#         QY = np.expand_dims(QY, axis = -2);
#         self._H = tanh(QY + KY);
#         S = self._vLayer.forward(self._H)[0];
#         S = np.squeeze(S, axis = -1);
#         Y = self._softmax.forward(S)[0];

#         return Y, ;


#     def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         dS = self._softmax.backward(*dout)[0];
#         dS = np.expand_dims(dS, axis = -1);
#         dH = self._vLayer.backward(dS)[0];
#         dH = dH * tanhGradient(self._H);
#         dQ = self._qLayer.backward(np.sum(dH, axis = -2))[0];
#         dK = self._kLayer.backward(dH)[0];

#         return dQ, dK;


# # additive attention weight for N time step
# class AdditiveAttentionWeightNTModule(AdditiveAttentionWeight1TModule):
#     def __init__(self, querySize : int, keySize : int, hiddenSize : int, Wq : Optional[np.ndarray] = None, Wk : Optional[np.ndarray] = None, wv : Optional[np.ndarray] = None):
#         super().__init__(querySize, keySize, hiddenSize, Wq, Wk, wv);

#         self._name = "AdditiveAttentionWeightNT";


#     def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
#         # Q: queries, N1 ... Nm * T2 * Q, K: keys, N1 ...Nm * T1 * K
#         Q, K = data;

#         QY = self._qLayer.forward(Q)[0];
#         KY = self._kLayer.forward(K)[0];

#         QY = np.expand_dims(QY, axis = -2);
#         KY = np.expand_dims(KY, axis = -3);
#         self._H = tanh(QY + KY);
#         S = self._vLayer.forward(self._H)[0];
#         S = np.squeeze(S, axis = -1);
#         Y = self._softmax.forward(S)[0];

#         return Y, ;


#     def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         dS = self._softmax.backward(*dout)[0];
#         dS = np.expand_dims(dS, axis = -1);
#         dH = self._vLayer.backward(dS)[0];
#         dH = dH * tanhGradient(self._H);
#         dQ = self._qLayer.backward(np.sum(dH, axis = -2))[0];
#         dK = self._kLayer.backward(np.sum(dH, axis = -3))[0];

#         return dQ, dK;


# # dot-product attention weight for 1 time step
# class DotProductAttentionWeight1TModule(NetModuleBase):
#     def __init__(self):
#         super().__init__();

#         self._K = None;
#         self._Q = None;
#         self._softmax = SoftmaxLayer();
#         self._name = "DotProductAttentionWeight1T";


#     def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
#         # Q: queries, N1 ... Nm * D, K: keys, N1 ...Nm * T1 * D
#         Q, K = data;
#         self._K, self._Q = K, np.expand_dims(Q, axis = -2);
#         Y = self._softmax.forward(np.sum(self._K * self._Q, axis = -1) / math.sqrt(Q.shape[-1]))[0];

#         return Y, ;


#     def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         dY = self._softmax.backward(*dout)[0] / math.sqrt(self._Q.shape[-1]);
#         dY = np.repeat(np.expand_dims(dY, axis = -1), self._Q.shape[-1], axis = -1);
#         dK = dY * self._Q;
#         dQ = np.sum(dY * self._K, axis = -2);

#         return dQ, dK;


# # dot-product attention weight for N time step
# class DotProductAttentionWeightNTModule(DotProductAttentionWeight1TModule):
#     def __init__(self):
#         super().__init__();

#         self._name = "DotProductAttentionWeightNT";


#     def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
#         # Q: queries, N1 ... Nm * T2 * D, K: keys N1 ...Nm * T1 * D
#         Q, K = data;
#         self._K, self._Q = np.expand_dims(K, axis = -3), np.expand_dims(Q, axis = -2);
#         Y = self._softmax.forward(np.sum(self._K * self._Q, axis = -1) / math.sqrt(Q.shape[-1]))[0];

#         return Y, ;


#     def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         dY = self._softmax.backward(*dout)[0] / math.sqrt(self._Q.shape[-1]);
#         dY = np.repeat(np.expand_dims(dY, axis = -1), self._Q.shape[-1], axis = -1);
#         dK = np.sum(dY * self._Q, axis = -3);
#         dQ = np.sum(dY * self._K, axis = -2);

#         return dQ, dK;


# class QKVAttentionLayer(AggregateNetModule):
#     def __init__(self, weightModule : INetModule, selectModule : INetModule):
#         super().__init__(weightModule, selectModule);

#         self._attentionWeight = None;
#         self._weightModule = weightModule;
#         self._selectModule = selectModule;
#         self._name = "QKVAttention";


#     @property
#     def attentionWeight(self) -> np.ndarray:
#         return self._attentionWeight;


#     def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
#         Q, K, V = data;
#         self._attentionWeight = self._weightModule.forward(Q, K)[0];
#         Y = self._selectModule.forward(self._attentionWeight, V)[0];

#         return Y, ;


#     def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         dW, dV = self._selectModule.backward(*dout);
#         dQ, dK = self._weightModule.backward(dW);

#         return dQ, dK, dV;


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


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X1, X2 = data;

        if self._p == 1:
            self._select1 = True;
        elif self._p == 0:
            self._select1 = False;
        else:
            self._select1 = float(np.random.rand(1)[0]) <= self._p;

        return (X1 if self._select1 else X2), ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];

        return (dY, 0) if self._select1 else (0, dY); # type: ignore


# # Note: the module must have no inner connection in each step!
# class RepeatedWrapper(NetModuleBase):
#     def __init__(self, target : INetModule):
#         super().__init__();

#         self._modules = [];
#         self._target = target;
#         self._stepIndex = 0;
#         self._name = f"Repeated{str(self._target)}";

#         self._params.extend(target.params);
#         self._grads.extend(target.grads);


#     def _setContext(self, context : INetContext):
#         self._target.context = context;

#         for m in self._modules:
#             m.context = value;


#     def _setParams(self, value: List[np.ndarray]):
#         self._target.params = value;
#         for m in self._modules:
#             m.params = value;


#     def _reset(self):
#         self._stepIndex = 0;
#         self._target.reset();
#         for m in self._modules:
#             m.reset();


#     def _copyMembers(self, module : INetModule, shareParams : bool):
#         raise NotImplementedError();


#     def forward(self, *data : np.ndarray) -> Tuple[np.ndarray]:
#         while len(self._modules) < self._stepIndex + 1:
#             self._modules.append(self._target.copy(True));

#         Y = self._modules[self._stepIndex].forward(*data);
#         self._stepIndex += 1;

#         return Y;


#     def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray]:
#         self._stepIndex -= 1;

#         module = self._modules[self._stepIndex];
#         dX = module.backward(*dout);

#         for i in range(len(self._grads)):
#             self._grads[i] += module.grads[i];

#         return dX;


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
        self._totalIterations = int(math.ceil(data[0].shape[1] / self._stepSize));

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


class MaeAccuracyEvaluator(INetAccuracyEvaluator):
    def __init__(self, scaler : Optional[IDataScaler] = None):
        self._rss = 0.0;
        self._totalCount = 0.0;
        self._scaler = scaler;


    @property
    def name(self) -> str:
        return "MAE";


    @property
    def high(self) -> bool:
        return False;


    @property
    def accuracy(self) -> float:
        # return math.sqrt(self._rss / self._totalCount) if self._totalCount > 0 else None;
        return (self._rss / self._totalCount) if self._totalCount > 0 else 0.0;


    def fromLoss(self, lossValues : Optional[List[float]] = None) -> bool:
        return False;


    def update(self, loss : float, *data: np.ndarray):
        if len(data) > 2:
            Y, W, T = data;
        else:
            Y, T = data; # type: ignore
        
        if self._scaler is not None:
            Y, T = self._scaler.inverse(Y), self._scaler.inverse(T);

        self._rss += float(np.sum(np.abs(Y - T)));
        self._totalCount += T.size;


    def reset(self):
        self._rss = 0.0;
        self._totalCount = 0.0;


class MseAccuracyEvaluator(INetAccuracyEvaluator):
    def __init__(self, takeRoot : bool = False, takeLog : bool = False, logMinValue : Optional[float] = None):
        self._takeRoot = takeRoot;
        self._takeLog = takeLog;
        self._logMinValue = logMinValue;

        self._rss = 0.0;
        self._totalCount = 0.0;


    @property
    def name(self) -> str:
        if self._takeRoot and self._takeLog:
            return "LOG_RMSE";
        elif self._takeRoot and not self._takeLog:
            return "RMSE";
        elif not self._takeRoot and self._takeLog:
            return "LOG_MSE";
        else:
            return "MSE";


    @property
    def high(self) -> bool:
        return False;


    @property
    def accuracy(self) -> float:
        result = (self._rss / self._totalCount) if self._totalCount > 0 else 0.0;

        if self._takeRoot:
            result = math.sqrt(result);

        return result;


    def fromLoss(self, lossValues : Optional[List[float]] = None) -> bool:
        return False;


    def update(self, loss : float, *data: np.ndarray):
        if len(data) > 2:
            Y, W, T = data;
        else:
            Y, T = data; # type: ignore

        if self._takeLog:
            if self._logMinValue is not None:
                Y = np.maximum(Y, self._logMinValue);

            Y = np.log(Y);
            T = np.log(T);

        self._rss += float(np.sum(np.square(Y - T)));
        self._totalCount += T.size;


    def reset(self):
        self._rss = 0.0;
        self._totalCount = 0.0;


class MapeAccuracyEvaluator(INetAccuracyEvaluator):
    def __init__(self, scaler : Optional[IDataScaler] = None):
        self._rss = 0.0;
        self._totalCount = 0.0;
        self._scaler = scaler;


    @property
    def name(self) -> str:
        return "MAPE";


    @property
    def high(self) -> bool:
        return False;


    @property
    def accuracy(self) -> float:
        return (self._rss / self._totalCount) * 100 if self._totalCount > 0 else 0.0;


    def fromLoss(self, lossValues : Optional[List[float]] = None) -> bool:
        return False;


    def update(self, loss : float, *data: np.ndarray):
        if len(data) > 2:
            Y, W, T = data;
        else:
            Y, T = data; # type: ignore
        
        if self._scaler is not None:
            Y, T = self._scaler.inverse(Y), self._scaler.inverse(T);

        self._rss += float(np.sum(np.abs((T - Y) / T)));
        self._totalCount += T.size;


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
    def accuracy(self) -> float:
        return self._rightCount / self._totalCount if self._totalCount > 0 else 0.0;


    def fromLoss(self, lossValues : Optional[List[float]] = None) -> bool:
        return False;


    def update(self, loss : float, *data: np.ndarray):
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
        self._perplexity = 0.0;


    @property
    def name(self) -> str:
        return "Perplexity";


    @property
    def high(self) -> bool:
        return False;


    @property
    def accuracy(self) -> float:
        return self._perplexity;


    def fromLoss(self, lossValues : Optional[List[float]] = None) -> bool:
        if lossValues is None:
            return False;

        self._perplexity = math.exp(sum(lossValues) / len(lossValues));
        return True;


    def update(self, loss : float, *data: np.ndarray):
        pass;


    def reset(self):
        self._perplexity = 0.0;


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


    def _setContext(self, context : INetContext):
        self._encoder.context = context;
        self._decoder.context = context;


    def getFinalTag(self, T: np.ndarray) -> Optional[np.ndarray]:
        return None;


    def encode(self, X : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M, V = tuple(np.split(self._encoder.forward(X)[0], 2, axis = -1));

        return M, softplus(V) + self._minStd;


    def reparameterize(self, mu : np.ndarray, sigma : np.ndarray, epsilon : Optional[np.ndarray] = None) -> np.ndarray:
        N, L, H = len(mu), self._sampleSize, self._latentSize;
        self._z0 = np.random.randn(N, L, H).astype(defaultDType) if epsilon is None else epsilon;
        Z = self._z0 * np.expand_dims(sigma, axis = 1) + np.expand_dims(mu, axis = 1);

        return Z;


    def decode(self, Z : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E, U = tuple(np.split(self._decoder.forward(Z)[0], 2, axis = -1));

        return E, softplus(U) + self._minStd;


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X, epsilon = data if len(data) > 1 else (data[0], None);
        M, V = self.encode(X);
        Z = self.reparameterize(M, V, epsilon = epsilon);
        E, U = self.decode(Z);

        self._V, self._U = V, U;

        return X, M, V, E, U;


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
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
    

    @property
    def name(self) -> str:
        return "GaussianVAE Loss";


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


    def backward(self) -> Tuple[np.ndarray, ...]:
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


    def _setContext(self, context : INetContext):
        self._encoder.context = context;
        self._decoder.context = context;


    def getFinalTag(self, T: np.ndarray) -> Optional[np.ndarray]:
        return None;


    def encode(self, X : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M, V = tuple(np.split(self._encoder.forward(X)[0], 2, axis = -1));

        return M, softplus(V) + self._minStd;


    def reparameterize(self, mu: np.ndarray, sigma: np.ndarray, epsilon: Optional[np.ndarray] = None) -> np.ndarray:
        N, L, H = len(mu), self._sampleSize, self._latentSize;
        self._z0 = np.random.randn(N, L, H).astype(defaultDType) if epsilon is None else epsilon;
        Z = self._z0 * np.expand_dims(sigma, axis = 1) + np.expand_dims(mu, axis = 1);

        return Z;


    def decode(self, Z : np.ndarray, toProbability : bool = False) -> np.ndarray:
        Y = self._decoder.forward(Z)[0];
        if toProbability:
            Y = sigmoid(Y);

        return Y;


    def forward(self, *data: np.ndarray) -> Tuple[np.ndarray, ...]:
        X, epsilon = data if len(data) > 1 else (data[0], None);
        M, V = self.encode(X);
        Z = self.reparameterize(M, V, epsilon = epsilon);
        Y = self.decode(Z);

        self._V = V;

        return X, M, V, Y;


    def backward(self, *dout: np.ndarray) -> Tuple[np.ndarray, ...]:
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
    

    @property
    def name(self) -> str:
        return "BernoulliVAE Loss";


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


    def backward(self) -> Tuple[np.ndarray, ...]:
        N, L = self._Y.shape[: 2];

        dX = np.sum(-self._Y, axis = 1) / (N * L);
        dM = self._M / N;
        dV = (self._V - 1 / self._V) / N;
        dY = (sigmoid(self._Y) - self._X) / (N * L);

        return dX, dM, dV, dY;
