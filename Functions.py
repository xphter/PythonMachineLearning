# -*- coding: utf-8 -*-
#
# Copyright (C) 2025 XphteR, Inc. All Rights Reserved
#
# @Time    : 2025-07-07
# @Author  : Du Peng
# @Email   : 278770518@qq.com
# @File    : Functions.py
######################################################


import enum;
import math;
import collections;

from ImportNumpy import *;
from typing import Callable, Union, Tuple, List, Optional;


@enum.unique
class LossReductionType(enum.Enum):
    No = 0x00;
    Mean = 0x10;
    Sum = 0x20;


# M is a 0-1 or True-False mask array which has the same shape of X
def putArrayMask(X : np.ndarray, M : np.ndarray, value: Union[int, float, np.ndarray]):
    X *= ~M.astype(bool);
    X += (M * value);
    return X;


def sigmoid(X : np.ndarray, threshold : float = -20) -> np.ndarray:
    ML = X < threshold;

    if np.any(ML):
        MH = ~ML;
        EX = np.exp(ML * X);

        YL = EX / (1 + EX);
        YH = 1 / (1 + np.exp(-X * MH));
        Y = ML * YL + MH * YH;
    else:
        Y = 1 / (1 + np.exp(-X));

    return Y;


# Y = sigmoid(X)
def sigmoidGradient(Y : np.ndarray) -> np.ndarray:
    return Y * (1.0 - Y);


def hardSigmoid(X : np.ndarray) -> np.ndarray:
    return np.maximum(np.minimum(0.25 * X + 0.5, 1.0), 0.0);


# Y = hardSigmoid(X)
def hardSigmoidGradient(Y : np.ndarray) -> np.ndarray:
    G = np.zeros_like(Y);
    G[np.logical_and(Y > 0, Y < 1)] = 0.25;

    return G;


def tanh(X : np.ndarray) -> np.ndarray:
    return 2.0 * sigmoid(2.0 * X) - 1.0;


# Y = tanh(X)
def tanhGradient(Y : np.ndarray) -> np.ndarray:
    return 1.0 - Y ** 2;


def hardTanh(X : np.ndarray) -> np.ndarray:
    return np.maximum(np.minimum(X, 1.0), -1.0);


# Y = hardTanh(X)
def hardTanhGradient(Y : np.ndarray) -> np.ndarray:
    G = np.zeros_like(Y);
    G[np.logical_and(Y > -1, Y < 1)] = 1;

    return G;


def relu(X : np.ndarray) -> np.ndarray:
    return np.maximum(0, X);


def reluGradient(X : np.ndarray) -> np.ndarray:
    return (X > 0).astype(X.dtype);


def prelu(X : np.ndarray, beta : Union[float, np.ndarray]) -> np.ndarray:
    return np.maximum(0, X) + beta * np.minimum(0, X);


def preluGradient(X : np.ndarray, beta : Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    MH = X > 0;
    ML = ~MH;

    dX = ML * beta + MH;
    dBeta = X * ML;

    return dX, dBeta;


# when x ≥ 20, log(1 + exp(x)) == x in numerical
def softplus(X : np.ndarray, threshold : float = 20) -> np.ndarray:
    MH = X > threshold;

    if np.any(MH):
        ML = ~MH;

        YL = np.log(1 + np.exp(X * ML));
        YH = X;
        Y = ML * YL + MH * YH;
    else:
        Y = np.log(1 + np.exp(X));

    return Y;


# Y = softplus(X), dX = 1 - exp(-Y) or dX = sigmoid(X)
def softplusGradient(X : np.ndarray, threshold : float = 20) -> np.ndarray:
    MH = X > threshold;

    if np.any(MH):
        ML = ~MH;
        dX = ML * sigmoid(X * ML) + MH;
    else:
        dX = sigmoid(X);

    return dX;


# return (sigmoid(beta * X), swish(X))
def swish(X : np.ndarray, beta : Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    S = sigmoid(beta * X);
    return S, X * S;


# Y = swish(X, beta), S = sigmoid(beta * X)
# return dX, dBeta
def swishGradient(Y : np.ndarray, S : np.ndarray, X : np.ndarray, beta : Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    dX = S + beta * Y * (1 - S);
    dBeta = (X - Y) * Y;

    return dX, dBeta;


# M is a 0-1 or True-False mask array which has the same shape of X
def softmax(X : np.ndarray, M : Optional[np.ndarray] = None) -> np.ndarray:
    if M is not None and X.shape != M.shape:
        raise ValueError("the shape of X and M are not same.");

    if M is not None:
        putArrayMask(X, ~M.astype(bool), -1e10);

    Y = np.exp(X - np.amax(X, -1, keepdims = True));

    if M is not None:
        Y *= M;

    # the row of M can not be all zero!
    return Y / np.sum(Y, -1, keepdims = True);


# Y = softmax(X), dX = Y * (dY - ∑(dY * Y))
def softmaxGradient(Y : np.ndarray, dY : np.ndarray) -> np.ndarray:
    Z = dY * Y;
    dX = Z - Y * np.sum(Z, axis = -1, keepdims = True);

    return dX;


def lengthExceptLastDimension(X : np.ndarray):
    return X.size // X.shape[-1] if X.ndim > 2 else len(X);


def meanSquareError(Y : np.ndarray, T : np.ndarray):
    if Y.shape != T.shape:
        raise ValueError("the shapes is not same.");

    return float(np.sum(np.square(Y - T))) / T.size;


def meanAbsoluteError(Y : np.ndarray, T : np.ndarray):
    if Y.shape != T.shape:
        raise ValueError("the shapes is not same.");

    return float(np.sum(np.abs(Y - T))) / T.size;


# Y and T has the same shape
# M is a 0-1 or True-False mask array represents whether each samples in Y are valid
def crossEntropyError(Y : np.ndarray, T : np.ndarray, M : Optional[np.ndarray] = None, reductionType : LossReductionType = LossReductionType.Mean, epsilon : float = 1e-8) -> Union[float, np.ndarray]:
    if Y.shape != T.shape:
        raise ValueError("the shape of Y and T are not same.");

    L = -T * np.log(Y + epsilon);
    if M is not None:
        L *= np.expand_dims(M, axis = -1);

    if reductionType == LossReductionType.Sum:
        return float(np.sum(L));
    elif reductionType == LossReductionType.Mean:
        return float(np.sum(L)) / len(T);
    else:
        return L;


# Y and T has the same shape
# M is a 0-1 or True-False mask array represents whether each samples in Y are valid
def crossEntropyErrorGradient(Y : np.ndarray, T : np.ndarray, M : Optional[np.ndarray] = None, reductionType : LossReductionType = LossReductionType.Mean) -> np.ndarray:
    dY = -(T / Y).astype(Y.dtype);

    if M is not None:
        dY *= np.expand_dims(M, axis = -1);

    if reductionType == LossReductionType.Mean:
        dY /= len(T);

    return dY;


# Y and T has the same shape
# M is a 0-1 or True-False mask array represents whether each samples in Y are valid
def softmaxWithCrossEntropyErrorGradient(Y : np.ndarray, T : np.ndarray, M : Optional[np.ndarray] = None, reductionType : LossReductionType = LossReductionType.Mean) -> np.ndarray:
    dX = (Y - T).astype(Y.dtype);

    if M is not None:
        dX *= np.expand_dims(M, axis = -1);

    if reductionType == LossReductionType.Mean:
        dX /= len(T);

    return dX;


# Y and T has the same shape
# M is a 0-1 or True-False mask array represents whether each samples in Y are valid
def sigmoidWithCrossEntropyErrorGradient(Y : np.ndarray, T : np.ndarray, M : Optional[np.ndarray] = None, reductionType : LossReductionType = LossReductionType.Mean) -> np.ndarray:
    dX = (Y - T).astype(Y.dtype);

    if M is not None:
        dX *= M;

    if reductionType == LossReductionType.Mean:
        dX /= len(T);

    return dX;


# Y and T has the same shape
def logitsCrossEntropyError(Y : np.ndarray, T : np.ndarray) -> float:
    if Y.shape != T.shape:
        raise ValueError("the shapes is not same.");

    return float(np.sum(Y * (1 - T) + np.log(1 + np.exp(-Y)))) / lengthExceptLastDimension(T);


def huberError(Y : np.ndarray, T : np.ndarray, delta : Union[float, np.ndarray] = 1.0) -> float:
    if Y.shape != T.shape:
        raise ValueError("the shapes is not same.");

    delta = np.abs(delta).astype(Y.dtype);
    ML, MH = Y < T - delta, Y > T + delta;
    MM = np.logical_and(~ML, ~MH);

    b = delta * delta / 2;
    EL = delta * (T - Y) - b;
    EM = np.square(Y - T) / 2;
    EH = delta * (Y - T) - b;
    E = ML * EL + MM * EM + MH * EH;

    return float(np.sum(E)) / T.size;


def getDropoutMask(inputs : np.ndarray, dropoutRatio : float):
    if dropoutRatio == 0:
        return np.ones_like(inputs);
    if dropoutRatio == 1:
        return np.zeros_like(inputs);
    else:
        return (np.random.rand(*inputs.shape) > dropoutRatio).astype(inputs.dtype);


# T was a label index array
# M is a 0-1 or True-False mask array represents whether each samples in Y are valid
def crossEntropyError1D(Y : np.ndarray, T : np.ndarray, M : Optional[np.ndarray] = None, reductionType : LossReductionType = LossReductionType.Mean, epsilon : float = 1e-8) -> Union[float, np.ndarray]:
    n = T.size;
    L = -np.log(Y.reshape(-1)[np.arange(n) * Y.shape[-1] + T.flatten()] + epsilon).reshape(T.shape);

    if M is not None:
        L *= M;

    if reductionType == LossReductionType.Sum:
        return float(np.sum(L));
    elif reductionType == LossReductionType.Mean:
        return float(np.sum(L)) / n;
    else:
        return L;


# T was a label index array
# M is a 0-1 or True-False mask array represents whether each samples in Y are valid
def softmaxWithCrossEntropyErrorGradient1D(Y : np.ndarray, T : np.ndarray, M : Optional[np.ndarray] = None, reductionType : LossReductionType = LossReductionType.Mean) -> np.ndarray:
    n = T.size;
    dX = Y.flatten();
    dX[np.arange(n) * Y.shape[-1] + T.flatten()] -= 1;
    dX = dX.reshape(Y.shape);

    if M is not None:
        dX *= np.expand_dims(M, axis = -1);

    if reductionType == LossReductionType.Mean:
        dX /= n;

    return dX;


def labelSmoothing(T : np.ndarray, alpha : float = 0.1) -> np.ndarray:
    if len(T.shape) < 2:
        # T is a 1-D label vector
        N = len(T);
        D = int(np.amax(T)) + 1;

        if D < 2:
            return T;
        else:
            Y = alpha / (D - 1) * np.ones((N, D), dtype = defaultDType);
            Y[np.arange(N), T.flatten()] = 1 - alpha;
            return Y;
    else:
        # each row of T is a one-hot vector
        D = T.shape[-1];
        if D < 2:
            raise ValueError("invalid one-hot vector");

        Y = alpha / (D - 1) * np.ones_like(T, dtype = defaultDType);
        Y[T == 1] = 1 - alpha;
        return np.reshape(Y, T.shape);


def numericGradient(f : Callable, X : np.ndarray):
    h = 1e-4;
    grad = np.zeros_like(X);

    it = np.nditer(X, flags = ["multi_index"], op_flags = [["readwrite"]]);
    while not it.finished:
        index = it.multi_index;
        temp = X[index];

        X[index] = temp - h;
        value1 = f(X);
        X[index] = temp + h;
        value2 = f(X);

        grad[index] = (value2 - value1) / (2 * h);
        X[index] = temp;

        it.iternext();

    return grad;


def calcConvSamePadding(width : int, kernel : int, stride : int, dilation : int) -> int:
    return width * (stride - 1) + dilation * (kernel - 1) + 1 - stride;


def getConvSamePadding(width : int, kernel : int, stride : int, dilation : int) -> Tuple[int, int]:
    padding = calcConvSamePadding(width, kernel, stride, dilation);
    paddingLeft = padding // 2;
    paddingRight = padding - paddingLeft;

    return paddingLeft, paddingRight;


def getConvCausalPadding(width : int, kernel : int, stride : int, dilation : int) -> Tuple[int, int]:
    padding = calcConvSamePadding(width, kernel, stride, dilation);
    paddingLeft = dilation * (kernel - 1);
    paddingRight = padding - paddingLeft;

    return paddingLeft, paddingRight;


# X shape: batch_size, input_channel, sequence_length
def seq2col(X : np.ndarray, FW : int, stride : int = 1, padding : Union[Tuple[int, int], int] = 0, dilation : int = 1) -> Tuple[np.ndarray, int]:
    if isinstance(padding, int):
        padding = (padding, padding);
    padNumber = sum(padding);

    N, C, T = X.shape;
    kernelSize = dilation * (FW - 1) + 1;

    if (T + padNumber - kernelSize) % stride != 0:
        raise ValueError("the convolution core unable to cover all data");

    OT = (T + padNumber - kernelSize) // stride + 1;

    seq = X if padNumber <= 0 else np.pad(X, [(0, 0), (0, 0), (padding[0], padding[1])], "constant");
    col = np.zeros((N, C, FW, OT), dtype = X.dtype);

    for t in range(FW):
        j = t * dilation;
        tMax = j + stride * OT;

        col[:, :, t, :] = seq[:, :, j : tMax : stride];

    return col.transpose(0, 3, 1, 2).reshape(N * OT, -1), OT;


def col2seq(X : np.ndarray, seqShape : tuple, FW : int, stride : int = 1, padding : Union[Tuple[int, int], int] = 0, dilation : int = 1, inDiff : bool = False) -> np.ndarray:
    if isinstance(padding, int):
        padding = (padding, padding);
    padNumber = sum(padding);

    N, C, T = seqShape;
    kernelSize = dilation * (FW - 1) + 1;

    if (T + padNumber - kernelSize) % stride != 0:
        raise ValueError("the convolution core unable to cover all data");

    OT = (T + padNumber - kernelSize) // stride + 1;

    col = X.reshape(N, OT, C, FW).transpose(0, 2, 3, 1);
    seq = np.zeros((N, C, T + padNumber), dtype = X.dtype);

    for t in range(FW):
        j = t * dilation;
        tMax = j + stride * OT;

        if inDiff:
            seq[:, :, j: tMax: stride] += col[:, :, t, :];
        else:
            seq[:, :, j: tMax: stride] = col[:, :, t, :];

    return seq[:, :, padding[0]: T + padding[0]];


def parseStride2D(stride : Union[Tuple[int, int], int]) -> Tuple[int, int]:
    strideH, strideW = stride[: 2] if isinstance(stride, tuple) else (stride, stride);
    return strideH, strideW;


def parsePadding2D(padding : Union[Tuple[int, ...], int]) -> Tuple[int, int, int, int]:
    paddingTop, paddingBottom, paddingLeft, paddingRight  = ((padding[0], padding[0], padding[1], padding[1]) if len(padding) == 2 else padding[: 4]) if isinstance(padding, tuple) else (padding, padding, padding, padding);
    return paddingTop, paddingBottom, paddingLeft, paddingRight;


# X shape: (batch_size, input_channel, image_height, image_width)
def im2col(X : np.ndarray, FH : int, FW : int, stride : Union[Tuple[int, int], int] = 1, padding : Union[Tuple[int, ...], int] = 0, dilation : int = 1) -> Tuple[np.ndarray, int, int]:
    N, C, H, W = X.shape;
    strideH, strideW = parseStride2D(stride);
    paddingTop, paddingBottom, paddingLeft, paddingRight = parsePadding2D(padding);
    paddingH, paddingW = paddingTop + paddingBottom, paddingLeft + paddingRight;
    kernelHeight = dilation * (FH - 1) + 1;
    kernelWidth = dilation * (FW - 1) + 1;

    if (H + paddingH - kernelHeight) % strideH != 0 or (W + paddingW - kernelWidth) % strideW != 0:
        raise ValueError("the convolution kernel unable to cover all data");

    OH = (H + paddingH - kernelHeight) // strideH + 1;
    OW = (W + paddingW - kernelWidth) // strideW + 1;

    img = X if paddingH + paddingW  <= 0 else np.pad(X, [(0, 0), (0, 0), (paddingTop, paddingBottom), (paddingLeft, paddingRight)], "constant");
    col = np.zeros((N, C, FH, FW, OH, OW), dtype = X.dtype);

    for y in range(FH):
        i = y * dilation;
        yMax = i + strideH * OH;

        for x in range(FW):
            j = x * dilation;
            xMax = j + strideW * OW;

            col[:, :, y, x, :, :] = img[:, :, i: yMax: strideH, j: xMax: strideW];

    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1), OH, OW;


def col2im(X : np.ndarray, imShape : Tuple[int, ...], FH : int, FW : int, stride : Union[Tuple[int, int], int] = 1, padding : Union[Tuple[int, ...], int] = 0, dilation : int = 1, inDiff : bool = False) -> np.ndarray:
    N, C, H, W = imShape;
    strideH, strideW  = parseStride2D(stride);
    paddingTop, paddingBottom, paddingLeft, paddingRight = parsePadding2D(padding);
    paddingH, paddingW  = paddingTop + paddingBottom, paddingLeft + paddingRight;
    kernelHeight = dilation * (FH - 1) + 1;
    kernelWidth = dilation * (FW - 1) + 1;

    if (H + paddingH - kernelHeight) % strideH != 0 or (W + paddingW - kernelWidth) % strideW != 0:
        raise ValueError("the convolution kernel unable to cover all data");

    OH = (H + paddingH - kernelHeight) // strideH + 1;
    OW = (W + paddingW - kernelWidth) // strideW + 1;

    col = X.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2);
    img = np.zeros((N, C, H + paddingH, W + paddingW), dtype = X.dtype);

    for y in range(FH):
        i = y * dilation;
        yMax = i + strideH * OH;

        for x in range(FW):
            j = x * dilation;
            xMax = j + strideW * OW;

            if inDiff:
                img[:, :, i: yMax: strideH, j: xMax: strideW] += col[:, :, y, x, :, :];
            else:
                img[:, :, i: yMax: strideH, j: xMax: strideW] = col[:, :, y, x, :, :];

    return img[:, :, paddingTop: H + paddingTop, paddingLeft: W + paddingLeft];


# validLength shape: (batch_size) or (batch_size, query_num)
def getAttentionMaskByValidLength(queryNum: int, keyNum, validLength: np.ndarray, onlyBatch : bool = False) -> np.ndarray:
    if onlyBatch or len(validLength.shape) == 1:
        validLength = np.repeat(np.expand_dims(validLength, axis = -1), queryNum, axis = -1);
    validLength = np.expand_dims(validLength, axis = -1);

    return (np.arange(keyNum) < validLength).astype(np.int32);


# validLength shape: (batch_size)
# the mask was used on each sample loss
def getLossMaskByValidLength(maxLength : int, validLength: np.ndarray) -> np.ndarray:
    return np.arange(maxLength) < np.expand_dims(validLength, axis = -1);


def convOutputSize(inputSize : int, filterSize : int, stride : int = 1, pad : int = 0) -> int:
    return (inputSize + pad - filterSize) // stride + 1;


# expand elements of last axis to a one-hot vector
def expand2OneHot(X : np.ndarray, size : int, dtype = defaultDType) -> np.ndarray:
    Y = np.zeros((X.size, size), dtype = dtype);
    Y[list(range(X.size)), X.flatten().tolist()] = 1;
    return Y.reshape(X.shape + (size,));


# change list of last axis to a one-hot vector
def list2OneHot(X : List[List[int]], size : int, dtype = np.float64) -> np.ndarray:
    Y = np.zeros((len(X), size));
    for i, x in enumerate(X):
        Y[i, x] = 1;
    return Y;


def npAddAt(X : np.ndarray, indices, Y : np.ndarray):
    if DeviceConfig.enableGPU:
        cpx.scatter_add(X, indices, Y);
    else:
        np.add.at(X, indices, Y);


'''
PS is the scores of positive samples;
NS is the scores of negative samples;
'''
def auc(PS : np.ndarray, NS : np.ndarray):
    P, N = len(PS), len(NS);
    ranks = np.argsort(np.concatenate((NS, PS), axis = None), axis = None) + 1;
    return (float(np.sum(ranks[N:]))  - P * (P + 1) / 2) / (P * N);


# BLEU: bilingual evaluation understudy
# BLEU = exp(min{0, 1 - len(label) / len(prediction)}) * ∏ p_n^{1/2^n}
def bleuNLP(prediction : List[str], label : List[str], gramNum : int = 2) -> float:
    labelLength, predictionLength = len(label), len(prediction);
    if predictionLength == 0:
        return 0.0;

    score = math.exp(min(0.0, 1.0 - labelLength / predictionLength));
    for n in range(1, min(gramNum, predictionLength) + 1):
        matchCount, labelGrams = 0, collections.defaultdict(int);

        for i in range(labelLength - n + 1):
            labelGrams[" ".join(label[i: i + n])] += 1;

        for i in range(predictionLength - n + 1):
            gram = " ".join(prediction[i: i + n]);
            if labelGrams[gram] > 0:
                matchCount += 1;
                labelGrams[gram] -= 1;

        score *= math.pow(matchCount / (predictionLength - n + 1), math.pow(0.5, n));

    return score;
