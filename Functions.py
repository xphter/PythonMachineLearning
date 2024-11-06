import math;
from ImportNumpy import *;
from typing import Callable, Union, Tuple, List;


def sigmoid(X : np.ndarray, threshold : float = -20) -> np.ndarray:
    mask = X <= -20;

    if np.any(mask):
        Y1 = np.exp(mask * X);
        Y1 = Y1 / (1 + Y1);

        Y2 = 1 / (1 + np.exp(-(~mask * X)));

        Y = mask * Y1 + ~mask * Y2;
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
    Y = np.zeros_like(X);
    Y[X > 0] = 1;

    return Y;


# when x ≥ 20, log(1 + exp(x)) == x in numerical
def softplus(X : np.ndarray, threshold : float = 20) -> np.ndarray:
    mask = X >= threshold;

    if np.any(mask):
        Y = np.log(1 + np.exp(~mask * X));
        Y[mask] = X[mask];
    else:
        Y = np.log(1 + np.exp(X));

    return Y;


# Y = softplus(X)
def softplusGradient(Y : np.ndarray = None, X : np.ndarray = None) -> np.ndarray:
    if Y is not None:
        return 1 - np.exp(-Y);
    elif X is not None:
        return sigmoid(X);
    else:
        raise ValueError("both X and Y are None");


# return (sigmoid(beta * X), swish(X))
def swish(X : np.ndarray, beta : Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    S = sigmoid(beta * X);
    return S, X * S;


# Y = swish(X, beta), S = sigmoid(beta * X)
# return dX, dBeta
def swishGradient(Y : np.ndarray, S : np.ndarray, X : np.ndarray, beta : Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    dX = S + beta * Y * (1 - S);
    dBeta = Y * (X - Y);

    return dX, dBeta;


def softmax(X : np.ndarray) -> np.ndarray:
    Y = np.exp(X - np.amax(X, -1, keepdims = True));

    return Y / np.sum(Y, -1, keepdims = True);


def lengthExceptLastDimension(X : np.ndarray):
    return X.size // X.shape[-1] if X.ndim > 2 else len(X);


def meanSquareError(Y : np.ndarray, T : np.ndarray):
    if Y.shape != T.shape:
        raise ValueError("the shapes is not same.");

    return float(np.sum(np.square(Y - T))) / (2 * lengthExceptLastDimension(T));


def meanAbsoluteError(Y : np.ndarray, T : np.ndarray):
    if Y.shape != T.shape:
        raise ValueError("the shapes is not same.");

    return float(np.sum(np.abs(Y - T))) / lengthExceptLastDimension(T);


# Y and T has the same shape
def crossEntropyError(Y : np.ndarray, T : np.ndarray, epsilon : float = 1e-8) -> float:
    if Y.shape != T.shape:
        raise ValueError("the shapes is not same.");

    return float(np.sum(-(T * np.log(Y + epsilon)))) / lengthExceptLastDimension(T);


# Y and T has the same shape
def logitsCrossEntropyError(Y : np.ndarray, T : np.ndarray) -> float:
    if Y.shape != T.shape:
        raise ValueError("the shapes is not same.");

    return float(np.sum(Y * (1 - T) + np.log(1 + np.exp(-Y)))) / lengthExceptLastDimension(T);


def huberError(Y : np.ndarray, T : np.ndarray, delta : float = 1) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if Y.shape != T.shape:
        raise ValueError("the shapes is not same.");

    delta = math.fabs(delta);
    if delta == 0.0:
        raise ValueError("the delta of Huber loss can not be zero.");

    TL, TH = T - delta, T + delta;
    ML, MH = Y < TL, Y > TH;
    MM = np.logical_and(~ML, ~MH);
    b = delta * delta / 2;

    E = np.zeros_like(Y, dtype = Y.dtype);
    E[ML] = delta * (T[ML] - Y[ML]) - b;
    E[MH] = delta * (Y[MH] - T[MH]) - b;
    E[MM] = np.square(Y[MM] - T[MM]) / 2;

    return float(np.sum(E)) / lengthExceptLastDimension(T), ML, MM, MH;


def getDropoutMask(inputs : np.ndarray, dropoutRatio : float):
    if dropoutRatio == 0:
        return np.ones_like(inputs);
    if dropoutRatio == 1:
        return np.zeros_like(inputs);
    else:
        return (np.random.rand(*inputs.shape) > dropoutRatio).astype(inputs.dtype);


# T was used as a 1-D index array
def crossEntropyError1D(Y : np.ndarray, T : np.ndarray, epsilon : float = 1e-8) -> float:
    n = T.size;
    return float(np.sum(-np.log(Y.reshape((n, -1))[np.arange(n), T.flatten()] + epsilon))) / n;


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

    it = np.nditer(X, flags = ["multi_index"], op_flags = ["readwrite"]);
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


def im2col(X : np.ndarray, FH : int, FW : int, stride : int = 1, pad : int = 0) -> np.ndarray:
    N, C, H, W = X.shape;

    if (H + 2 * pad - FH) % stride != 0 or (W + 2 * pad - FW) % stride != 0:
        raise ValueError("the convolution core unable to cover all data");

    OH = (H + 2 * pad - FH) // stride + 1;
    OW = (W + 2 * pad - FW) // stride + 1;

    img = X if pad <= 0 else np.pad(X,[(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant");
    col = np.zeros((N, C, FH, FW, OH, OW), dtype = X.dtype);

    for y in range(FH):
        yMax = y + stride * OH;

        for x in range(FW):
            xMax = x + stride * OW;

            col[:, :, y, x, :, :] = img[:, :, y: yMax: stride, x: xMax: stride];

    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1);


def col2im(X : np.ndarray, imShape : tuple, FH : int, FW : int, stride : int = 1, pad : int = 0, inDiff : bool = False) -> np.ndarray:
    N, C, H, W = imShape;

    if (H + 2 * pad - FH) % stride != 0 or (W + 2 * pad - FW) % stride != 0:
        raise ValueError("the convolution core unable to cover all data");

    OH = (H + 2 * pad - FH) // stride + 1;
    OW = (W + 2 * pad - FW) // stride + 1;

    col = X.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2);
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype = X.dtype);

    for y in range(FH):
        yMax = y + stride * OH;

        for x in range(FW):
            xMax = x + stride * OW;

            if inDiff:
                img[:, :, y: yMax: stride, x: xMax: stride] += col[:, :, y, x, :, :];
            else:
                img[:, :, y: yMax: stride, x: xMax: stride] = col[:, :, y, x, :, :];

    return img[:, :, pad: H + pad, pad: W + pad];


def convOutputSize(inputSize : int, filterSize : int, stride : int = 1, pad : int = 0) -> int:
    return (inputSize + 2 * pad - filterSize) // stride + 1;


# expand elements of last axis to a one-hot vector
def expand2OneHot(X : np.ndarray, size : int, dtype = np.float64) -> np.ndarray:
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
