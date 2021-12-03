from ImportNumpy import *;
from typing import Callable, Tuple;


def sigmoid(X : np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X));


# Y = sigmoid(X)
def sigmoidGradient(Y : np.ndarray) -> np.ndarray:
    return Y * (1.0 - Y);


def tanh(X : np.ndarray) -> np.ndarray:
    return 2.0 * sigmoid(2.0 * X) - 1.0;


# Y = tanh(X)
def tanhGradient(Y : np.ndarray) -> np.ndarray:
    return 1.0 - Y ** 2;


def relu(X : np.ndarray) -> np.ndarray:
    return np.maximum(0, X);


def reluGradient(X : np.ndarray) -> np.ndarray:
    Y = np.zeros_like(X);
    Y[X > 0] = 1;

    return Y;


def softmax(X : np.ndarray) -> np.ndarray:
    Y = np.exp(X - np.amax(X, -1, keepdims = True));

    return Y / np.sum(Y, -1, keepdims = True);


def lengthExceptLastDimension(X : np.ndarray):
    return X.size // X.shape[-1] if X.ndim > 2 else len(X);


def meanSquareError(Y, T):
    return float(np.sum(np.square(Y - T))) / (2 * lengthExceptLastDimension(T));


# Y and T has the same shape
def crossEntropyError(Y : np.ndarray, T : np.ndarray, epsilon : float = 1e-8) -> float:
    return float(np.sum(-(T * np.log(Y + epsilon)))) / lengthExceptLastDimension(T);


# T was used as a 1-D index array
def crossEntropyError1D(Y : np.ndarray, T : np.ndarray, epsilon : float = 1e-8) -> float:
    n = T.size;
    return float(np.sum(-np.log(Y.reshape((n, -1))[np.arange(n), T.flatten()] + epsilon))) / n;


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


def convert2OneHot(X : np.ndarray, size : int) -> np.ndarray:
    Y = np.zeros((X.size, size), dtype = X.dtype);
    Y[list(range(X.size)), X.flatten().tolist()] = 1;
    return Y.reshape(X.shape + (size,));


def npAddAt(X : np.ndarray, indices, Y : np.ndarray):
    if DeviceConfig.enableGPU:
        cpx.scatter_add(X, indices, Y);
    else:
        np.add.at(X, indices, Y);