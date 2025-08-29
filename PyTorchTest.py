import multiprocessing;
import re;
import math;
import os;
import os.path
import pickle
import time;
import datetime;
import random;
import itertools;
import collections;
from typing import Any, Optional, List, Tuple, Dict;

import numpy as np;
import pandas as pd;
import scipy;
import scipy.stats;
import matplotlib;
import matplotlib.pyplot as plt
import torch;
import DeviceConfig;

# DeviceConfig.enableGPU = True;
# DeviceConfig.floatLength = 64;

import torch.nn.functional as F;
import torchvision;

from Functions import *;
import NN;
import MNIST;

from torch.utils import data;
from torch import nn;
# from d2l import torch as d2l;
# from IPython import display;

matplotlib.use("Qt5Agg");


class Timer:
    def __init__(self):
        self._times = [];
        self._running = False;
        self._startTIme = 0.0;


    @property
    def times(self) -> List[float]:
        return self._times.copy();


    def start(self):
        if self._running:
            return;

        self._running = True;
        self._startTIme = time.time();


    def stop(self):
        if not self._running:
            return;

        self._times.append(time.time() - self._startTIme);
        self._running = False;


def testModuleGradient(m : NN.INetModule, title: str, *data : np.ndarray):
    numGradients = [];

    for p in m.params:
        v = p.value;
        numGradients.append(numericGradient(lambda x: sum([float(np.sum(item)) for item in m.copy(True).forward(*data)]), v));

    message = '\n'.join([f'param {m.params[i].name}{i}{m.params[i].value.shape} error value: {np.sum(np.fabs(m.params[i].grad - numGradients[i]))}, error ratio: {np.linalg.norm(m.params[i].grad - numGradients[i]) / (np.linalg.norm(m.params[i].grad) + np.linalg.norm(numGradients[i]))}' for i in range(len(m.params))]);
    print(f"{title}\n{message}");


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def tryGPU(i=0):
    # if torch.cuda.device_count() >= i + 1:
    #     return torch.device(f"cuda:{i}");

    return torch.device('cpu');


def tryAllGPUs():
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())];

    return devices if len(devices) > 0 else [torch.device("cpu")];


def accuracy_num(yHat, y):
    return (yHat.argmax(-1) == y).sum();


def accuracy_mae(yHat, y) -> float:
    return float(torch.sum(torch.abs(yHat - y)));


def accuracy_mse(yHat, y) -> float:
    return float(torch.sum(torch.square(yHat - y)));


def accuracy_log_mse(yHat, y) -> float:
    return float(torch.sum(torch.square(torch.log(torch.clamp(yHat, 1, None)) - torch.log(y))));


def evaluate_accuracy(net, dataIter, lossFunc, accuracyFunc, device = None):
    # if device is None:
    #     device = tryGPU();

    if isinstance(net, torch.nn.Module):
        net.eval();

    metric = Accumulator(3);
    for X, y in dataIter:
        # X = X.to(device);
        # y = y.to(device);
        yHat = net(X);
        loss = lossFunc(yHat, y);

        metric.add(loss * len(y), accuracyFunc(yHat, y), len(y));

    return metric[0] / metric[2], metric[1] / metric[2];


# def model(X, W1, b1, W2, b2):
#     X = X.reshape(X.shape[0], -1);
#     Y1 = X @ W1 + b1;
#     A1 = relu(Y1);
#     Y2 = A1 @ W2 + b2;
#     return Y2;


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot'];
    return [text_labels[int(i)] for i in labels];


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten();
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def trainEpoch(net, trainIter, lossFunc, optimizer, lr, accuracyFunc, device = None):
    # if device is None:
    #     device = tryGPU();

    if isinstance(net, torch.nn.Module):
        net.train();

    metric = Accumulator(3);
    for X, y in trainIter:
        # X = X.to(device);
        # y = y.to(device);

        yHat = net(X);
        loss = lossFunc(yHat, y);

        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            with torch.no_grad():
                metric.add(float(loss) * len(y), accuracyFunc(yHat, y), len(y))
        else:
            totalLoss = loss.sum();
            totalLoss.backward();
            optimizer(lr, len(y));

            with torch.no_grad():
                metric.add(totalLoss, accuracyFunc(yHat, y), len(y));

    return metric[0] / metric[2], metric[1] / metric[2];


def train(net, trainIter, testIter, lossFunc, optimizer, lr, epochNum, accuracyFunc, device = None, plot : bool = True) -> NN.INetFitResult:
    # if device is None:
    #     device = tryGPU();
    #
    # net.to(device);

    trainLossData, trainAccuracyData, testLossData, testAccuracyData = [], [], [], [];
    # animator = Animator(xlabel = 'epoch', xlim = [1, epochNum], ylim = [0.3, 0.9], legend = ['train loss', 'train acc', 'test acc']);
    for epoch in range(epochNum):
        train_loss, train_acc = trainEpoch(net, trainIter, lossFunc, optimizer, lr, accuracyFunc, device);
        # train_loss, train_acc = evaluate_accuracy(net, trainIter, lossFunc, accuracyFunc, device);
        test_loss, test_acc = evaluate_accuracy(net, testIter, lossFunc, accuracyFunc, device);

        trainLossData.append(train_loss);
        trainAccuracyData.append(train_acc);
        testLossData.append(test_loss);
        testAccuracyData.append(test_acc);

        print(f"epoch: {epoch}, train loss: {train_loss}, train accuracy: {train_acc}, test loss: {test_loss}, test accuracy: {test_acc}");
        # animator.add(epoch + 1, train_metrics + (test_acc,));
    print("train completed.\n\n");

    if plot:
        fig = plt.figure(1);

        ax1 = fig.add_subplot(111);
        ax1.set_xlabel("epoch");
        ax1.set_ylabel('loss');
        ax1.plot(trainLossData, "o-k", label = "training loss");
        ax1.plot(testLossData, "o-b", label = "test loss");

        ax2 = ax1.twinx();
        ax2.set_ylabel('accuracy');
        ax2.plot(trainAccuracyData, "D-m", label = "training accuracy");
        ax2.plot(testAccuracyData, "D-r", label = "test accuracy");

        fig.legend(loc = "upper left", bbox_to_anchor = (0, 1), bbox_transform = ax1.transAxes)
        plt.show(block = True);
        plt.close();

    return NN.NetFitResult(trainLossData, trainAccuracyData, testLossData, testAccuracyData);


class Accumulator():
    def __init__(self, n):
        self._data = [0.0] * n;


    def add(self, *args):
        self._data = [a + float(b) for a, b in zip(self._data, args)];


    def reset(self):
        self._data = [0.0] * len(self._data);


    def __getitem__(self, item):
        return self._data[item];


class Animator:
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def initWeights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        # nn.init.normal_(m.weight, std = 0.01);
        # torch.nn.init.xavier_normal(m.weight, 0, 0.01);
        # torch.nn.init.xavier_uniform_(m.weight);
        torch.nn.init.kaiming_normal_(m.weight);
        # torch.nn.init.zeros_(m.bias);


class CenteredLayer(torch.nn.Module):
    def forward(self, X):
        return X - X.mean();


    def backward(self, dout):
        return dout - dout.sum() / dout.numel() * torch.ones_like(dout);


def corr2D(X, K):
    H, W = X.shape;
    FH, FW = K.shape;
    OH, OW = H - FH + 1, W - FW + 1;
    Y = torch.zeros(OH, OW);

    for i in range(OH):
        for j in range(OW):
            Y[i, j] = (X[i:i + FH, j:j + FW] * K).sum();

    return Y;


# K: C * FH * FW
def corr2DMultiIn(X, K):
    return sum([corr2D(x, k) for x, k in zip(X, K)]);


# K : FN * C * FH * FW
def corr2DMultiInOut(X, K):
    return torch.stack([corr2DMultiIn(X, k) for k in K], 0);


def corr2DMultiInOut1x1(X, K):
    C, H, W = X.shape;
    FN = K.shape[0];

    K = K.reshape(FN, -1);
    X = X.reshape(C, -1);

    Y = K @ X;
    return Y.reshape(FN, H, W);


def pooling2D(X, PH, PW, mode = "max"):
    H, W = X.shape;
    OH, OW = H - PH + 1, W - PW + 1;
    Y = torch.zeros(OH, OW);

    for i in range(OH):
        for j in range(OW):
            if mode == "max":
                Y[i, j] = X[i:i + PH, j:j + PW].max();
            else:
                Y[i, j] = X[i:i + PH, j:j + PW].mean();

    return Y;


class Conv2D(torch.nn.Module):
    def __init__(self, filterSize):
        super().__init__();
        self.weight = torch.nn.Parameter(torch.randn(filterSize));
        self.bias = torch.nn.Parameter(torch.zeros(1));


    def forward(self, X):
        return corr2D(X, self.weight) + self.bias;


def comp_conv(conv2d, X):
    X = X.reshape((1, 1) + X.shape);
    Y = conv2d(X);
    return Y.reshape(Y.shape[2:]);


def test():
    X = torch.rand(size = (1, 1, 28, 28), dtype = torch.float32);
    net = torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, (5, 5), padding = (2, 2)),
        torch.nn.Sigmoid(),
        torch.nn.AvgPool2d((2, 2), stride = 2),
        torch.nn.Conv2d(6, 16, (5, 5)),
        torch.nn.Sigmoid(),
        torch.nn.AvgPool2d((2, 2), stride = 2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 5 * 5, 120),
        torch.nn.Sigmoid(),
        torch.nn.Linear(120, 84),
        torch.nn.Sigmoid(),
        torch.nn.Linear(84, 10),
    );

    for layer in net:
        X = layer(X);
        print(f"{layer.__class__.__name__}, output shape: {X.shape}");

    batchSize = 256;
    epochNum, lr = 100, 0.5;
    trainIter, testIter = loadFashionMNIST(batchSize);

    # inputNum, hiddenNum, outputNum = 28 ** 2, 2 ** 20, 10;
    # # W1 = torch.nn.Parameter(torch.randn(inputNum, hiddenNum, requires_grad = True, device = tryGPU(1)) * 0.01);
    # # b1 = torch.nn.Parameter(torch.zeros(hiddenNum, requires_grad = True, device = tryGPU(1)));
    # # W2 = torch.nn.Parameter(torch.randn(hiddenNum, outputNum, requires_grad = True, device = tryGPU(1)) * 0.01);
    # # b2 = torch.nn.Parameter(torch.zeros(outputNum, requires_grad = True, device = tryGPU(1)));
    # # train(lambda X: model(X, W1, b1, W2, b2), trainIter, testIter, nn.CrossEntropyLoss(), torch.optim.SGD([W1, b1, W2, b2], lr), lr, epochNum);
    #
    # net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(inputNum, hiddenNum), torch.nn.ReLU(), torch.nn.Linear(hiddenNum, outputNum));
    net.apply(initWeights);
    train(net, trainIter, testIter, torch.nn.CrossEntropyLoss(), torch.optim.SGD(net.parameters(), lr), lr, epochNum);
    #
    # input("press enter to exit");
    print("exit");


# def synthetic_data(w, b, n):
#     X = torch.normal(0, 1, (n, len(w)));
#     Y = X @ w + b;
#     Y += torch.normal(0, 0.01, Y.shape);
#     return X, Y;


def synthetic_Data(fillName: str, w : np.ndarray, b : float, dataSize = 200, sigma : float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    filePth = f"./data/PyTorchTest/{fillName}";
    if os.path.isfile(filePth):
        with open(filePth, "rb") as file:
            X, y = pickle.load(file);
    else:
        X = np.random.randn(dataSize, len(w));
        y = X @ np.reshape(w, (-1, 1)) + b;
        y += np.random.randn(*y.shape) * sigma;

        with open(filePth, "wb") as file:
            pickle.dump((X, y), file);

    return X, y;


def getLoaderWorks():
    return 0;


def loadFashionMNIST(batchSize, resize = None) -> Tuple[data.DataLoader, data.DataLoader]:
    trans = [torchvision.transforms.ToTensor()];
    if resize is not None:
        trans.insert(0, torchvision.transforms.Resize(resize));
    trans = torchvision.transforms.Compose(trans);

    mnistTrain = torchvision.datasets.FashionMNIST("data", train=True, transform=trans, download=False);
    mnistTest = torchvision.datasets.FashionMNIST("data", train=False, transform=trans, download=False);

    return data.DataLoader(mnistTrain, batchSize, shuffle=True, num_workers=getLoaderWorks()), \
           data.DataLoader(mnistTest, batchSize, shuffle=True, num_workers=getLoaderWorks());


def loadKaggleHousePrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trainSet = pd.read_csv("/media/WindowsE/Data/Kaggle/House Prices - Advanced Regression Techniques/train.csv", na_values = "NA");
    testSet = pd.read_csv("/media/WindowsE/Data/Kaggle/House Prices - Advanced Regression Techniques/test.csv", na_values = "NA");

    allFeatures = pd.concat((trainSet.iloc[:, 1: -1], testSet.iloc[:, 1: ]), axis = 0);
    numericFeatureNames = allFeatures.dtypes[allFeatures.dtypes != "object"].index;
    allFeatures[numericFeatureNames] = allFeatures[numericFeatureNames].apply(lambda x: (x - x.mean()) / x.std(), axis = 0);
    allFeatures[numericFeatureNames] = allFeatures[numericFeatureNames].fillna(0);
    allFeatures = pd.get_dummies(allFeatures, dummy_na = True, dtype = np.int);

    trainSize = len(trainSet);
    return allFeatures.iloc[: trainSize].to_numpy().astype(defaultDType),\
           trainSet.iloc[: trainSize, -1].to_numpy().astype(defaultDType),\
           allFeatures.iloc[trainSize: ].to_numpy().astype(defaultDType),\
           testSet.iloc[:, 0].to_numpy();


def plotFitResult(filePath : str):
    with open(filePath, "rb") as file:
        result = pickle.load(file);

    NN.NetUtility.plotFitResult(result);


def chapter2_Preprocess():
    filePath = "data/PyTorchTest/house_tiny.csv";
    with open(filePath, "wt", encoding = "utf-8") as file:
        file.write("NumRooms,Alley,Proce\n");
        file.write("NaN,Pave,127000\n");
        file.write("2,NaN,106000\n");
        file.write("4,NaN,178100\n");
        file.write("NaN,NaN,140000\n");

    data = pd.read_csv(filePath);
    print(data);

    inputs, outputs = data.iloc[:, 0:1], data.iloc[:, -1];
    inputs = inputs.fillna(inputs.mean());
    print(inputs);

    inputs = pd.get_dummies(inputs, dummy_na = True);
    print(inputs);

    X, y = torch.tensor(inputs.values), torch.tensor(outputs.values);
    print(X);
    print(y);


def chapter3_LR_PyTorch():
    filePath = "data/PyTorchTest/LR.pkl";
    # w, b = torch.tensor([2, -3.4]), 4.2;
    # X, Y = synthetic_data(torch.reshape(w, (-1, 1)), 4.2, 1000);
    # with open(filePath, "wb") as file:
    #     pickle.dump((X.numpy(), Y.numpy()), file, protocol = pickle.DEFAULT_PROTOCOL);
    # return;

    with open(filePath, "rb") as file:
        features, labels = pickle.load(file);
    features, labels = torch.tensor(features), torch.tensor(labels);

    batchSize, maxEpoch = 10, 3;
    dataIterator = data.DataLoader(data.TensorDataset(features, labels), batch_size = batchSize, shuffle = True);
    lossFunc = nn.MSELoss();

    net = nn.Sequential(nn.Linear(2, 1));
    net[0].weight.data.normal_(0, 0.01);
    net[0].bias.data.fill_(0);
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.03);

    for epoch in range(maxEpoch):
        for X, Y in dataIterator:
            loss = lossFunc(net(X), Y);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        loss = lossFunc(net(features), labels);
        print(f"echo: {epoch}, loss: {loss}");
    print(f"weight: {net[0].weight.data}");
    print(f"bias: {net[0].bias.data}");


def chapter3_LR_My(plot : bool = True):
    filePath = "data/PyTorchTest/LR.pkl";
    with open(filePath, "rb") as file:
        features, labels = pickle.load(file);

    if plot:
        plt.figure(figsize = (12, 14));
        plt.subplot(1, 2, 1);
        plt.scatter(features[:, 0], labels.flatten());
        plt.subplot(1, 2, 2);
        plt.scatter(features[:, 1], labels.flatten());
        plt.show(block = True);

    batchSize, maxEpoch = 10, 3;
    trainIterator = NN.SequentialDataIterator([features, labels], batchSize = batchSize, shuffle = True);
    testIterator = NN.SequentialDataIterator([features, labels], batchSize = len(features), shuffle = False);
    lossFunc = NN.IdentityWithMeanSquareLoss();
    optimizer = NN.SGD(lr = 0.03);
    # optimizer = NN.Adam(lr = 0.03);
    evaluator = NN.MaeAccuracyEvaluator();

    model = NN.SequentialContainer(NN.AffineLayer(2, 1));
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = plot);

    print(f"weight: {model.modules[0].weight}");
    print(f"bias: {model.modules[0].bias}");


def chapter3_LR_MSE_MAE_Huber():
    start, end, size = 1, 40, 100;
    filePath = "data/PyTorchTest/MSE_MAE_Huber.npy";

    if not os.path.isfile(filePath):
        x = np.linspace(start, end, size);
        y = x + 4 + np.random.randn(len(x));
        y[-25:] -= 10;

        x = np.reshape(x, (len(x), 1));
        y = np.reshape(y, (len(y), 1));

        with open(filePath, "wb") as file:
            pickle.dump((x, y), file);
    else:
        with open(filePath, "rb") as file:
            x, y = pickle.load(file);
    x, y = x.astype(defaultDType), y.astype(defaultDType);

    batchSize, maxEpoch = 10, 100;
    trainIterator = NN.SequentialDataIterator([x, y], batchSize = batchSize, shuffle = True);
    testIterator = NN.SequentialDataIterator([x, y], batchSize = len(x), shuffle = False);
    # lossFunc = NN.IdentityWithMeanSquareLoss();
    # lossFunc = NN.IdentityWithMeanAbsoluteLoss();
    lossFunc = NN.IdentityWithHuberLoss(delta = 1);
    optimizer = NN.SGD(lr = 0.001);
    evaluator = NN.MaeAccuracyEvaluator();

    layer = NN.AffineLayer(1, 1);
    model = NN.SequentialContainer(layer);
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = True);
    print(f"weight: {layer.weight}, bias: {layer.bias}");

    x0 = np.array([start, end]).reshape((2, 1));
    y0 = model.forward(x0)[0];

    plt.figure();
    plt.scatter(x, y);
    plt.plot(x0.flatten(), y0.flatten(), color = "r");
    plt.show(block = True);


def chapter3_SoftmaxR_PyTorch(plot : bool = True):
    batchSize, lr, maxEpoch = 256, 0.1, 10;
    trainLoader, testLoader = loadFashionMNIST(256);
    lossFunc = nn.CrossEntropyLoss();

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10));
    net.apply(initWeights);
    optimizer = torch.optim.SGD(net.parameters(), lr = lr);

    # for epoch in range(maxEpoch):
    #     for X, Y in trainLoader:
    #         loss = lossFunc(net(X), Y);
    #         optimizer.zero_grad();
    #         loss.backward();
    #         optimizer.step();
    #     # loss = lossFunc(net(features), labels);
    #     print(f"echo: {epoch}, loss: {loss}");

    train(net, trainLoader, testLoader, lossFunc, optimizer, lr, maxEpoch, accuracy_num);


def chapter3_SoftmaxR_My(plot : bool = True):
    batchSize, maxEpoch = 256, 50;
    mnist = MNIST.MNIST("data/FashionMNIST/raw/");
    trainIterator = NN.SequentialDataIterator([mnist.trainX, mnist.trainY], batchSize = batchSize, shuffle = True);
    testIterator = NN.SequentialDataIterator([mnist.testX, mnist.testY], batchSize = batchSize, shuffle = True);
    lossFunc = NN.SoftmaxWithCrossEntropy1DLoss();
    # trainIterator = NN.SequentialDataIterator([mnist.trainX, labelSmoothing(mnist.trainY)], batchSize = batchSize, shuffle = True);
    # testIterator = NN.SequentialDataIterator([mnist.testX, labelSmoothing(mnist.testY)], batchSize = batchSize, shuffle = True);
    # lossFunc = NN.SoftmaxWithCrossEntropyLoss();
    optimizer = NN.SGD(lr = 0.1);
    evaluator = NN.ClassifierAccuracyEvaluator();

    model = NN.SequentialContainer(
        NN.FlattenLayer(),
        NN.AffineLayer(784, 10),
    );
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = plot);

    X, y = next(iter(testIterator));
    yHat, = model.predictOne(X);
    rightCount = int(np.sum(np.argmax(yHat, axis = -1) == y));
    print(f"{rightCount}, {rightCount / len(y)}\n\n");


def chapter3_SoftmaxR_My_SoftLabel(plot : bool = True):
    batchSize, maxEpoch = 256, 10;
    mnist = MNIST.MNIST("data/FashionMNIST/raw/");
    trainIterator = NN.SequentialDataIterator([mnist.trainX, labelSmoothing(mnist.trainY)], batchSize = batchSize, shuffle = True);
    testIterator = NN.SequentialDataIterator([mnist.testX, labelSmoothing(mnist.testY)], batchSize = batchSize, shuffle = True);
    lossFunc = NN.SoftmaxWithCrossEntropyLoss();
    optimizer = NN.SGD(lr = 0.1);
    evaluator = NN.ClassifierAccuracyEvaluator();

    model = NN.SequentialContainer(
        NN.FlattenLayer(),
        NN.AffineLayer(784, 10),
    );
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = plot);

    X, y = next(iter(testIterator));
    yHat, = model.predictOne(X);
    rightCount = int(np.sum(np.argmax(yHat, axis = -1) == np.argmax(y, axis = -1)));
    print(f"{rightCount}, {rightCount / len(y)}\n\n");


def chapter4_MLP_PyTorch(plot : bool = True):
    batchSize, lr, maxEpoch, hiddenSize = 256, 0.1, 10, 256;
    trainLoader, testLoader = loadFashionMNIST(batchSize);
    lossFunc = nn.CrossEntropyLoss();

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, hiddenSize), nn.ReLU(), nn.Linear(hiddenSize, 10));
    net.apply(initWeights);
    optimizer = torch.optim.SGD(net.parameters(), lr = lr);

    train(net, trainLoader, testLoader, lossFunc, optimizer, lr, maxEpoch, accuracy_num);


def chapter4_MLP_My(plot : bool = True):
    def getActivationFunc(size : int) -> NN.INetModule:
        # return NN.SigmoidLayer();
        # return NN.ReluLayer();
        # return NN.PReluLayer(None, size);
        # return NN.SoftplusLayer();
        # return NN.SwishLayer(None, size);
        return NN.MaxoutLayer(4);

    lr = 0.01;
    batchSize, maxEpoch = 256, 10;
    inputSize, outputSize = 28 * 28, 10;

    mnist = MNIST.MNIST("data/FashionMNIST/raw/");
    trainIterator = NN.SequentialDataIterator([mnist.trainX, mnist.trainY], batchSize = batchSize, shuffle = True);
    testIterator = NN.SequentialDataIterator([mnist.testX, mnist.testY], batchSize = batchSize, shuffle = True);
    lossFunc = NN.SoftmaxWithCrossEntropy1DLoss();
    # trainIterator = NN.SequentialDataIterator([mnist.trainX, labelSmoothing(mnist.trainY)], batchSize = batchSize, shuffle = True);
    # testIterator = NN.SequentialDataIterator([mnist.testX, labelSmoothing(mnist.testY)], batchSize = batchSize, shuffle = True);
    # lossFunc = NN.SoftmaxWithCrossEntropyLoss();
    # optimizer = NN.SGD(lr = lr);
    # optimizer = NN.SGDM(lr = lr);
    # optimizer = NN.AdaGrad(lr = lr);
    # optimizer = NN.RMSProp(lr = lr);
    # optimizer = NN.AdaDelta();
    # optimizer = NN.Adam(lr = lr);
    optimizer = NN.Adam(lr = lr, yogi = True);
    evaluator = NN.ClassifierAccuracyEvaluator();

    layersSize = [inputSize] + [256];
    modules = [NN.FlattenLayer()];
    for i in range(1, len(layersSize)):
        modules.append(NN.AffineLayer(layersSize[i - 1], layersSize[i] * 4));
        modules.append(getActivationFunc(layersSize[i]));
    modules.append(NN.AffineLayer(layersSize[-1], outputSize));

    model = NN.SequentialContainer(*tuple(modules));
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = plot);

    X, y = next(iter(testIterator));
    yHat, = model.predictOne(X);
    rightCount = int(np.sum(np.argmax(yHat, axis = -1) == y));
    print(f"{rightCount}, {rightCount / len(y)}\n\n");


def polynomial_Data(fillName: str, w : np.ndarray, b : float, dataSize = 200, sigma : float = 0.01, plot : bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    filePth = f"./data/PyTorchTest/{fillName}";
    if os.path.isfile(filePth):
        with open(filePth, "rb") as file:
            x, X, y = pickle.load(file);
    else:
        x = np.random.randn(dataSize) * 1;
        np.random.shuffle(x);

        X = np.power(np.reshape(x, (-1, 1)), np.arange(1, len(w) + 1));
        y = X @ np.reshape(w, (-1, 1)) + b;
        y += np.random.randn(*y.shape) * sigma;

        with open(filePth, "wb") as file:
            pickle.dump((x, X, y), file);

    if plot:
        plt.figure();
        plt.scatter(x, y.flatten());
        plt.show(block = True);

    return x, X, y;


def chapter4_PR_underfitting_overfitting(plot : bool = True):
    # y = 5.0 + 1.2 * x - 3.4 * x^2 / 2! + 5.6 * x^3 / 3! + z, z ~ N(0, 0.01^2)
    b = 5.0;
    w = np.zeros(20);
    w[: 3] = np.array([1.2, -3.4, 5.6]);
    w /= np.array([math.gamma(i + 2) for i in range(len(w))]);

    x, X, y = polynomial_Data("chapter4_PR_Data.pkl", w, b, plot = plot);
    trainSize, inputSize, outputSize = 100, 3, 1;

    dataSet = X[:, : inputSize];
    trainX = dataSet[: trainSize];
    testX = dataSet[trainSize:];
    trainY = y[: trainSize];
    testY = y[trainSize:];

    batchSize, maxEpoch = 10, 400;
    trainIterator = NN.SequentialDataIterator([trainX, trainY], batchSize = batchSize, shuffle = True);
    testIterator = NN.SequentialDataIterator([testX, testY], batchSize = batchSize, shuffle = False);
    lossFunc = NN.IdentityWithMeanSquareLoss();
    optimizer = NN.SGD(lr = 0.01);
    evaluator = NN.MaeAccuracyEvaluator();

    model = NN.SequentialContainer(NN.AffineLayer(inputSize, outputSize));
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = plot);
    print(f"weight: {model.modules[0].weight}, bias: {model.modules[0].bias}");

    model.isTrainingMode = False;
    yHat = model.forward(dataSet)[0];

    if plot:
        index = np.argsort(x);

        plt.figure();
        plt.scatter(x, y.flatten());
        plt.plot(x[index], yHat[index].flatten(), color = "r");
        plt.show(block = True);


def chapter4_Weight_Decay_PyTorch(plot : bool = True):
    trainSize, testSize, inputSize, outputSize = 20, 100, 200, 1;

    b = 0.05;
    w = np.ones(inputSize) * 0.01;
    X, y = synthetic_Data("chapter4_Weight_Decay_Data.pkl", w, b, dataSize = trainSize + testSize);
    X, y = torch.tensor(X, dtype = torch.float32), torch.tensor(y, dtype = torch.float32);

    dataSet = X;
    trainX = dataSet[: trainSize];
    testX = dataSet[trainSize:];
    trainY = y[: trainSize];
    testY = y[trainSize:];

    wd = 3.0;
    batchSize, lr, maxEpoch = 5, 0.003, 100;
    trainLoader = data.DataLoader(data.TensorDataset(trainX, trainY), batchSize, True);
    testLoader = data.DataLoader(data.TensorDataset(testX, testY), batchSize, False);
    lossFunc = nn.MSELoss();

    net = nn.Sequential(nn.Linear(inputSize, outputSize));
    net.apply(initWeights);
    optimizer = torch.optim.SGD([
        {"params":net[0].weight, 'weight_decay': wd},
        {"params":net[0].bias}], lr = lr);

    train(net, trainLoader, testLoader, lossFunc, optimizer, lr, maxEpoch, accuracy_mae);

    print(f"weight L2 normal: {torch.norm(net[0].weight)}");


def chapter4_Weight_Decay_My():
    trainSize, testSize, inputSize, outputSize = 20, 100, 200, 1;

    b = 0.05;
    w = np.ones(inputSize) * 0.01;
    X, y = synthetic_Data("chapter4_Weight_Decay_Data.pkl", w, b, dataSize = trainSize + testSize);

    dataSet = X;
    trainX = dataSet[: trainSize];
    testX = dataSet[trainSize:];
    trainY = y[: trainSize];
    testY = y[trainSize:];

    wd = 3.0;
    batchSize, maxEpoch = 5, 100;
    trainIterator = NN.SequentialDataIterator([trainX, trainY], batchSize = batchSize, shuffle = True);
    testIterator = NN.SequentialDataIterator([testX, testY], batchSize = batchSize, shuffle = False);
    lossFunc = NN.IdentityWithMeanSquareLoss();
    optimizer = NN.SGD(lr = 0.003);
    evaluator = NN.MaeAccuracyEvaluator();

    model = NN.SequentialContainer(NN.AffineLayer(inputSize, outputSize, weightHandler = NN.L2WeightDecay(wd)));
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = True);
    print(f"weight: {model.modules[0].weight}, bias: {model.modules[0].bias}");
    print(f"weight normal: {np.linalg.norm(model.modules[0].weight)}");


def chapter4_Dropout_PyTorch(plot : bool = True):
    lr = 0.5;
    batchSize, maxEpoch = 256, 10;
    dropout1, dropout2 = 0.0, 0.0;
    inputSize, hiddenSize1, hiddenSize2, outputSize = 28 * 28, 256, 256, 10;

    x = torch.Tensor([1]);
    m = (x > 1).float();
    np.array([1]).astype(np.int8)
    trainLoader, testLoader = loadFashionMNIST(batchSize);
    lossFunc = nn.CrossEntropyLoss();
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(inputSize, hiddenSize1),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(hiddenSize1, hiddenSize2),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(hiddenSize2, outputSize),
    );
    net.apply(initWeights);
    optimizer = torch.optim.SGD(net.parameters(), lr = lr);

    train(net, trainLoader, testLoader, lossFunc, optimizer, lr, maxEpoch, accuracy_num);


def chapter4_Dropout_My(plot : bool = True):
    lr = 0.01;
    batchSize, maxEpoch = 256, 100;
    dropout1, dropout2 = 0.2, 0.5;
    inputSize, hiddenSize1, hiddenSize2, outputSize = 28 * 28, 256, 256, 10;

    mnist = MNIST.MNIST("data/FashionMNIST/raw/");
    trainIterator = NN.SequentialDataIterator([mnist.trainX, mnist.trainY], batchSize = batchSize, shuffle = True);
    testIterator = NN.SequentialDataIterator([mnist.testX, mnist.testY], batchSize = batchSize, shuffle = True);
    lossFunc = NN.SoftmaxWithCrossEntropy1DLoss();
    optimizer = NN.SGD(lr = lr);
    evaluator = NN.ClassifierAccuracyEvaluator();

    model = NN.SequentialContainer(
        NN.FlattenLayer(),
        NN.AffineLayer(inputSize, hiddenSize1 * 4),
        NN.MaxoutLayer(k = 4),
        NN.DropoutLayer(dropout1),
        NN.AffineLayer(hiddenSize1, hiddenSize2 * 4),
        NN.MaxoutLayer(k = 4),
        NN.DropoutLayer(dropout2),
        NN.AffineLayer(hiddenSize2, outputSize, weightHandler = None),
    );
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = plot);

    X, y = next(iter(testIterator));
    yHat, = model.predictOne(X);
    rightCount = int(np.sum(np.argmax(yHat, axis = -1) == y));
    print(f"{rightCount}, {rightCount / len(y)}\n\n");


def kFold(k : int, trainSetX : np.ndarray, trainSetY: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    foldSize = int(len(trainSetX) // k);
    data = [(trainSetX[i * foldSize: (i + 1) * foldSize], trainSetY[i * foldSize : (i + 1) * foldSize]) for i in range(k - 1)];
    data.append((trainSetX[(k - 1) * foldSize: ], trainSetY[(k - 1) * foldSize: ]));

    folds = [];
    for i in list(range(k)):
        validX, validY = data.pop(i);
        trainX, trainY = np.concatenate([item[0] for item in data], axis = 0), np.concatenate([item[1] for item in data], axis = 0);
        folds.append((trainX, trainY, validX, validY));
        data.insert(i, (validX, validY));

    return folds;


def chapter4_Kaggle_HousePrices_LR_PyTorch(plot : bool = True):
    fitResults  = [];
    trainSetX, trainSetY, testX, testID = loadKaggleHousePrices();

    k, lr, batchSize, maxEpoch = 5, 5, 64, 100;
    lossFunc = nn.MSELoss();

    for trainX, trainY, validX, validY in kFold(k, trainSetX, trainSetY):
        net = nn.Sequential(
            nn.Linear(trainSetX.shape[-1], 1),
        );
        net.apply(initWeights);

        optimizer = torch.optim.Adam(net.parameters(), lr);
        # trainIterator = data.DataLoader(data.TensorDataset(torch.Tensor(trainX), torch.Tensor(trainY).reshape(-1, 1)), batch_size = batchSize, shuffle = True);
        # validIterator = data.DataLoader(data.TensorDataset(torch.Tensor(validX), torch.Tensor(validY).reshape(-1, 1)), batch_size = batchSize, shuffle = False);
        # fitResults.append(train(net, trainIterator, validIterator, lossFunc, optimizer, lr, maxEpoch, accuracy_log_mse, plot = plot));
        trainIterator = data.DataLoader(data.TensorDataset(torch.Tensor(trainX), torch.Tensor(np.log(trainY)).reshape(-1, 1)), batch_size = batchSize, shuffle = True);
        validIterator = data.DataLoader(data.TensorDataset(torch.Tensor(validX), torch.Tensor(np.log(validY)).reshape(-1, 1)), batch_size = batchSize, shuffle = False);
        fitResults.append(train(net, trainIterator, validIterator, lossFunc, optimizer, lr, maxEpoch, accuracy_mse, plot = plot));

    for i, item in enumerate(fitResults):
        print(f"fold {i + 1}, test loss: {item.testLossData[-1]}, test accuracy: {item.testAccuracyData[-1]}");

    meanLoss = sum([item.testLossData[-1] for item in fitResults]) / k;
    meanAccuracy = sum([math.sqrt(item.testAccuracyData[-1]) for item in fitResults]) / k;
    print(f"mean loss: {meanLoss}, mean accuracy: {meanAccuracy}");


def chapter4_Kaggle_HousePrices_LR_My(plot : bool = False):
    fitResults  = [];
    trainSetX, trainSetY, testX, testID = loadKaggleHousePrices();

    k, lr, batchSize, maxEpoch = 5, 3, 32, 500;
    lossFunc = NN.IdentityWithMeanSquareLoss();
    evaluator = NN.MseAccuracyEvaluator(takeRoot = True, takeLog = True, logMinValue = 1.0);

    for trainX, trainY, validX, validY in kFold(k, trainSetX, trainSetY):
        idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 58, 59, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 194, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 239, 240, 241, 243, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329];
        trainX, validX = trainX[:, idx], validX[:, idx];

        model = NN.SequentialContainer(
            NN.AffineLayer(trainX.shape[1], 1),
            # NN.MinMaxLayer(1),
            # NN.FunctionalNetModule("Log", lambda x: np.log(x), lambda x, y: 1/ x),
        );

        optimizer = NN.Adam(lr);
        # trainIterator = NN.SequentialDataIterator([trainX, np.log(trainY).reshape(-1, 1)], batchSize = batchSize, shuffle = True);
        # validIterator = NN.SequentialDataIterator([validX, np.log(validY).reshape(-1, 1)], batchSize = batchSize, shuffle = False);
        trainIterator = NN.SequentialDataIterator([trainX, trainY.reshape(-1, 1)], batchSize = batchSize, shuffle = True);
        validIterator = NN.SequentialDataIterator([validX, validY.reshape(-1, 1)], batchSize = batchSize, shuffle = False);
        fitResults.append(model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator = validIterator, evaluator = evaluator, minEpoch = 0, plot = plot));

    for i, item in enumerate(fitResults):
        print(f"fold {i + 1}, test loss: {item.finalTestLoss}, test accuracy: {item.finalTestAccuracy}");

    meanLoss = sum([item.finalTestLoss for item in fitResults]) / k;
    meanAccuracy = sum([item.finalTestAccuracy for item in fitResults]) / k;
    print(f"mean loss: {meanLoss}, mean accuracy: {meanAccuracy}");


def chapter4_Kaggle_HousePrices_LR_Predication_My(plot : bool = True):
    fitResults  = [];
    trainSetX, trainSetY, testX, testID = loadKaggleHousePrices();

    trainSize = int(len(trainSetX) * 0.8);
    idx = np.random.permutation(len(trainSetX));
    trainX, trainY = trainSetX[idx[: trainSize]], trainSetY[idx[: trainSize]];
    validX, validY = trainSetX[idx[trainSize: ]], trainSetY[idx[trainSize: ]];

    lr, batchSize, maxEpoch = 5, 64, 1000;
    lossFunc = NN.IdentityWithMeanSquareLoss();
    evaluator = NN.MseAccuracyEvaluator(takeRoot = True, takeLog = True, logMinValue = 1.0);

    model = NN.SequentialContainer(
        NN.AffineLayer(trainSetX.shape[1], 1, weightHandler = NN.L1WeightDecay(0.00001)),
    );
    optimizer = NN.Adam(lr);
    trainIterator = NN.SequentialDataIterator([trainX, trainY.reshape(-1, 1)], batchSize = batchSize, shuffle = True);
    validIterator = NN.SequentialDataIterator([validX, validY.reshape(-1, 1)], batchSize = batchSize, shuffle = False);
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator = validIterator, evaluator = evaluator, minEpoch = 0, plot = plot);

    testYHat, = model.predictOne(testX);
    pdf = pd.DataFrame({"Id" : testID.flatten(), "SalePrice": testYHat.flatten()});
    pdf.to_csv("data/PyTorchTest/chapter4_Kaggle_HousePrices_LR_Predication_My.csv", index = False);


def chapter4_Kaggle_HousePrices_MLP_PyTorch(plot : bool = False):
    fitResults  = [];
    trainSetX, trainSetY, testX, testID = loadKaggleHousePrices();

    k, lr, batchSize, maxEpoch = 5, 0.1, 64, 100;
    lossFunc = nn.MSELoss();

    for trainX, trainY, validX, validY in kFold(k, trainSetX, trainSetY):
        idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 58, 59, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 194, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
               239, 240, 241, 243, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329];
        trainX, validX = trainX[:, idx], validX[:, idx];

        drop = 0.5;
        net = nn.Sequential(
            nn.Linear(trainX.shape[-1], 1024),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        );
        net.apply(initWeights);

        optimizer = torch.optim.Adam(net.parameters(), lr);
        # trainIterator = data.DataLoader(data.TensorDataset(torch.Tensor(trainX), torch.Tensor(np.log(trainY)).reshape(-1, 1)), batch_size = batchSize, shuffle = True);
        # validIterator = data.DataLoader(data.TensorDataset(torch.Tensor(validX), torch.Tensor(np.log(validY)).reshape(-1, 1)), batch_size = batchSize, shuffle = False);
        # fitResults.append(train(net, trainIterator, validIterator, lossFunc, optimizer, lr, maxEpoch, accuracy_mse, plot = plot));
        trainIterator = data.DataLoader(data.TensorDataset(torch.Tensor(trainX), torch.Tensor(trainY).reshape(-1, 1)), batch_size = batchSize, shuffle = True);
        validIterator = data.DataLoader(data.TensorDataset(torch.Tensor(validX), torch.Tensor(validY).reshape(-1, 1)), batch_size = batchSize, shuffle = False);
        fitResults.append(train(net, trainIterator, validIterator, lossFunc, optimizer, lr, maxEpoch, accuracy_log_mse, plot = plot));

    for i, item in enumerate(fitResults):
        print(f"fold {i + 1}, train loss: {item.trainingLossData[-1]}, train accuracy: {math.sqrt(item.trainingAccuracyData[-1])}, test loss: {item.testLossData[-1]}, test accuracy: {math.sqrt(item.testAccuracyData[-1])}");

    meanLoss = sum([item.testLossData[-1] for item in fitResults]) / k;
    meanAccuracy = sum([math.sqrt(item.testAccuracyData[-1]) for item in fitResults]) / k;
    print(f"mean valid loss: {meanLoss}, mean valid accuracy: {meanAccuracy}");


def chapter4_Kaggle_HousePrices_MLP_My(plot : bool = True):
    fitResults  = [];
    trainSetX, trainSetY, testX, testID = loadKaggleHousePrices();

    k, lr, batchSize, maxEpoch = 5, 0.1, 64, 100;
    lossFunc = NN.IdentityWithHuberLoss();
    evaluator = NN.MseAccuracyEvaluator(takeRoot = True, takeLog = True, logMinValue = 1.0);

    for trainX, trainY, validX, validY in kFold(k, trainSetX, trainSetY):
        # idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 58, 59, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 194, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
        #        239, 240, 241, 243, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329];
        # trainX, validX = trainX[:, idx], validX[:, idx];

        drop = 0.0;
        weightDecay = NN.L1Regularization(0.001);
        model = NN.SequentialContainer(
            NN.AffineLayer(trainX.shape[1], 1024, weightHandler = weightDecay),
            NN.ReluLayer(),
            NN.DropoutLayer(drop),
            NN.AffineLayer(1024, 1, weightHandler = weightDecay),
        );

        optimizer = NN.Adam(lr);
        trainIterator = NN.SequentialDataIterator([trainX, trainY.reshape(-1, 1)], batchSize = batchSize, shuffle = True);
        validIterator = NN.SequentialDataIterator([validX, validY.reshape(-1, 1)], batchSize = batchSize, shuffle = False);
        fitResults.append(model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator = validIterator, evaluator = evaluator, minEpoch = 0, plot = plot));

    for i, item in enumerate(fitResults):
        print(f"fold {i + 1}, train loss: {item.trainingLossData[-1]}, train accuracy: {item.trainingAccuracyData[-1]}, final test loss: {item.finalTestLoss}, final test accuracy: {item.finalTestAccuracy}");

    meanLoss = sum([item.finalTestLoss for item in fitResults]) / k;
    meanAccuracy = sum([item.finalTestAccuracy for item in fitResults]) / k;
    print(f"mean loss: {meanLoss}, mean accuracy: {meanAccuracy}");


def chapter4_Kaggle_HousePrices_MLP_Predication_My(plot : bool = True):
    fitResults  = [];
    trainSetX, trainSetY, testX, testID = loadKaggleHousePrices();

    trainSize = int(len(trainSetX) * 0.8);
    idx = np.random.permutation(len(trainSetX));
    trainX, trainY = trainSetX[idx[: trainSize]], trainSetY[idx[: trainSize]];
    validX, validY = trainSetX[idx[trainSize: ]], trainSetY[idx[trainSize: ]];

    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 58, 59, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 194, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
           239, 240, 241, 243, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329];
    trainX, validX, testX = trainX[:, idx], validX[:, idx], testX[:, idx];

    lr, batchSize, maxEpoch = 0.1, 64, 100;
    lossFunc = NN.IdentityWithHuberLoss();
    evaluator = NN.MseAccuracyEvaluator(takeRoot = True, takeLog = True, logMinValue = 1.0);

    drop = 0.5;
    weightDecay = NN.L1Regularization(0.00);
    model = NN.SequentialContainer(
        NN.AffineLayer(trainX.shape[1], 1024, weightHandler = weightDecay),
        NN.ReluLayer(),
        NN.DropoutLayer(drop),
        NN.AffineLayer(1024, 1, weightHandler = weightDecay),
    );
    optimizer = NN.Adam(lr);
    trainIterator = NN.SequentialDataIterator([trainX, trainY.reshape(-1, 1)], batchSize = batchSize, shuffle = True);
    validIterator = NN.SequentialDataIterator([validX, validY.reshape(-1, 1)], batchSize = batchSize, shuffle = False);
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator = validIterator, evaluator = evaluator, minEpoch = 0, plot = plot);

    testYHat, = model.predictOne(testX);
    pdf = pd.DataFrame({"Id" : testID.flatten(), "SalePrice": testYHat.flatten()});
    pdf.to_csv("data/PyTorchTest/chapter4_Kaggle_HousePrices_MLP_Predication_My.csv", index = False);


def chapter4_Kaggle_HousePrices_MLP_LogY_My(plot : bool = True):
    fitResults  = [];
    trainSetX, trainSetY, testX, testID = loadKaggleHousePrices();
    trainSetY = np.log(trainSetY);

    k, lr, batchSize, maxEpoch = 5, 0.1, 64, 200;
    lossFunc = NN.IdentityWithHuberLoss();
    evaluator = NN.MseAccuracyEvaluator(takeRoot = True, takeLog = False, logMinValue = 1.0);

    for trainX, trainY, validX, validY in kFold(k, trainSetX, trainSetY):
        # idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 58, 59, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 194, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
        #        239, 240, 241, 243, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329];
        # trainX, validX = trainX[:, idx], validX[:, idx];

        drop = 0.0;
        weightDecay = NN.L1Regularization(0.000);
        model = NN.SequentialContainer(
            NN.AffineLayer(trainX.shape[1], 256, weightHandler = weightDecay),
            NN.ReluLayer(),
            NN.DropoutLayer(drop),
            NN.AffineLayer(256, 1, weightHandler = weightDecay),
        );

        optimizer = NN.Adam(lr);
        trainIterator = NN.SequentialDataIterator([trainX, trainY.reshape(-1, 1)], batchSize = batchSize, shuffle = True);
        validIterator = NN.SequentialDataIterator([validX, validY.reshape(-1, 1)], batchSize = batchSize, shuffle = False);
        fitResults.append(model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator = validIterator, evaluator = evaluator, minEpoch = 0, plot = plot));

    for i, item in enumerate(fitResults):
        print(f"fold {i + 1}, train loss: {item.trainingLossData[-1]}, train accuracy: {item.trainingAccuracyData[-1]}, final test loss: {item.finalTestLoss}, final test accuracy: {item.finalTestAccuracy}");

    meanLoss = sum([item.finalTestLoss for item in fitResults]) / k;
    meanAccuracy = sum([item.finalTestAccuracy for item in fitResults]) / k;
    print(f"mean loss: {meanLoss}, mean accuracy: {meanAccuracy}");


def chapter4_Kaggle_HousePrices_MLP_LogY_Predication_My(plot : bool = True):
    fitResults  = [];
    trainSetX, trainSetY, testX, testID = loadKaggleHousePrices();

    trainSetY = np.log(trainSetY);
    scalerY = NN.StandardScaler();
    scalerY.fit(trainSetY);
    trainSetY = scalerY.transform(trainSetY);

    trainSize = int(len(trainSetX) * 0.8);
    idx = np.random.permutation(len(trainSetX));
    trainX, trainY = trainSetX[idx[: trainSize]], trainSetY[idx[: trainSize]];
    validX, validY = trainSetX[idx[trainSize: ]], trainSetY[idx[trainSize: ]];

    # idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 58, 59, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 194, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
    #        239, 240, 241, 243, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329];
    # trainX, validX, testX = trainX[:, idx], validX[:, idx], testX[:, idx];

    lr, batchSize, maxEpoch = 0.1, 64, 100;
    lossFunc = NN.IdentityWithHuberLoss();
    evaluator = NN.MseAccuracyEvaluator(takeRoot = True, takeLog = False, logMinValue = 1.0);

    drop = 0.0;
    weightDecay = NN.L2Regularization(0.0001);
    model = NN.SequentialContainer(
        NN.AffineLayer(trainX.shape[1], 1024, weightHandler = weightDecay),
        NN.ReluLayer(),
        NN.DropoutLayer(drop),
        NN.AffineLayer(1024, 1, weightHandler = weightDecay),
    );
    optimizer = NN.Adam(lr);
    trainIterator = NN.SequentialDataIterator([trainX, trainY.reshape(-1, 1)], batchSize = batchSize, shuffle = True);
    validIterator = NN.SequentialDataIterator([validX, validY.reshape(-1, 1)], batchSize = batchSize, shuffle = False);
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator = validIterator, evaluator = evaluator, minEpoch = 0, plot = plot);

    print(f"final LOG_RMSE: {math.sqrt(meanSquareError(scalerY.inverse(model.predictOne(validX)[0]), scalerY.inverse(validY.reshape(-1, 1))))}");

    testYHat, = model.predictOne(testX);
    testYHat = np.exp(scalerY.inverse(testYHat));
    pdf = pd.DataFrame({"Id" : testID.flatten(), "SalePrice": testYHat.flatten()});
    pdf.to_csv("data/PyTorchTest/chapter4_Kaggle_HousePrices_MLP_LogY_Predication_My.csv", index = False);


def chapter6_learnKernel(plot : bool = True):
    X = np.ones((6, 8));
    X[:, 2:6] = 0;

    Y = np.zeros((6, 7));
    Y[:, 1] = 1;
    Y[:, 5] = -1;

    lr, batchSize, maxEpoch = 0.03, 1, 100;
    lossFunc = NN.IdentityWithMeanSquareLoss();

    optimizer = NN.SGD(lr);
    trainIterator = NN.SequentialDataIterator([np.reshape(X, (1, 1) + X.shape), np.reshape(Y, (1, -1))], batchSize = batchSize, shuffle = False);
    model = NN.SequentialContainer(
        NN.Convolution2DLayer(1, 1, (1, 2)),
        NN.FlattenLayer(),
    );
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, plot = plot);

    print(f"weight: {model.params[0].value}");
    print(f"bias: {model.params[1].value}");

    print("exit.");


def chapter6_LeNet5_PyTorch(plot : bool = True):
    lr = 0.1;
    batchSize, maxEpoch = 256, 10;

    trainLoader, testLoader = loadFashionMNIST(batchSize);
    lossFunc = nn.CrossEntropyLoss();

    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size = 5, padding = 2), nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),
        nn.Conv2d(6, 16, kernel_size = 5), nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10),
    );
    net.apply(initWeights);
    optimizer = torch.optim.SGD(net.parameters(), lr = lr);

    train(net, trainLoader, testLoader, lossFunc, optimizer, lr, maxEpoch, accuracy_num);


def chapter6_LeNet5_My(plot : bool = True):
    lr = 0.3;
    batchSize, maxEpoch = 256, 100;

    mnist = MNIST.MNIST("data/FashionMNIST/raw/");
    trainIterator = NN.SequentialDataIterator([mnist.trainX, mnist.trainY], batchSize = batchSize, shuffle = True);
    testIterator = NN.SequentialDataIterator([mnist.testX, mnist.testY], batchSize = batchSize, shuffle = True);
    lossFunc = NN.SoftmaxWithCrossEntropy1DLoss();
    evaluator = NN.ClassifierAccuracyEvaluator();

    model = NN.SequentialContainer(
        NN.Convolution2DLayer(1, 6, kernelSize = 5, padding = 2), NN.ReluLayer(),
        NN.MaxPooling2DLayer(poolingSize = 2),
        NN.Convolution2DLayer(6, 16, kernelSize = 5), NN.ReluLayer(),
        NN.MaxPooling2DLayer(poolingSize = 2),
        NN.FlattenLayer(),
        NN.AffineLayer(16 * 5 * 5, 120), NN.ReluLayer(),
        NN.AffineLayer(120, 84), NN.ReluLayer(),
        NN.AffineLayer(84, 10),
    );

    minLr = 1e-7;

    optimizer = NN.SGDM(lr = lr, weightDecay = 0.0, decoupledDecay = False);
    optimizer = NN.NetOptimizerWithLrScheduler(optimizer, NN.AggregateNetLrScheduler([
        NN.LinearNetLrScheduler(lr, startFactor = 0.01, minEpoch = 0, maxEpoch = 5),
        NN.MultiStepNetLrScheduler(lr, list(range(10, 100, 5)), minEpoch = 5),
        # NN.CyclicNetLrScheduler(NN.CosineNetLrScheduler(lr, minLr, minEpoch = 0, maxEpoch = 9), cycleSize = 10, minEpoch = 5, maxEpoch = 94),
        # # NN.CosineNetLrScheduler(lr, minLr, minEpoch = 5, maxEpoch = 24),
        # NN.ConstantNetLrScheduler(minLr, minEpoch = 94),
    ]));

    # optimizer = NN.AdamW(lr = lr);
    # optimizer = NN.NetOptimizerWithLrScheduler(optimizer, NN.AggregateNetLrScheduler([
    #     NN.LinearNetLrScheduler(lr, startFactor = 0.01, minEpoch = 0, maxEpoch = 4),
    #     NN.CosineNetLrScheduler(lr, minLr, minEpoch = 4, maxEpoch = 24),
    #     NN.ConstantNetLrScheduler(minLr, minEpoch = 24),
    # ]));

    result = model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = plot);

    with open("data/chapter6_LeNet5_My.pkl", "wb") as file:
        pickle.dump(result, file, pickle.DEFAULT_PROTOCOL);

    X, y = next(iter(testIterator));
    yHat, = model.predictOne(X);
    rightCount = int(np.sum(np.argmax(yHat, axis = -1) == y));
    print(f"{rightCount}, {rightCount / len(y)}\n\n");


def chapter8_PredictSine_My(plot : bool = True):
    def getSineWave(number, tau : int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ts = np.arange(number, dtype = defaultDType);
        xs = np.sin(0.01 * ts) + np.random.randn(number) * 0.2;

        labels = xs[tau: ];
        features = np.zeros((len(labels), tau));
        for i in range(tau):
            features[:, i] = xs[i: -tau + i];

        return xs, ts, features, labels;


    arSize = 4;
    x, t, X, y = getSineWave(1000, tau = arSize);
    if plot:
        plt.figure();
        plt.plot(t, x, "-");
        plt.show(block = True);

    trainSize = 600;
    trainX, trainY = X[: trainSize], y[: trainSize].reshape(-1, 1);

    lr = 0.01;
    batchSize, maxEpoch = 16, 5;
    trainIterator = NN.SequentialDataIterator([trainX, trainY], batchSize = batchSize, shuffle = True);
    lossFunc = NN.IdentityWithMeanSquareLoss();
    optimizer = NN.Adam(lr = lr);

    model = NN.SequentialContainer(
        NN.AffineLayer(arSize, 10),
        NN.ReluLayer(),
        NN.AffineLayer(10, 1),
    );
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, plot = plot);

    oneStepPreds = model.predictOne(X)[0].flatten();
    multiStepPreds = np.zeros_like(x);
    multiStepPreds[: trainSize + arSize] = x[: trainSize + arSize];
    for i in range(trainSize + arSize, len(multiStepPreds)):
        multiStepPreds[i] = model.predictOne(multiStepPreds[i - arSize: i].reshape(1, -1))[0].flatten();

    plt.figure();
    plt.plot(t, x, "-g");
    plt.plot(t[arSize:], oneStepPreds, "-b");
    plt.plot(t[trainSize + arSize:], multiStepPreds[trainSize + arSize:], "-r");
    plt.show(block = True);


def readTimeMachine():
    with open("/media/WindowsE/Data/timemachine.txt", "rt", encoding = "utf8") as file:
        lines = file.readlines();

    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines];


def tokenize(lines : List[str], tokenType : str = "word") -> List[List[str]]:
    if tokenType == "word":
        return [line.split() for line in lines];
    elif tokenType == "char":
        return [list(line) for line in lines];
    else:
        print(f"error: unknown token type {tokenType}");


class Vocab:
    def __init__(self, tokens : Union[List[str], List[List[str]]], minFreq : int = 0, reservedTokens : List[str] = None):
        counter = self._countCorpus(tokens);

        self._tokenFreq = sorted(counter.items(), key = lambda item: item[1], reverse = True);
        self._id2Token = ["<unk>"] + (reservedTokens if reservedTokens is not None else []);
        self._token2ID = {token : idx for idx, token in enumerate(self._id2Token)};

        for token, freq in self._tokenFreq:
            if freq < minFreq:
                break;

            self._token2ID[token] = len(self._id2Token);
            self._id2Token.append(token);


    def __len__(self):
        return len(self._id2Token);


    def __getitem__(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.__getitem__(token) for token in tokens];
        else:
            return self._token2ID[tokens] if tokens in self._token2ID else 0;


    def getTokens(self, indices):
        if isinstance(indices, (list, tuple)):
            return [self.getTokens(idx) for idx in indices];
        else:
            return self._id2Token[indices];


    @property
    def unk(self) -> int:
        return 0;


    @property
    def tokenFreq(self) -> List[Tuple[str, int]]:
        return self._tokenFreq;


    @property
    def token2ID(self) -> Dict[str, int]:
        return self._token2ID;


    @property
    def id2Token(self) -> List[str]:
        return self._id2Token;


    def _countCorpus(self, tokens) -> collections.Counter:
        if len(tokens) > 0 and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line];

        return collections.Counter(tokens);


def loadCorpusTimeMachine(tokenType = "char", maxTokens : int = -1) -> Tuple[List[int], Vocab]:
    lines = readTimeMachine();
    tokens = tokenize(lines, tokenType);
    vocab = Vocab(tokens);
    corpus = [vocab[token] for line in tokens for token in line];

    if maxTokens > 0:
        corpus = corpus[:maxTokens];

    return corpus, vocab;


def loadDataTimeMachine(batchSize : int, stepSize : int, tokenType = "char", maxTokens : int = -1, randomSample : bool = False, useTorch : bool = False) -> Tuple[NN.IDataIterator, Vocab]:
    corpus, vocab = loadCorpusTimeMachine(tokenType, maxTokens);
    if not useTorch:
        X, Y = expand2OneHot(np.array(corpus[: -1]), len(vocab)), np.array(corpus[1:]);
    else:
        X, Y = torch.Tensor(corpus[: -1]).long(), torch.Tensor(corpus[1:]);
    return NN.PartitionedDataIterator([X, Y], batchSize, stepSize, randomSample), vocab;


def chapter8_get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def chapter8_init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def chapter8_rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


def predict_ch8(prefix, num_preds, net, vocab : Vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.id2Token[i] for i in outputs])


def chapter8_grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state = None
    metric = Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            chapter8_grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            chapter8_grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1])


#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()

    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)

    # 训练和预测
    pplList = [];
    for epoch in range(num_epochs):
        ppl = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        pplList.append(ppl);
        print(f"epoch: {epoch}, perplexity: {ppl}");

    plt.figure();
    plt.plot(pplList);
    plt.show(block = True);


class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = chapter8_get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


def chapter8_TimeMachine_Zero(plot : bool = True):
    batch_size, num_steps = 32, 35
    train_iter, vocab = loadDataTimeMachine(batch_size, num_steps, maxTokens = 10000, useTorch = True);

    num_hiddens = 512
    device = tryGPU();
    net = RNNModelScratch(len(vocab), num_hiddens, device, chapter8_get_params, chapter8_init_rnn_state, chapter8_rnn)

    print(predict_ch8('time traveller', 10, net, vocab, device));

    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)

    print(predict_ch8('time traveller', 50, net, vocab, device));
    print(predict_ch8('traveller', 50, net, vocab, device));


def chapter8_TimeMachine_PyTorch(plot : bool = True):
    batch_size, num_steps = 32, 35;
    train_iter, vocab = loadDataTimeMachine(batch_size, num_steps, maxTokens = 10000, useTorch = True);

    num_hiddens = 512
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    device = tryGPU();
    net = RNNModel(rnn_layer, vocab_size = len(vocab))
    net = net.to(device)

    print(predict_ch8('time traveller', 10, net, vocab, device));

    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)

    print(predict_ch8('time traveller', 50, net, vocab, device));
    print(predict_ch8('traveller', 50, net, vocab, device));


class TimeMachineRnnModel(NN.SequentialContainer):
    def __init__(self, *modules : NN.INetModule):
        super().__init__(*modules);


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X = data[0];
        return super().forward(X.transpose(1, 0, 2));


    def getFinalTag(self, T : np.ndarray) -> Optional[np.ndarray]:
        return T.T.flatten();


def chapter8_TimeMachine_My(plot : bool = True):
    def predict(prefix : str, size : int, net : NN.INetModel, vocab : Vocab) -> str:
        net.reset();
        net.context.isTrainingMode = False;

        outputs = vocab[list(prefix)];

        net.forward(expand2OneHot(np.array(outputs[: -1]).reshape(1, -1), len(vocab)));

        for _ in range(size):
            Y, = net.forward(expand2OneHot(np.array(outputs[-1]).reshape(1, 1), len(vocab)));
            outputs.append(int(np.argmax(Y)));

        return "".join([vocab.id2Token[i] for i in outputs]);


    batchSize, stepSize = 32, 35;
    trainIterator, vocab = loadDataTimeMachine(batchSize, stepSize, maxTokens = 10000, randomSample = False);
    vocabSize, hiddenSize = len(vocab), 256;

    lr = 1.0;
    maxEpoch = 500;

    lossFunc = NN.SoftmaxWithCrossEntropy1DLoss();
    optimizer = NN.GradientsClipping(1.0, NN.SGD(lr = lr));
    # optimizer = NN.SGD(lr = lr);
    evaluator = NN.PerplexityAccuracyEvaluator();

    model = TimeMachineRnnModel(
        # NN.GruLayer(vocabSize, hiddenSize, stateful = True),
        NN.StackRnnLayer(vocabSize, hiddenSize, NN.GruLayer, layersNum = 2, stateful = True),
        NN.AffineLayer(hiddenSize, vocabSize),
    );

    print(predict("time traveller", 10, model, vocab));

    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, evaluator = evaluator, plot = plot);

    print(predict("time traveller", 50, model, vocab));
    print(predict("traveller", 50, model, vocab));

    # oneStepPreds = model.predictOne(X)[0].flatten();
    # multiStepPreds = np.zeros_like(x);
    # multiStepPreds[: trainSize + arSize] = x[: trainSize + arSize];
    # for i in range(trainSize + arSize, len(multiStepPreds)):
    #     multiStepPreds[i] = model.predictOne(multiStepPreds[i - arSize: i].reshape(1, -1))[0].flatten();
    #
    # plt.figure();
    # plt.plot(t, x, "-g");
    # plt.plot(t[arSize:], oneStepPreds, "-b");
    # plt.plot(t[trainSize + arSize:], multiStepPreds[trainSize + arSize:], "-r");
    # plt.show(block = True);


def tokenizeNmt(line : str) -> Tuple[List[str], List[str]]:
    text = line.replace("\u202f", " ").replace("\u00a0", "").lower();
    text = ''.join([" " + c if c in ",.!?" and text[i - 1] != ' ' else c for i, c in enumerate(text)]);
    parts = text.split("\t");
    return parts[0].split(" "), parts[1].split(" ");


def truncatePad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[: num_steps]  # 截断

    return line + [padding_token] * (num_steps - len(line))  # 填充


def buildArrayNmt(lines : List[List[str]], vocab : Vocab, stepSize : int) -> Tuple[List[List[int]], List[int]]:
    padID, eosID = [vocab["<pad>"]], [vocab["<eos>"]];

    lines = [vocab[line] + eosID for line in lines];
    lines = [(line[: stepSize], stepSize) if len(line) >= stepSize else (line + padID * (stepSize - len(line)), len(line)) for line in lines];

    return [line for line, length in lines], [length for line, length in lines];


def loadDataNmt(filePath : str, batchSize : int, stepSize : int, exampleSize : Optional[int] = 600, useTorch : bool = False) -> Tuple[NN.IDataIterator, Vocab, Vocab]:
    with open(filePath, "r", encoding = "utf8") as file:
        lines = file.readlines();

    sourceLines, targetLines = [], [];
    with multiprocessing.Pool() as pool:
        for item1, item2 in pool.map(tokenizeNmt, lines if exampleSize is None or exampleSize < 0 else lines[: exampleSize]):
            sourceLines.append(item1);
            targetLines.append(item2);

    # plt.figure();
    # plt.hist([[len(item) for item in sourceLines], [len(item) for item in targetLines]]);
    # plt.show(block = True);

    sourceVocab = Vocab(sourceLines, minFreq = 2, reservedTokens = ["<pad>", "<bos>", "<eos>"]);
    sourceData, sourceLength = buildArrayNmt(sourceLines, sourceVocab, stepSize);

    targetVocab = Vocab(targetLines, minFreq = 2, reservedTokens = ["<pad>", "<bos>", "<eos>"]);
    targetData, targetLength = buildArrayNmt(targetLines, targetVocab, stepSize);

    if useTorch:
        dataIterator = NN.SequentialDataIterator([torch.Tensor(sourceData).long(), torch.Tensor(sourceLength).long(), torch.Tensor(targetData).long(), torch.Tensor(targetLength).long()], batchSize, shuffle = True);
    else:
        dataIterator = NN.SequentialDataIterator([np.array(sourceLength), np.array(targetLength), np.array(sourceData), np.array(targetData)], batchSize, shuffle = True);

    # for X, Xl, Y, Yl in dataIterator:
    #     print(X);
    #     print(Xl);
    #     print(Y);
    #     print(Yl);
    #     break;

    return dataIterator, sourceVocab, targetVocab;


class EncoderTorch(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs);


    def forward(self, X, *args):
        raise NotImplementedError();


class DecoderTorch(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs);


    def initState(self, encOutputs, *args):
        raise NotImplementedError();


    def forward(self, X, state):
        raise NotImplementedError();


class EncoderDecoderTorch(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs);

        self.encoder = encoder;
        self.decoder = decoder;


    def forward(self, encX, decX, *args):
        encOutputs = self.encoder(encX, *args);
        decState = self.decoder.initState(encOutputs, *args);
        return self.decoder(decX, decState);


class Seq2SeqEncoderTorch(EncoderTorch):
    def __init__(self, vocabSize : int, embedSize : int, hiddenSize : int, layersNum : int, dropout : float = 0.0, **kwargs):
        super().__init__(**kwargs);

        self.embedding = nn.Embedding(vocabSize, embedSize);
        self.rnn = nn.GRU(embedSize, hiddenSize, layersNum, dropout = dropout);


    def forward(self, X, *args):
        X = self.embedding(X);
        X = torch.permute(X, (1, 0, 2));

        outputs, states = self.rnn(X);
        return outputs, states;


class Seq2SeqDecoderTorch(DecoderTorch):
    def __init__(self, vocabSize : int, embedSize : int, hiddenSize : int, layersNum : int, dropout : float = 0.0, **kwargs):
        super().__init__(**kwargs);

        self.embedding = nn.Embedding(vocabSize, embedSize);
        self.rnn = nn.GRU(embedSize + hiddenSize, hiddenSize, layersNum, dropout = dropout);
        self.dense = nn.Linear(hiddenSize, vocabSize);


    def initState(self, encOutputs, *args):
        return encOutputs[1], encOutputs[1][-1];


    def forward(self, X, state):
        X = self.embedding(X);
        X = torch.permute(X, (1, 0, 2));

        context = state[-1].repeat(X.shape[0], 1, 1);
        X_and_context = torch.cat((X, context), 2);
        outputs, states = self.rnn(X_and_context, state[0]);
        outputs = self.dense(outputs).permute(1, 0, 2);
        return outputs, (states, state[-1]);


#@save
class AttentionDecoder(DecoderTorch):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


def sequenceMask(X, validLen, value = 0.0):
    mask = torch.arange(X.shape[1], dtype = torch.int32)[None, :] >= validLen[:, None];
    X[mask] = value;
    return X;


class MaskedSoftmaxCELossPyTorch(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequenceMask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        # weighted_loss = (unweighted_loss * weights).mean(dim=1)
        weighted_loss = (unweighted_loss * weights) / torch.sum(valid_len);
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device, plot : bool = True):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    # net.load_state_dict(torch.load("data/PyTorchTest/chapter10_Nmt_Transformer_PyTorch.params"));

    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    loss = MaskedSoftmaxCELossPyTorch();
    lossData = [];

    for epoch in range(num_epochs):
        metric = Accumulator(2)

        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            chapter8_grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                # metric.add(l.sum(), num_tokens);
                metric.add(l.sum(), 1);

        lossData.append(metric[0] / metric[1]);
        print(f"epoch: {epoch}, loss: {lossData[-1]}");

    if plot:
        plt.figure();
        plt.plot(lossData);
        plt.show(block = True);


def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0.0, 1.0 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncatePad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.initState(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.getTokens(output_seq)), attention_weight_seq


def chapter9_Nmt_PyTorch():
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, tryGPU();

    train_iter, src_vocab, tgt_vocab = loadDataNmt("/media/WindowsE/Data/ANKI/fra-eng/fra.txt", batch_size, num_steps, exampleSize = 1000, useTorch = True)
    encoder = Seq2SeqEncoderTorch(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoderTorch(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoderTorch(encoder, decoder)

    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k = 2):.3f}');


def sequenceMask_My(maxLen : int, validLen : np.ndarray) -> np.ndarray:
    return np.arange(maxLen, dtype = np.int32)[None, :] < validLen[:, None];


class MaskedSoftmaxCELossMy(NN.SoftmaxWithCrossEntropy1DLoss):
    def __init__(self):
        super().__init__(reductionType = NN.LossReductionType.No);

        self._n = 0;


    def forward(self, *data: np.ndarray) -> float:
        X, validLen, T = data;
        # M = sequenceMask_My(T.shape[-1], validLen);
        M = getLossMaskByValidLength(T.shape[-1], validLen);
        L = super().forward(X, M, T);

        self._n = np.sum(M);
        # self._n = L.shape[-1];
        return float(np.sum(L) / self._n);


    def backward(self) -> Tuple[np.ndarray, ...]:
        dX, = super().backward();
        dX /= self._n;
        return dX, ;


def testSequenceSoftmaxWithCrossEntropy1DLoss1():
    N, D, C = 320, 24, 10;
    X, T = np.random.randn(N, D, C), np.random.choice(np.arange(C), size = (N, D), replace = True);
    validLen = np.random.choice(np.arange(D), size = N, replace = True);
    torchModule = MaskedSoftmaxCELossPyTorch();
    myModule = NN.SequenceSoftmaxWithCrossEntropy1DLoss();

    X1 = X;
    Y1 = myModule.forward(X1, validLen, T);
    dX1, = myModule.backward();

    X2 = torch.tensor(X, requires_grad = True);
    Y2 = torchModule(X2, torch.as_tensor(T), torch.as_tensor(validLen));
    Y2 = torch.sum(Y2);
    Y2.backward();
    Y2 = Y2.detach().numpy();
    dX2 = X2.grad.detach().numpy();

    print(f"SequenceSoftmaxWithCrossEntropy1DLoss, value1, Y error: {np.sum(np.abs(Y1 - Y2))}, dX error: {np.sum(np.abs(dX1 - dX2))}");
    print("\n");


class NmtSeq2SeqEncoderMy(NN.SequentialContainer):
    def __init__(self, embedSize: int, hiddenSize, layersNum: int, vocabSize : int, dropout : float = 0.0):
        super().__init__(
            NN.EmbeddingLayer(vocabSize, embedSize),
            NN.StackRnnLayer(embedSize, hiddenSize, NN.GruLayer, layersNum = layersNum, dropoutRatio = dropout, stateful = False, returnSequence = False, returnState = True)
        );


def testNmtSeq2SeqEncoderMyGradient1():
    batchSize, stepSize = 32, 12,
    embedSize, hiddenSize, layersNum, vocabSize = 14, 16, 2, 10;
    X = np.random.choice(np.arange(vocabSize), size = (stepSize, batchSize), replace = True);
    m = NmtSeq2SeqEncoderMy(embedSize, hiddenSize, layersNum, vocabSize);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    testModuleGradient(m, "NmtSeq2SeqEncoderMy, numericGradient1", X);
    print("\n");


class NmtSeq2SeqEncoderMyBiRnn(NN.SequentialContainer):
    _g_rnnCount = 0;

    @staticmethod
    def rnnSelector(inputSize, hiddenSize, stateful, returnSequence, returnState):
        NmtSeq2SeqEncoderMyBiRnn._g_rnnCount += 1;
        return NN.BiRnnLayer(inputSize if NmtSeq2SeqEncoderMyBiRnn._g_rnnCount == 1 else 2 * inputSize, hiddenSize, NN.GruLayer, stateful = stateful, returnSequence = returnSequence, returnState = returnState);


    def __init__(self, embedSize: int, hiddenSize, layersNum: int, vocabSize : int, dropout : float = 0.0):
        super().__init__(
            NN.EmbeddingLayer(vocabSize, embedSize),
            NN.StackRnnLayer(embedSize, hiddenSize, NmtSeq2SeqEncoderMyBiRnn.rnnSelector, layersNum = layersNum, dropoutRatio = dropout, stateful = False, returnSequence = False, returnState = True)
        );


def testNmtSeq2SeqEncoderMyBiRnnGradient1():
    batchSize, stepSize = 32, 12,
    embedSize, hiddenSize, layersNum, vocabSize = 14, 18, 2, 10;
    X = np.random.choice(np.arange(vocabSize), size = (stepSize, batchSize), replace = True);
    m = NmtSeq2SeqEncoderMyBiRnn(embedSize, hiddenSize, layersNum, vocabSize);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    testModuleGradient(m, "NmtSeq2SeqEncoderMyBiRnn, numericGradient1", X);
    print("\n");


class NmtSeq2SeqDecoderMy(NN.SequentialContainer):
    def __init__(self, embedSize: int, hiddenSize, layersNum: int, vocabSize : int, dropout : float = 0.0):
        self._embedSize = embedSize;

        self._embedding = NN.EmbeddingLayer(vocabSize, embedSize);
        self._decoder = NN.SequentialContainer(
            NN.StackRnnLayer(embedSize + hiddenSize, hiddenSize, NN.GruLayer, layersNum = layersNum, dropoutRatio = dropout, stateful = True),
            NN.AffineLayer(hiddenSize, vocabSize),
        );

        super().__init__(self._embedding, self._decoder);


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        X, State = data;

        embedX, = self._embedding.forward(X);
        contextX = np.concatenate((embedX, np.tile(State[-1], (len(embedX), 1, 1))), axis = -1);
        output, = self._decoder.forward(contextX, State);
        return output.transpose(1, 0, 2), ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dOutput = dout[0].transpose(1, 0, 2);
        dContextX, dState = self._decoder.backward(dOutput);
        dEmbedX = dContextX[:, :, : self._embedSize];
        dState[-1] += np.sum(dContextX[:, :, self._embedSize:], axis = 0);
        dX, = self._embedding.backward(dEmbedX);

        return dX, dState;


    def translate(self, State : np.ndarray, maxLen : int, bosID : int, eosID : int) -> List[int]:
        X, Y = np.array([bosID]).reshape(1, 1), [];
        for i in range(maxLen):
            embedX, = self._embedding.forward(X);
            contextX = np.concatenate((embedX, np.tile(State[-1], (len(embedX), 1, 1))), axis = -1);
            if i == 0:
                output, = self._decoder.forward(contextX, State);
            else:
                output, = self._decoder.forward(contextX);

            idx = int(np.argmax(output));
            if idx == eosID:
                break;

            Y.append(idx);
            X = np.array([idx]).reshape(1, 1);

        return Y;


def testNmtSeq2SeqDecoderMyGradient1():
    batchSize, stepSize = 32, 12,
    embedSize, hiddenSize, layersNum, vocabSize = 14, 16, 2, 10;
    X = np.random.choice(np.arange(vocabSize), size = (stepSize, batchSize), replace = True);
    State = np.random.randn(layersNum, batchSize, hiddenSize);
    m = NmtSeq2SeqDecoderMy(embedSize, hiddenSize, layersNum, vocabSize);
    Y, = m.forward(X, State);
    dX1, dState1 = m.backward(np.ones_like(Y));
    dStateN = numericGradient(lambda x: np.sum(m.forward(X, x)), State);
    print(f"NmtSeq2SeqDecoderMy, numericGradient1, dState error: {np.sum(np.abs(dState1 - dStateN))}");
    testModuleGradient(m, "NmtSeq2SeqDecoderMy, numericGradient1", X, State);
    print("\n");


class NmtEncoderDecoderMy(NN.NetModelBase):
    def __init__(self, embedSize : int, hiddenSize, layersNum : int, sourceVocab : Vocab, targetVocab : Vocab, dropout : float = 0.0):
        self._sourceVocab = sourceVocab;
        self._targetVocab = targetVocab;
        # self._encoder = NmtSeq2SeqEncoderMy(embedSize, hiddenSize, layersNum, len(sourceVocab), dropout = dropout);
        # self._decoder = NmtSeq2SeqDecoderMy(embedSize, hiddenSize, layersNum, len(targetVocab), dropout = dropout);
        self._encoder = NmtSeq2SeqEncoderMyBiRnn(embedSize, hiddenSize, layersNum, len(sourceVocab), dropout = dropout);
        self._decoder = NmtSeq2SeqDecoderMy(embedSize, 2 * hiddenSize, layersNum, len(targetVocab), dropout = dropout);
        self._bos = np.array([targetVocab["<bos>"]]).reshape(1, 1);

        super().__init__(self._encoder, self._decoder);


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        XLen, YLen, X, Y = data;
        X, Y = X.transpose(1, 0), Y.transpose(1, 0);
        Y = np.concatenate((np.tile(self._bos, (1, Y.shape[-1])), Y), axis = 0);

        XState, = self._encoder.forward(X);
        output, = self._decoder.forward(Y[: -1], XState);

        return output, YLen;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dOutput = dout[0];
        dY, dXState = self._decoder.backward(dOutput);
        dX, = self._encoder.backward(dXState);

        return dX, dY;


    def translate(self, sourceSentence : str, stepSize : int):
        self.reset();
        self._context.isTrainingMode = False;

        encoderX = np.array(truncatePad(self._sourceVocab[sourceSentence.split(" ")] + [self._sourceVocab["<eos>"]], stepSize, self._sourceVocab["<pad>"])).reshape(-1, 1);
        decoderState, = self._encoder.forward(encoderX);
        translateY = self._decoder.translate(decoderState, stepSize, self._targetVocab["<bos>"], self._targetVocab["<eos>"]);

        return " ".join(self._targetVocab.getTokens(translateY));


def testNmtEncoderDecoderMyGradient1():
    batchSize, stepSize = 32, 12,
    embedSize, hiddenSize, layersNum = 14, 18, 2;
    trainIterator, sourceVocab, targetVocab = loadDataNmt("/media/WindowsE/Data/ANKI/fra-eng/fra.txt", batchSize, stepSize, useTorch = False);

    XLen = np.random.choice(np.arange(stepSize), size = batchSize, replace = True);
    YLen = np.random.choice(np.arange(stepSize), size = batchSize, replace = True);
    X = np.random.choice(np.arange(len(sourceVocab)), size = (batchSize, stepSize), replace = True);
    Y = np.random.choice(np.arange(len(targetVocab)), size = (batchSize, stepSize), replace = True);
    m = NmtEncoderDecoderMy(embedSize, hiddenSize, layersNum, sourceVocab, targetVocab);
    m.context.isTrainingMode = True;
    YOutput, YLen = m.forward(XLen, YLen, X, Y);
    dX1, dY1 = m.backward(np.ones_like(YOutput));
    testModuleGradient(m, "NmtEncoderDecoderMy, numericGradient1", XLen, YLen, X, Y);
    print("\n");


def chapter9_Nmt_My(plot : bool = True):
    lr, maxEpoch = 0.005, 300;
    batchSize, stepSize = 64, 10;
    embedSize, hiddenSize, layersNum, dropout = 32, 32, 2, 0.1;
    trainIterator, sourceVocab, targetVocab = loadDataNmt("/media/WindowsE/Data/ANKI/fra-eng/fra.txt", batchSize, stepSize, exampleSize = 1000, useTorch = False);

    lossFunc = NN.SequenceSoftmaxWithCrossEntropy1DLoss();
    optimizer = NN.GradientsClipping(1.0, NN.Adam(lr = lr));
    model = NmtEncoderDecoderMy(embedSize, hiddenSize, layersNum, sourceVocab, targetVocab, dropout = dropout);

    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, plot = plot);
    with open("data/PyTorchTest/chapter9_Nmt_My.pkl", "wb") as file:
        pickle.dump(model.params, file);

    # with open("data/PyTorchTest/chapter9_Nmt_My.pkl", "rb") as file:
    #     params = pickle.load(file);
    # model.params = params;

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation = model.translate(eng, stepSize);
        print(f'{eng} => {translation}, bleu {bleuNLP(translation.split(" "), fra.split(" "), gramNum = 2):.3f}');


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex="all", sharey="all", squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix, cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
    plt.show(block = True);


def createData4KernelReg(trainSize : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def fKernelReg(x : np.ndarray) -> np.ndarray:
        return 2 * np.sin(x) + x ** 0.8;

    trainX = np.sort(np.random.rand(trainSize)) * 5;
    trainY = fKernelReg(trainX) + 0.5 * np.random.randn(trainSize);
    testX = np.arange(0, 5, 0.1);
    testY = fKernelReg(testX);

    return trainX, trainY, testX, testY;


def plotData4KernelReg(trainX : np.ndarray, trainY : np.ndarray, testX : np.ndarray, testY : np.ndarray, testYHat : np.ndarray):
    plt.figure();
    plt.plot(testX, testY, "-b", label = "True Value");
    plt.plot(testX, testYHat, "-r", label = "Predicate Value");
    plt.scatter(trainX, trainY, marker = "o");
    plt.legend(loc = 'upper right');
    plt.show(block = True);


class NWKernelRegressionMy(NN.NetModelBase):
    def __init__(self):
        super().__init__();
        self._name = "NWKernelRegressionMy";

        self._S = None;
        self._Q, self._K, self._V = None, None, None;

        self._w = np.array(np.random.rand());
        self._attentionWeight = None;
        self._params.append(NN.NetParamDefinition("width", self._w));


    @property
    def attentionWeight(self) -> np.ndarray:
        return self._attentionWeight;


    def _setParams(self, params: List[NN.INetParamDefinition]):
        self._w = params[0].value;


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        Q, K, V = data[: 3];

        Q = np.repeat(Q.reshape(-1, 1), K.shape[-1], axis = -1);
        self._S = (Q - K) * self._w;
        self._attentionWeight = softmax(self._S ** 2 * -0.5);
        Y = np.sum(self._attentionWeight * V, axis = -1);

        self._Q, self._K, self._V = Q, K, V;

        return Y, ;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dY = dout[0];
        dAttentionWeight = dY.reshape(-1, 1) * self._V;
        dS = softmaxGradient(self._attentionWeight, dAttentionWeight) * -1 * self._S;
        dw = np.sum(dS * (self._Q - self._K));

        self._params[0].grad[...] = dw;

        return np.zeros_like(self._Q), np.zeros_like(self._K), np.zeros_like(self._V);


def testNWKernelRegressionMyGradient1():
    batchSize, keySize = 320, 24,
    Q, K, V = np.random.randn(batchSize), np.random.randn(batchSize, keySize), np.random.randn(batchSize, keySize);

    m = NWKernelRegressionMy();
    m.context.isTrainingMode = True;
    Y, = m.forward(Q, K, V);
    m.backward(np.ones_like(Y));
    testModuleGradient(m, "NWKernelRegressionMy, numericGradient1", Q, K, V);
    print("\n");


def chapter10_kernelReg():
    trainSize = 50;
    trainX, trainY, testX, testY = createData4KernelReg(trainSize);

    # testYHat = np.ones_like(testY) * np.mean(trainY);

    attentionWeight1 = softmax(np.square(testX.reshape(-1, 1) - trainX.reshape(1, -1)) * -0.5);
    testYHat = (attentionWeight1 @ trainY.reshape(-1, 1)).flatten();
    show_heatmaps(np.expand_dims(np.expand_dims(attentionWeight1, axis = 0), axis = 0), "train inputs", "test inputs");
    plotData4KernelReg(trainX, trainY, testX, testY, testYHat);

    trainQ = trainX;
    trainK = np.tile(trainX, (trainSize, 1))[(1 - np.eye(trainSize)).astype(bool)].reshape(trainSize, -1);
    trainV = np.tile(trainY, (trainSize, 1))[(1 - np.eye(trainSize)).astype(bool)].reshape(trainSize, -1);

    lr, maxEpoch = 0.5, 150;
    trainIterator = NN.SequentialDataIterator([trainQ, trainK, trainV, trainY], batchSize = trainSize, shuffle = False);
    lossFunc = NN.IdentityWithMeanSquareLoss();
    optimizer = NN.SGD(lr);
    # optimizer = NN.Adam(lr);
    model = NWKernelRegressionMy();
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, plot = True);

    testQ = testX;
    testSize = len(testX);
    testK = np.tile(trainX, (testSize, 1));
    testV = np.tile(trainY, (testSize, 1));
    testYHat, = model.forward(testQ, testK, testV);
    show_heatmaps(np.expand_dims(np.expand_dims(model.attentionWeight, axis = 0), axis = 0), "train inputs", "test inputs");
    plotData4KernelReg(trainX, trainY, testX, testY, testYHat);


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequenceMask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def injectParams(torchParams : List[nn.Parameter], myModule : NN.INetModule, transposeFlags : List[bool] = None):
    if transposeFlags is None:
        transposeFlags = [True] * len(myModule.params);

    numpyParams = [NN.NetParamDefinition(mp.name, tp.data.T.numpy() if flag and len(tp.shape) > 1 else tp.data.numpy()) for tp, mp, flag in zip(torchParams, myModule.params, transposeFlags)];
    myModule.params = numpyParams;


def injectModelParams(torchModule : nn.Module, myModule : NN.INetModule, transposeFlags : List[bool] = None):
    injectParams(list(torchModule.parameters()), myModule, transposeFlags = transposeFlags);


def compareGrads(torchModule : nn.Module, myModule : NN.INetModule, transposeFlags : List[bool] = None):
    if transposeFlags is None:
        transposeFlags = [True] * len(myModule.params);

    messages = [];
    torchParams = list(torchModule.parameters());

    for i, (tp, mp, flag) in enumerate(zip(torchParams, myModule.params, transposeFlags)):
        tg, mg = tp.grad.T.numpy() if flag and len(tp.shape) > 1 else tp.grad.numpy(), mp.grad;
        messages.append(f"param {mp.name}{i}{mp.value.shape}, total value: {np.sum(np.fabs(mg - tg))}, mean value: {np.sum(np.fabs(mg - tg)) / mg.size}, error ratio: {np.linalg.norm(mg - tg) / (np.linalg.norm(mg) + np.linalg.norm(tg))}");

    print("\n".join(messages));


def testMaskedSoftmax():
    batchSize, queryNum, keyNum = 32, 8, 10;

    X = np.random.randn(batchSize, queryNum, keyNum);
    validLen = np.random.randint(1, keyNum + 1, batchSize);

    Y1 = softmax(X, getAttentionMaskByValidLength(queryNum, keyNum, validLen));
    Y2 = masked_softmax(torch.as_tensor(X), torch.as_tensor(validLen)).detach().numpy();

    print(f"testMaskedSoftmax, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    print("\n");


class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_q = nn.Linear(query_size, num_hiddens, bias = False)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


def testAdditiveAttentionModule():
    batchSize, queryNum, keyNum = 32, 8, 10;
    querySize, keySize, valueSize, hiddenSize, dropoutRatio = 11, 21, 31, 41, 0;
    torchModule = AdditiveAttention(keySize, querySize, hiddenSize, dropoutRatio);
    myModule = NN.AdditiveAttentionModule(querySize, keySize, hiddenSize, dropoutRatio = dropoutRatio);

    injectModelParams(torchModule, myModule);

    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    validLen = np.random.randint(1, keyNum + 1, batchSize);

    Y1, = myModule.forward(Q, K, V, getAttentionMaskByValidLength(queryNum, keyNum, validLen));
    myModule.backward(np.ones_like(Y1));

    Y2 = torchModule(torch.Tensor(Q), torch.Tensor(K), torch.Tensor(V), torch.Tensor(validLen));
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testAdditiveAttentionModule, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule);
    print("\n");


#@save
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def testDotProductAttentionModule():
    batchSize, queryNum, keyNum = 32, 8, 10;
    querySize, keySize, valueSize, dropoutRatio = 11, 11, 21, 0;
    torchModule = DotProductAttention(dropoutRatio);
    myModule = NN.DotProductAttentionModule(dropoutRatio = dropoutRatio);

    injectModelParams(torchModule, myModule);

    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    validLen = np.random.randint(1, keyNum + 1, batchSize);

    Y1, = myModule.forward(Q, K, V, getAttentionMaskByValidLength(queryNum, keyNum, validLen));
    Y2 = torchModule(torch.Tensor(Q), torch.Tensor(K), torch.Tensor(V), torch.Tensor(validLen)).detach().numpy();

    print(f"testDotProductAttentionModule, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    print("\n");


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def testMultiHeadAttentionModule():
    batchSize, queryNum, keyNum, headNum = 32, 8, 10, 8;
    querySize, keySize, valueSize, hiddenSize, dropoutRatio = 11, 11, 21, 31, 0;
    torchModule = MultiHeadAttention(keySize, querySize, valueSize, hiddenSize * headNum, headNum, dropoutRatio);
    myModule = NN.MultiHeadAttentionModule(NN.DotProductAttentionModule(dropoutRatio = dropoutRatio), querySize, keySize, valueSize, (hiddenSize, hiddenSize, hiddenSize, hiddenSize * headNum), headNum);

    injectModelParams(torchModule, myModule);

    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    validLen = np.random.randint(1, keyNum + 1, batchSize);

    Y1, = myModule.forward(Q, K, V, getAttentionMaskByValidLength(queryNum, keyNum, validLen));
    myModule.backward(np.ones_like(Y1));

    Y2 = torchModule(torch.Tensor(Q), torch.Tensor(K), torch.Tensor(V), torch.Tensor(validLen));
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testMultiHeadAttentionModule, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule);
    print("\n");


def testSelfAttentionModule():
    batchSize, headNum = 32, 6;
    sequenceLength, inputSize, hiddenSize, dropoutRatio = 64, 120, 20, 0;
    torchModule = MultiHeadAttention(inputSize, inputSize, inputSize, hiddenSize * headNum, headNum, dropoutRatio);
    myModule = NN.SelfAttentionModule(NN.MultiHeadAttentionModule(NN.DotProductAttentionModule(dropoutRatio = dropoutRatio), inputSize, inputSize, inputSize, (hiddenSize, hiddenSize, hiddenSize, hiddenSize * headNum), headNum));

    injectModelParams(torchModule, myModule);

    X = np.random.randn(batchSize, sequenceLength, inputSize);
    validLen = np.random.randint(1, sequenceLength + 1, batchSize);

    Y1, = myModule.forward(X, getAttentionMaskByValidLength(sequenceLength, sequenceLength, validLen));
    myModule.backward(np.ones_like(Y1));

    Y2 = torchModule(torch.Tensor(X), torch.Tensor(X), torch.Tensor(X), torch.Tensor(validLen));
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testSelfAttentionModule, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule);
    print("\n");


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


def testSinePositionalEncodingModule():
    batchSize, sequenceLength, inputSize, dropoutRatio = 32, 64, 28, 0;
    torchModule = PositionalEncoding(inputSize, dropoutRatio);
    myModule = NN.SinePositionalEncodingModule(inputSize, dropoutRatio = dropoutRatio);

    injectModelParams(torchModule, myModule);

    X = np.random.randn(batchSize, sequenceLength, inputSize);

    Y1, = myModule.forward(X);
    Y2 = torchModule(torch.Tensor(X)).detach().numpy();

    print(f"testSinePositionalEncodingModule, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    print("\n");


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


def testAffineLayer1():
    batchSize, inputSize, hiddenSize = 32, 28, 46;
    torchModule = nn.Linear(inputSize, hiddenSize);
    myModule = NN.AffineLayer(inputSize, hiddenSize);

    injectModelParams(torchModule, myModule);

    X = np.random.randn(batchSize, inputSize);

    Y1, = myModule.forward(X);
    myModule.backward(np.ones_like(Y1));

    Y2 = torchModule(torch.Tensor(X));
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testAffineLayer1, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule);
    print("\n");


def testAffineLayer2():
    batchSize, sequenceLength, inputSize, hiddenSize = 32, 64, 120, 240;
    torchModule = nn.Linear(inputSize, hiddenSize);
    myModule = NN.AffineLayer(inputSize, hiddenSize);

    injectModelParams(torchModule, myModule);

    X = np.random.randn(batchSize, sequenceLength, inputSize).astype(defaultDType);

    Y1, = myModule.forward(X);
    myModule.backward(np.ones_like(Y1));

    Y2 = torchModule(torch.Tensor(X));
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testAffineLayer2, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule);
    print("\n");


def testTransformerPositionwiseFFNModule():
    batchSize, sequenceLength, inputSize, hiddenSize = 32, 64, 28, 46;
    torchModule = PositionWiseFFN(inputSize, hiddenSize, inputSize);
    myModule = NN.TransformerPositionwiseFFNModule(inputSize, hiddenSize);

    injectModelParams(torchModule, myModule);

    X = np.random.randn(batchSize, sequenceLength, inputSize);

    Y1, = myModule.forward(X);
    myModule.backward(np.ones_like(Y1));

    Y2 = torchModule(torch.Tensor(X));
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testTransformerPositionwiseFFNModule, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule);
    print("\n");


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


def testTransformerAddNormalizationModule():
    batchSize, sequenceLength, inputSize, dropoutRatio = 32, 64, 28, 0.0;
    C = np.random.randn(batchSize, sequenceLength, inputSize);
    torchModule = AddNorm(inputSize, dropoutRatio);
    myModule = NN.SequentialContainer(
        NN.TransformerAddNormalizationModule(inputSize, dropoutRatio = dropoutRatio),
        NN.FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    injectModelParams(torchModule, myModule);

    X = np.random.randn(batchSize, sequenceLength, inputSize);
    F = np.random.randn(batchSize, sequenceLength, inputSize);

    Y1, = myModule.forward(X, F);
    myModule.backward(np.ones_like(Y1));

    Y2 = torchModule(torch.Tensor(X), torch.Tensor(F)) * torch.Tensor(C);
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testTransformerAddNormalizationModule, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule);
    print("\n");


class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


def testTransformerEncoderBlock():
    batchSize, headNum = 32, 6;
    sequenceLength, inputSize, attentionHiddenSize, ffnHiddenSize, dropoutRatio = 64, 120, 20, 30, 0;
    C = np.random.randn(batchSize, sequenceLength, inputSize).astype(defaultDType);
    torchModule = EncoderBlock(inputSize, inputSize, inputSize, inputSize, inputSize, inputSize, ffnHiddenSize, headNum, dropoutRatio);
    myModule = NN.SequentialContainer(
        NN.TransformerEncoderBlock(inputSize, attentionHiddenSize, ffnHiddenSize, inputSize, headNum = headNum, dropoutRatio = dropoutRatio),
        NN.FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    injectModelParams(torchModule, myModule);

    X = np.random.randn(batchSize, sequenceLength, inputSize).astype(defaultDType);
    validLen = np.random.randint(1, sequenceLength + 1, batchSize);

    Y1, = myModule.forward(X, getAttentionMaskByValidLength(sequenceLength, sequenceLength, validLen));
    myModule.backward(np.ones_like(Y1));

    Y2 = torchModule(torch.as_tensor(X), torch.as_tensor(validLen)) * torch.as_tensor(C);
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testTransformerEncoderBlock, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule);
    print("\n");


class TransformerEncoder(EncoderTorch):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


def testEmbeddingLayer():
    batchSize, vocabSize = 32, 240;
    sequenceLength, embeddingSize = 64, 120;
    C = np.random.randn(batchSize, sequenceLength, embeddingSize).astype(defaultDType);
    torchModule = nn.Embedding(vocabSize, embeddingSize);
    myModule = NN.SequentialContainer(
        NN.EmbeddingLayer(vocabSize, embeddingSize),
        NN.FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    transposeFlags = [False];
    injectModelParams(torchModule, myModule, transposeFlags);

    X = np.random.randint(0, vocabSize, (batchSize, sequenceLength));

    Y1, = myModule.forward(X);
    myModule.backward(np.ones_like(Y1));

    Y2 = torchModule(torch.as_tensor(X)) * torch.as_tensor(C);
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testEmbeddingLayer, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule, transposeFlags);
    print("\n");


def testTransformerEmbeddingEncoder():
    batchSize, vocabSize, headNum, blockNum = 32, 240, 6, 2;
    sequenceLength, embeddingSize, attentionHiddenSize, ffnHiddenSize, dropoutRatio = 64, 120, 20, 30, 0;
    C = np.random.randn(batchSize, sequenceLength, embeddingSize).astype(defaultDType);
    torchModule = TransformerEncoder(vocabSize, embeddingSize, embeddingSize, embeddingSize, embeddingSize, embeddingSize, embeddingSize, ffnHiddenSize, headNum, blockNum, dropoutRatio);
    myModule = NN.SequentialContainer(
        NN.TransformerEmbeddingEncoder(vocabSize, embeddingSize, attentionHiddenSize, ffnHiddenSize, embeddingSize, headNum = headNum, blockNum = blockNum, dropoutRatio = dropoutRatio),
        NN.FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    transposeFlags = [False] + [True] * (len(myModule.params) - 1);
    injectModelParams(torchModule, myModule, transposeFlags);

    X = np.random.randint(0, vocabSize, (batchSize, sequenceLength));
    validLen = np.random.randint(1, sequenceLength + 1, batchSize);

    Y1, = myModule.forward(X, validLen);
    myModule.backward(np.ones_like(Y1));

    Y2 = torchModule(torch.as_tensor(X), torch.as_tensor(validLen)) * torch.as_tensor(C);
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testTransformerEmbeddingEncoder, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule, transposeFlags);
    print("\n");


class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


def testTransformerDecoderBlock():
    batchSize, headNum = 32, 6;
    sequenceLength, inputSize, attentionHiddenSize, ffnHiddenSize, dropoutRatio = 64, 120, 20, 30, 0;
    C = np.random.randn(batchSize, sequenceLength, inputSize).astype(defaultDType);
    torchModule = DecoderBlock(inputSize, inputSize, inputSize, inputSize, inputSize, inputSize, ffnHiddenSize, headNum, dropoutRatio, 0);
    myModule = NN.SequentialContainer(
        NN.TransformerDecoderBlock(inputSize, inputSize, attentionHiddenSize, ffnHiddenSize, inputSize, headNum = headNum, dropoutRatio = dropoutRatio),
        NN.FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    myModule.context.isTrainingMode = True;

    injectModelParams(torchModule, myModule);

    decoderX = np.random.randn(batchSize, sequenceLength, inputSize).astype(defaultDType);
    encoderY = np.random.randn(batchSize, sequenceLength + 1, inputSize).astype(defaultDType);
    encoderValidLen = np.random.randint(1, sequenceLength + 1, batchSize);

    Y1, = myModule.forward(decoderX, decoderX, decoderX, encoderY, getAttentionMaskByValidLength(sequenceLength, sequenceLength + 1, encoderValidLen));
    myModule.backward(np.ones_like(Y1));

    Y2, state = torchModule(torch.as_tensor(decoderX), [torch.as_tensor(encoderY), torch.as_tensor(encoderValidLen), [None]]);
    Y2 *= torch.as_tensor(C);
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testTransformerDecoderBlock, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule);
    print("\n");


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def initState(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


def testTransformerEmbeddingDecoder():
    batchSize, vocabSize, headNum, blockNum = 32, 240, 6, 2;
    sequenceLength, embeddingSize, attentionHiddenSize, ffnHiddenSize, dropoutRatio = 64, 120, 20, 30, 0;
    C = np.random.randn(batchSize, sequenceLength, vocabSize).astype(defaultDType);
    torchModule = TransformerDecoder(vocabSize, embeddingSize, embeddingSize, embeddingSize, embeddingSize, embeddingSize, embeddingSize, ffnHiddenSize, headNum, blockNum, dropoutRatio);
    myModule = NN.SequentialContainer(
        NN.TransformerEmbeddingDecoder(vocabSize, embeddingSize, embeddingSize, attentionHiddenSize, ffnHiddenSize, embeddingSize, headNum = headNum, blockNum = blockNum, dropoutRatio = dropoutRatio),
        NN.AffineLayer(embeddingSize, vocabSize),
        NN.FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    myModule.context.isTrainingMode = True;

    transposeFlags = [False] + [True] * (len(myModule.params) - 1);
    injectModelParams(torchModule, myModule, transposeFlags);

    X = np.random.randint(0, vocabSize, (batchSize, sequenceLength));
    encoderY = np.random.randn(batchSize, sequenceLength + 1, embeddingSize).astype(defaultDType);
    encoderValidLen = np.random.randint(1, sequenceLength + 2, batchSize);

    Y1, = myModule.forward(X, encoderY, encoderValidLen);
    myModule.backward(np.ones_like(Y1));

    Y2, state = torchModule(torch.as_tensor(X), [torch.as_tensor(encoderY), torch.as_tensor(encoderValidLen), [None] * blockNum]);
    Y2 *= torch.as_tensor(C);
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testTransformerEmbeddingDecoder, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule, transposeFlags);
    print("\n");


def chapter10_Nmt_Transformer_PyTorch():
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, tryGPU()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = loadDataNmt("data/PyTorchTest/fra-eng/fra.txt", batch_size, num_steps, exampleSize = 1000, useTorch = True)

    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = EncoderDecoderTorch(encoder, decoder)

    # net.load_state_dict(torch.load("data/PyTorchTest/chapter10_Nmt_Transformer_PyTorch.params"));

    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device, plot = False)
    torch.save(net.state_dict(), "data/PyTorchTest/chapter10_Nmt_Transformer_PyTorch.params");
    with open("data/PyTorchTest/chapter10_Nmt_Transformer_PyTorch.pkl", "wb") as file:
        pickle.dump(list(net.parameters()), file);

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k = 2):.3f}');


class NmtTransformerEncoderDecoderMy(NN.NetModelBase):
    def __init__(self, embedSize : int, attentionHiddenSize: int, ffnHiddenSize: int, sourceVocab : Vocab, targetVocab : Vocab, headNum: int = 2, blockNum : int = 2, dropout : float = 0.0):
        self._sourceVocab = sourceVocab;
        self._targetVocab = targetVocab;
        self._blockNum = blockNum;
        self._encoder = NN.TransformerEmbeddingEncoder(len(sourceVocab), embedSize, attentionHiddenSize, ffnHiddenSize, embedSize, headNum = headNum, blockNum = blockNum, dropoutRatio = dropout);
        self._decoder = NN.TransformerEmbeddingDecoder(len(targetVocab), embedSize, embedSize, attentionHiddenSize, ffnHiddenSize, embedSize, headNum = headNum, blockNum = blockNum, dropoutRatio = dropout);
        self._dense = NN.AffineLayer(embedSize, len(targetVocab));
        self._bos = np.array([targetVocab["<bos>"]]).reshape(1, 1);

        super().__init__(self._encoder, self._decoder, self._dense);


    def forward(self, *data : np.ndarray) -> Tuple[np.ndarray, ...]:
        XLen, YLen, X, Y = data;
        Y = np.concatenate((np.tile(self._bos, (len(Y), 1)), Y), axis = -1);

        encoderY, = self._encoder.forward(X, XLen);
        decoderY, = self._decoder.forward(Y[..., : -1], encoderY, XLen);
        output, = self._dense.forward(decoderY);

        return output, YLen;


    def backward(self, *dout : np.ndarray) -> Tuple[np.ndarray, ...]:
        dOutput = dout[0];

        dDecoderY, = self._dense.backward(dOutput);
        dY, dEncoderY = self._decoder.backward(dDecoderY);
        dX, = self._encoder.backward(dEncoderY);

        return dX, dY;


    def translate(self, sourceSentence : str, stepSize : int):
        self.reset();
        self._context.isTrainingMode = False;

        sourceTokens = self._sourceVocab[sourceSentence.split(" ")] + [self._sourceVocab["<eos>"]];
        encoderValidLen = np.array([len(sourceTokens)]);

        # encoderX = np.array(truncatePad(sourceTokens, stepSize, self._sourceVocab["<pad>"])).reshape(1, -1);
        # encoderY, = self._encoder.forward(encoderX, encoderValidLen);

        encoderX = np.array(sourceTokens).reshape(1, -1);
        encoderY, = self._encoder.forward(encoderX);

        bosID, eosID = self._targetVocab["<bos>"], self._targetVocab["<eos>"];
        blockInputs = [None] * self._blockNum;
        decoderX, Y = np.array([bosID]).reshape(1, 1), [];
        for i in range(stepSize):
            decoderY, = self._decoder.predict(decoderX, encoderY, encoderValidLen, blockInputs = blockInputs);
            output, = self._dense.forward(decoderY);

            idx = int(np.argmax(output));
            if idx == eosID:
                break;

            Y.append(idx);
            decoderX = np.array([idx]).reshape(1, 1);

        return " ".join(self._targetVocab.getTokens(Y));


def testNmtTransformerEncoderDecoderMy1():
    batchSize, headNum, blockNum = 64, 4, 2;
    sequenceLength, embeddingSize, attentionHiddenSize, ffnHiddenSize, dropoutRatio = 10, 32, 8, 64, 0;

    train_iter, src_vocab, tgt_vocab = loadDataNmt("/media/WindowsE/Data/ANKI/fra-eng/fra.txt", batchSize, sequenceLength, exampleSize = 1000, useTorch = True)
    sourceVocabSize, targetVocabSize = len(src_vocab), len(tgt_vocab);

    encoder = TransformerEncoder(
        sourceVocabSize, embeddingSize, embeddingSize, embeddingSize, embeddingSize,
        embeddingSize, embeddingSize, ffnHiddenSize, headNum,
        blockNum, dropoutRatio)
    decoder = TransformerDecoder(
        targetVocabSize, embeddingSize, embeddingSize, embeddingSize, embeddingSize,
        embeddingSize, embeddingSize, ffnHiddenSize, headNum,
        blockNum, dropoutRatio)
    torchModule = EncoderDecoderTorch(encoder, decoder)

    myModule = NmtTransformerEncoderDecoderMy(embeddingSize, attentionHiddenSize, ffnHiddenSize, src_vocab, tgt_vocab, headNum = headNum, blockNum = blockNum, dropout = dropoutRatio);
    myModule.context.isTrainingMode = True;

    transposeFlags = [False] + [True] * (len(list(encoder.parameters())) - 1) + [False] + [True] * (len(list(decoder.parameters())) - 1);
    injectModelParams(torchModule, myModule, transposeFlags);

    encoderX = np.random.randint(0, sourceVocabSize, (batchSize, sequenceLength));
    decoderX = np.random.randint(0, targetVocabSize, (batchSize, sequenceLength + 1));
    encoderValidLen = np.random.randint(1, sequenceLength + 1, batchSize);
    decoderValidLen = np.random.randint(1, sequenceLength + 2, batchSize);

    Y1, _ = myModule.forward(encoderValidLen, decoderValidLen, encoderX, decoderX);
    myModule.backward(np.ones_like(Y1));

    Y2, _ = torchModule(torch.as_tensor(encoderX), torch.as_tensor(np.concatenate((np.tile(np.array([tgt_vocab["<bos>"]]).reshape(1, 1), (len(decoderX), 1)), decoderX), axis = -1)[..., : -1]), torch.as_tensor(encoderValidLen));
    torch.sum(Y2).backward();
    Y2 = Y2.detach().numpy();

    print(f"testNmtTransformerEncoderDecoderMy, Y error: {np.sum(Y1 - Y2)}, {np.linalg.norm(Y1 - Y2) / (np.linalg.norm(Y1) + np.linalg.norm(Y2))}");
    compareGrads(torchModule, myModule, transposeFlags);
    print("\n");


def testNmtTransformerEncoderDecoderMyGradient1():
    batchSize, headNum, blockNum = 2, 4, 2;
    sequenceLength, embeddingSize, attentionHiddenSize, ffnHiddenSize, dropoutRatio = 10, 32, 8, 64, 0;

    train_iter, src_vocab, tgt_vocab = loadDataNmt("/media/WindowsE/Data/ANKI/fra-eng/fra.txt", batchSize, sequenceLength, exampleSize = 1000, useTorch = True)
    sourceVocabSize, targetVocabSize = len(src_vocab), len(tgt_vocab);

    myModule = NmtTransformerEncoderDecoderMy(embeddingSize, attentionHiddenSize, ffnHiddenSize, src_vocab, tgt_vocab, headNum = headNum, blockNum = blockNum, dropout = dropoutRatio);
    myModule.context.isTrainingMode = True;

    encoderX = np.random.randint(0, sourceVocabSize, (batchSize, sequenceLength));
    decoderX = np.random.randint(0, targetVocabSize, (batchSize, sequenceLength + 1));
    encoderValidLen = np.random.randint(1, sequenceLength + 1, batchSize);
    decoderValidLen = np.random.randint(1, sequenceLength + 2, batchSize);

    Y, _ = myModule.forward(encoderValidLen, decoderValidLen, encoderX, decoderX);
    myModule.backward(np.ones_like(Y));

    testModuleGradient(myModule, "NmtTransformerEncoderDecoderMy, numericGradient1", encoderValidLen, decoderValidLen, encoderX, decoderX);
    print("\n");


def chapter10_Nmt_Transformer_My(plot : bool = True):
    lr, maxEpoch = 0.005, 300;
    batchSize, headNum, blockNum = 64, 4, 2;
    sequenceLength, embeddingSize, attentionHiddenSize, ffnHiddenSize, dropoutRatio = 10, 32, 8, 64, 0.1;
    trainIterator, sourceVocab, targetVocab = loadDataNmt("data/PyTorchTest/fra-eng/fra.txt", batchSize, sequenceLength, exampleSize = 1000, useTorch = False);

    lossFunc = NN.SequenceSoftmaxWithCrossEntropy1DLoss();
    optimizer = NN.GradientsClipping(1.0, NN.Adam(lr = lr));
    model = NmtTransformerEncoderDecoderMy(embeddingSize, attentionHiddenSize, ffnHiddenSize, sourceVocab, targetVocab, headNum = headNum, blockNum = blockNum, dropout = dropoutRatio);

    # with open("data/PyTorchTest/chapter10_Nmt_Transformer_PyTorch.pkl", "rb") as file:
    #     torchParams = pickle.load(file);
    # transposeFlags = [False] + [True] * 24 + [False] + [True] * 38;
    # injectParams(torchParams, model, transposeFlags);

    # model.fit(trainIterator, lossFunc, optimizer, maxEpoch, minEpoch = 2, plot = plot);
    # with open("data/PyTorchTest/chapter10_Nmt_Transformer_My.pkl", "wb") as file:
    #     pickle.dump(model.params, file);

    with open("data/PyTorchTest/chapter10_Nmt_Transformer_My.pkl", "rb") as file:
        params = pickle.load(file);
    model.params = params;

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation = model.translate(eng, sequenceLength);
        print(f'{eng} => {translation}, bleu {bleuNLP(translation.split(" "), fra.split(" "), gramNum = 2):.3f}');


def chapter11_plotRisk():
    def f(x):
        return x * torch.cos(torch.pi * x);

    def g(x):
        return f(x) + 0.2 * torch.cos(5 * torch.pi * x);

    xd = torch.arange(0.5, 1.5, 0.01);
    plt.figure();
    plt.plot(xd, f(xd), "-r");
    plt.plot(xd, g(xd), "--b");
    plt.show(block = True);


def chapter11_matmul():
    n = 1024;
    batchSize = 256;
    timer = Timer();
    A = np.zeros((n, n));
    B, C = np.random.randn(n, n), np.random.randn(n, n);

    timer.start();
    for i in range(n):
        for j in range(n):
            A[i, j] = B[i, :] @ C[:, j];
    timer.stop();

    timer.start();
    for j in range(n):
        A[:, j] = B @ C[:, j];
    timer.stop();

    timer.start();
    A = B @ C;
    timer.stop();

    timer.start();
    for j in range(0, n, batchSize):
        A[:, j: j + batchSize] = B @ C[:, j: j + batchSize];
    timer.stop();

    print(timer.times);
    print(timer.times[-2] / timer.times[-1]);


class SquareRootLrScheduler(NN.INetOptimizer):
    def __init__(self, net : NN.INetModel, optimizer : NN.INetOptimizer, lr : float):
        self._model = net;
        self._optimizer = optimizer;
        self._baseLr = lr;
        self._currentLr = lr;


    @property
    def learningRate(self) -> float:
        return self._optimizer.learningRate;


    @learningRate.setter
    def learningRate(self, value: float):
        self._optimizer.learningRate = value;


    def updateStep(self, params: List[NN.INetParamDefinition], context : NN.INetContext):
        lr = self._baseLr * math.pow(self._model.context.trainingEpoch + 1, -0.5);

        if self._currentLr != lr:
            self._optimizer.learningRate = lr;
            self._currentLr = lr;
            print(f"current LR: {lr}");

        self._optimizer.updateStep(params, context);


def testAny():
    x = np.arange(-10, 10 + 0.01, 0.01);
    p = scipy.stats.norm.cdf(x);

    y1 = x * p;
    y2 = NN.GeluLayer().forward(x)[0];
    y3 = nn.GELU("tanh").forward(torch.tensor(x)).numpy();

    plt.figure();
    plt.plot(x, y1, "k");
    plt.plot(x, y2, "r");
    # plt.plot(x, y3, "y");
    plt.show(block = True);


if __name__ == "__main__":
    # invalidFiles = [];
    #
    # for dirpath, dirnames, filenames in os.walk("/media/WindowsE/7/log/rtdb/"):
    #     for filename in filenames:
    #         filePath = os.path.join(dirpath, filename);
    #         x = np.load(filePath);
    #
    #         if np.all(x[:, 1] == 0):
    #             invalidFiles.append(filePath);
    #
    # print("\n".join(invalidFiles));


    # X = np.random.randn(24, 10);
    # T = np.random.randint(0, 10, 24);
    # Th = expand2OneHot(T, 10);
    #
    # l1 = crossEntropyError(softmax(X), Th);
    # l2 = F.cross_entropy(torch.Tensor(X), torch.Tensor(Th), reduction = "none");
    # l2 = nn.CrossEntropyLoss().forward(torch.Tensor(X), torch.Tensor(Th));
    # l3 = crossEntropyError1D(softmax(X), T);
    # l4 = F.cross_entropy(torch.Tensor(X), torch.Tensor(T).long(), reduction = "none");
    # l4 = nn.CrossEntropyLoss().forward(torch.Tensor(X), torch.Tensor(T).long());

    # loss1 = MaskedSoftmaxCELoss_PyTorch();
    # l1 = loss1(torch.ones(3, 4, 10), torch.ones((3, 4), dtype = torch.long), torch.tensor([4, 2, 0]));
    #
    # loss2 = MaskedSoftmaxCELoss_My();
    # l2 = loss2.forward(np.ones((3, 4, 10), dtype = np.float32), np.array([4, 2, 0], dtype = np.int32), np.ones((3, 4), dtype = np.int32));
    # dX, = loss2.backward();

    # batchSize, queryNum, keyNum = 2, 1, 10;
    # querySize, keySize, valueSize, hiddenSize = 20, 2, 4, 8;
    # Q = np.random.randn(batchSize, queryNum, querySize);
    # K = np.ones((batchSize, keyNum, keySize));
    # V = np.tile(np.arange(40, dtype = np.float32).reshape(1, keyNum, valueSize), (2, 1, 1));
    # validLen = np.array([2, 6]);

    # m1 = NN.AdditiveAttentionModule(querySize, keySize, hiddenSize);
    # Y1, = m1.forward(Q, K, V, validLen);
    # dQ1, dK1, dV1 = m1.backward(np.ones_like(Y1));
    # show_heatmaps(m1.attentionWeight.reshape(1, 1, batchSize, keyNum), "Keys", "Queries");
    #
    # Q = np.random.randn(batchSize, queryNum, keySize);
    # m2 = NN.DotProductAttentionModule();
    # Y2, = m2.forward(Q, K, V, validLen);
    # dQ2, dK2, dV2 = m2.backward(np.ones_like(Y2));
    # show_heatmaps(m2.attentionWeight.reshape(1, 1, batchSize, keyNum), "Keys", "Queries");

    # encoding_dim, num_steps = 32, 60;
    # pos_encoding = NN.SinePositionalEncodingModule(encoding_dim)
    # X, = pos_encoding.forward(np.zeros((num_steps, encoding_dim)));
    # P = pos_encoding.positionalEncoding[: X.shape[1]];
    #
    # plt.figure();
    # for i in range(6, 10):
    #     plt.plot(np.arange(num_steps), X[:, i].flatten(), label = f"Col {i}");
    # plt.legend(loc = 'upper right');
    # plt.show(block = True);
    #
    # show_heatmaps(np.expand_dims(np.expand_dims(X, axis = 0), axis = 0), "Columns", "Row Position");


    print("go.");

    # plotFitResult("data/chapter6_LeNet5_My.pkl");

    # chapter2_Preprocess();

    # chapter3_LR_PyTorch();
    # chapter3_LR_My();
    # chapter3_LR_MSE_MAE_Huber();
    # chapter3_SoftmaxR_PyTorch();
    # chapter3_SoftmaxR_My();
    # chapter3_SoftmaxR_My_SoftLabel();

    # chapter4_MLP_PyTorch();
    # chapter4_MLP_My();
    # chapter4_PR_underfitting_overfitting();
    # chapter4_Weight_Decay_PyTorch();
    # chapter4_Weight_Decay_My();
    # chapter4_Dropout_PyTorch();
    # chapter4_Dropout_My();
    # chapter4_Kaggle_HousePrices_LR_PyTorch();
    # chapter4_Kaggle_HousePrices_LR_My();
    # chapter4_Kaggle_HousePrices_LR_Predication_My();
    # chapter4_Kaggle_HousePrices_MLP_PyTorch();
    # chapter4_Kaggle_HousePrices_MLP_My();
    # chapter4_Kaggle_HousePrices_MLP_Predication_My();
    # chapter4_Kaggle_HousePrices_MLP_LogY_My();
    # chapter4_Kaggle_HousePrices_MLP_LogY_Predication_My();

    # chapter6_learnKernel();
    # chapter6_LeNet5_PyTorch();
    # chapter6_LeNet5_My(True);

    # chapter8_PredictSine_My(False);
    # chapter8_TimeMachine_Zero();
    # chapter8_TimeMachine_PyTorch();
    # chapter8_TimeMachine_My();
    # chapter9_Nmt_PyTorch();
    # testSequenceSoftmaxWithCrossEntropy1DLoss1();
    # testNmtSeq2SeqEncoderMyGradient1();
    # testNmtSeq2SeqEncoderMyBiRnnGradient1();
    # testNmtSeq2SeqDecoderMyGradient1();
    # testNmtEncoderDecoderMyGradient1();
    # chapter9_Nmt_My();

    # testNWKernelRegressionMyGradient1();
    # chapter10_kernelReg();
    # testMaskedSoftmax();
    # testAdditiveAttentionModule();
    # testDotProductAttentionModule();
    # testMultiHeadAttentionModule();
    # testSelfAttentionModule();
    # testSinePositionalEncodingModule();
    # testAffineLayer1();
    # testAffineLayer2();
    # testTransformerPositionwiseFFNModule();
    # testTransformerAddNormalizationModule();
    # testTransformerEncoderBlock();
    # testEmbeddingLayer();
    # testTransformerEmbeddingEncoder();
    # testTransformerDecoderBlock();
    testTransformerEmbeddingDecoder();
    # chapter10_Nmt_Transformer_PyTorch();
    # testNmtTransformerEncoderDecoderMy1();
    # testNmtTransformerEncoderDecoderMyGradient1();
    # chapter10_Nmt_Transformer_My(True);

    # chapter11_plotRisk();
    # chapter11_matmul();

    # testAny();

    print("exit.");
