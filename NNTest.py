import re;
import abc;
import math;
import time;
import pickle;
import collections;
import matplotlib.pyplot as plt;
import mpl_toolkits.mplot3d as p3d;

from typing import List, Tuple, Callable, Any;
from sklearn.utils.extmath import randomized_svd;

from NN import *;
from MNIST import *;


class EmbeddingLayer(INetModule):
    def __init__(self, inputSize : int, outputSize : int, W : np.ndarray = None):
        self._X = None;
        self._shape = None;
        self._inputSize = inputSize;
        self._outputSize = outputSize;
        self._isTrainingMode = True;

        self._weight = 0.01 * np.random.randn(inputSize, outputSize) if W is None else W;
        self._params = [self._weight];
        self._grads = [np.zeros_like(self._weight)];


    def __repr__(self):
        return self.__str__();


    def __str__(self):
        return f"Embedding {self._inputSize}*{self._outputSize}";


    @property
    def weight(self):
        return self._weight;


    @property
    def isTrainingMode(self) -> bool:
        return self._isTrainingMode;


    @isTrainingMode.setter
    def isTrainingMode(self, value: bool):
        self._isTrainingMode = value;


    @property
    def params(self):
        return self._params;


    @property
    def grads(self):
        return self._grads;


    def forward(self, X : np.ndarray) -> np.ndarray:
        self._shape = X.shape;
        self._X = X.flatten();

        out = self._weight[self._X];

        return out;


    def backward(self, dout : np.ndarray) -> np.ndarray:
        dW = self._grads[0];
        dW[...] = 0;

        np.add.at(dW, self._X, dout);

        return np.zeros(self._shape);


class CBOW(INetModule):
    def __init__(self, windowSize : int, vocabSize : int, hiddenSize : int):
        self._windowSize = windowSize;
        self._vocabSize = vocabSize;
        self._hiddenSize = hiddenSize;
        self._W0 = 0.01 * np.random.randn(vocabSize, hiddenSize);
        self._W1 = 0.01 * np.random.randn(hiddenSize, vocabSize);

        self._hiddenLayers = [AffineLayer(vocabSize, hiddenSize, includeBias = False, W = self._W0) for i in range(2 * windowSize)];
        self._outputLayer = AffineLayer(hiddenSize, vocabSize, includeBias = False, W = self._W1);
        self._isTrainingMode = True;
        self._params = [self._W0, self._W1];
        self._grads = [np.zeros_like(self._W0), np.zeros_like(self._W1)];

        self.isTrainingMode = True;


    @property
    def isTrainingMode(self) -> bool:
        return self._isTrainingMode;


    @isTrainingMode.setter
    def isTrainingMode(self, value: bool):
        self._isTrainingMode = value;

        for layer in self._hiddenLayers:
            layer.isTrainingMode = value;
        self._outputLayer.isTrainingMode = value;


    @property
    def params(self) -> List[np.ndarray]:
        return self._params;


    @property
    def grads(self) -> List[np.ndarray]:
        return self._grads;


    @property
    def wordVector(self):
        return self._W0;


    def forward(self, X : np.ndarray) -> np.ndarray:
        N, H, W = X.shape;
        Y = np.zeros((N, self._hiddenSize));

        for i in range(H):
            Y += self._hiddenLayers[i].forward(X[:, i, :]);

        Y /= H;
        out = self._outputLayer.forward(Y);

        return out;


    def backward(self, dout : np.ndarray) -> np.ndarray:
        N, H, W = len(dout), 2 * self._windowSize, self._vocabSize;
        dX = np.zeros((N, H, W));
        dS = self._outputLayer.backward(dout);

        self._grads[0][...] = 0;
        self._grads[1][...] = self._outputLayer.grads[0];

        for i in range(H):
            layer = self._hiddenLayers[i];
            dX[:, i, :] = layer.backward(dS / H);
            self._grads[0] += layer.grads[0];

        return dX;


class SkipGram(INetModule):
    def __init__(self, windowSize : int, vocabSize : int, hiddenSize : int):
        self._params = [];
        self._grads = [];
        self._isTrainingMode = True;
        self._windowSize = windowSize;
        self._vocabSize = vocabSize;
        self._hiddenSize = hiddenSize;

        self._hiddenLayer = AffineLayer(vocabSize, hiddenSize, includeBias = False);
        self._outputLayer = AffineLayer(hiddenSize, vocabSize, includeBias = False);
        for layer in [self._hiddenLayer, self._outputLayer]:
            self._params.extend(layer.params);
            self._grads.extend(layer.grads);

        self.isTrainingMode = True;


    @property
    def isTrainingMode(self) -> bool:
        return self._isTrainingMode;


    @isTrainingMode.setter
    def isTrainingMode(self, value: bool):
        self._isTrainingMode = value;

        self._hiddenLayer.isTrainingMode = value;
        self._outputLayer.isTrainingMode = value;


    @property
    def params(self) -> List[np.ndarray]:
        return self._params;


    @property
    def grads(self) -> List[np.ndarray]:
        return self._grads;


    @property
    def wordVector(self):
        return self._hiddenLayer.params[0];


    def forward(self, X : np.ndarray) -> np.ndarray:
        Y = self._hiddenLayer.forward(X);
        Z = self._outputLayer.forward(Y);

        out = np.expand_dims(Z, 1);
        out = np.repeat(out, 2 * self._windowSize, 1);

        return out;


    def backward(self, dout : np.ndarray) -> np.ndarray:
        dZ = dout.sum(1);
        dY = self._outputLayer.backward(dZ);
        dX = self._hiddenLayer.backward(dY);

        return dX;


class MultiLayerNet:
    def __init__(self, inputSize, hiddenSizes, outputSize, activeLayerType = ReluLayer, lastLayerType = SoftmaxWithCrossEntropyLoss, initStd = None, initCoef = None, useBatchNormalization = False, weightDecayLambda: float = 0, useDropout = False, dropoutRatio = 0.5):
        if initStd is None and initCoef is None:
            initCoef = math.sqrt(2);

        self.__inputSize = inputSize;
        self.__hiddenSizes = hiddenSizes;
        self.__outputSize = outputSize;
        self.__useBatchNormalization = useBatchNormalization;
        self.__weightDecayLambda = weightDecayLambda;
        self.__useDropout = useDropout;

        self.params = {};
        self.__layers = collections.OrderedDict();

        allSizes = hiddenSizes + [outputSize];
        for i in range(len(allSizes)):
            currentSize = allSizes[i];
            previousSize = inputSize if i == 0 else allSizes[i - 1];

            self.params["W{0}".format(i + 1)] = (W := self.__initWeight((previousSize, currentSize), initStd, initCoef));
            self.params["b{0}".format(i + 1)] = (b := np.zeros(currentSize));
            self.__layers["Affine{0}".format(i + 1)] = AffineLayer(W, b);

            if i < len(allSizes) - 1:
                if useBatchNormalization:
                    self.params["gamma{0}".format(i + 1)] = (gamma := np.ones(currentSize));
                    self.params["beta{0}".format(i + 1)] = (beta := np.zeros(currentSize));
                    self.__layers["BatchNormalization{0}".format(i + 1)] = BatchNormalizationLayer(gamma, beta);

                self.__layers["Activation{0}".format(i + 1)] = activeLayerType();

                if useDropout:
                    self.__layers["Dropout{0}".format(i + 1)] = DropoutLayer(dropoutRatio);

        self.__lastLayer = lastLayerType();

        backLayers = list(self.__layers.values());
        backLayers.reverse();
        self.__backLayers = backLayers;


    def __initWeight(self, shape, initStd, initCoef):
        return np.random.randn(*shape) * (initCoef / math.sqrt(shape[0]) if initCoef is not None else initStd);


    def predict(self, X, isTraining):
        Y = X;
        for layer in self.__layers.values():
            Y = layer.forward(Y, isTraining);
        return Y;


    def loss(self, X, T, isTraining):
        weightDecay = 0;

        if self.__weightDecayLambda != 0:
            for i in range(len(self.__hiddenSizes) + 1):
                weightDecay += 0.5 * self.__weightDecayLambda * np.square(self.params["W{0}".format(i + 1)]).sum();

        return self.__lastLayer.forward(self.predict(X, isTraining), T, isTraining) + weightDecay;


    def predictWithdX(self, X):
        Y = self.predict(X, False);

        dout = np.ones((X.shape[0], self.__outputSize));
        for layer in self.__backLayers:
            dout = layer.backward(dout);

        return Y, dout;


    def gradient(self, X, T, isTraining):
        loss = self.loss(X, T, isTraining);

        dout = self.__lastLayer.backward(1);
        for layer in self.__backLayers:
            dout = layer.backward(dout);

        gradients = {};
        hiddenLayersNum = len(self.__hiddenSizes);
        for i in range(hiddenLayersNum + 1):
            gradients["W{0}".format(i + 1)] = self.__layers["Affine{0}".format(i + 1)].dW + self.__weightDecayLambda * self.params["W{0}".format(i + 1)];
            gradients["b{0}".format(i + 1)] = self.__layers["Affine{0}".format(i + 1)].db;

            if self.__useBatchNormalization and i < hiddenLayersNum:
                gradients["gamma{0}".format(i + 1)] = self.__layers["BatchNormalization{0}".format(i + 1)].dGamma;
                gradients["beta{0}".format(i + 1)] = self.__layers["BatchNormalization{0}".format(i + 1)].dBeta;


        return gradients, loss;


    def accuracy(self, X, T):
        Y = self.predict(X, False);
        return (Y.argmax(1) == T.argmax(1)).sum() / float(X.shape[0]);


class SimpleCNN:
    def __init__(self, inputDim, convParams, poolingParams, hiddenSize, outputSize):
        self.params = {};
        self._layers = collections.OrderedDict();

        C, H, W = inputDim;
        FN, FH, FW, convStride, convPad = convParams;
        PH, PW, poolingStride, poolingPad = poolingParams;

        self.params["convW1"] = (convW := np.random.randn(FN, C, FH, FW) * math.sqrt(2) / math.sqrt(C * H * W));
        # self.params["convW1"] = (convW := np.load("W1.npy"));
        self.params["convb1"] = (convb := np.zeros(FN));
        self._layers["conv"] = ConvolutionLayer(convW, convb, convStride, convPad);
        self._layers["relu0"] = ReluLayer();
        OH = convOutputSize(H, FH, convStride, convPad);
        OW = convOutputSize(W, FW, convStride, convPad);

        self._layers["pooling"] = MaxPoolingLayer(PH, PW, poolingStride, poolingPad);
        OH = convOutputSize(OH, PH, poolingStride, poolingPad);
        OW = convOutputSize(OW, PW, poolingStride, poolingPad);

        self.params["W1"] = (W1 := np.random.randn(FN * OH * OW, hiddenSize) * math.sqrt(2) / math.sqrt(FN * OH * OW));
        # self.params["W1"] = (W1 := np.load("W2.npy"));
        self.params["b1"] = (b1 := np.zeros(hiddenSize));
        self._layers["affine1"] = AffineLayer(W1, b1);
        self._layers["relu1"] = ReluLayer();

        self.params["W2"] = (W2 := np.random.randn(hiddenSize, outputSize) * math.sqrt(2) / hiddenSize);
        # self.params["W2"] = (W2 := np.load("W3.npy"));
        self.params["b2"] = (b2 := np.zeros(outputSize));
        self._layers["affine2"] = AffineLayer(W2, b2);

        self._lastLayer = SoftmaxWithLossLayer();

        backLayers = list(self._layers.values());
        backLayers.reverse();
        self.__backLayers = backLayers;


    def _internalPredict(self, X, isTraining):
        Y = X;
        for layer in self._layers.values():
            Y = layer.forward(Y, isTraining);
        return Y;


    def predict(self, X, isTraining, batchSize = 10000):
        N = X.shape[0];

        if N <= batchSize:
            return self._internalPredict(X, isTraining);

        values = [];
        for i in range(0, N, batchSize):
            values.append(self._internalPredict(X[i: i + batchSize], isTraining));
        if N % batchSize != 0:
            values.append(self._internalPredict(X[N // batchSize * batchSize:], isTraining));

        return np.vstack(tuple(values));


    def accuracy(self, X, T, batchSize = 10000):
        value = 0.0;
        N = X.shape[0];

        for i in range(0, N, batchSize):
            value += (self._internalPredict(X[i: i + batchSize], False).argmax(1) == T[i: i + batchSize].argmax(1)).sum();
        if N % batchSize != 0:
            value += (self._internalPredict(X[N // batchSize * batchSize:], False).argmax(1) == T[N // batchSize * batchSize:].argmax(1)).sum();

        return value / N;


    def loss(self, X, T, isTraining):
        return self._lastLayer.forward(self.predict(X, isTraining), T, isTraining);


    def numericGradient(self, X, T, isTraining):
        f = lambda x: self.loss(X, T, isTraining);

        gradients = {};
        gradients["convW1"] = numericGradient(f, self.params["convW1"]);
        gradients["convb1"] = numericGradient(f, self.params["convb1"]);
        gradients["W1"] = numericGradient(f, self.params["W1"]);
        gradients["b1"] = numericGradient(f, self.params["b1"]);
        gradients["W2"] = numericGradient(f, self.params["W2"]);
        gradients["b2"] = numericGradient(f, self.params["b2"]);

        return gradients;


    def gradient(self, X, T, isTraining):
        loss = self.loss(X, T, isTraining);

        dout = self._lastLayer.backward(1);
        for layer in self.__backLayers:
            dout = layer.backward(dout);

        gradients = {};
        gradients["convW1"] = self._layers["conv"].dW;
        gradients["convb1"] = self._layers["conv"].db;
        gradients["W1"] = self._layers["affine1"].dW;
        gradients["b1"] = self._layers["affine1"].db;
        gradients["W2"] = self._layers["affine2"].dW;
        gradients["b2"] = self._layers["affine2"].db;
        # print("b1: {0}\r\n".format(gradients["convb1"]));
        #
        # numGradient = self.numericGradient(X, T, isTraining);
        # print("b1: {0}\r\n".format(numGradient["convb1"]));
        #
        # for key in gradients.keys():
        #     print("diff of {0}: {1}".format(key, np.abs(gradients[key] - numGradient[key]).mean()));
        #
        # raise ValueError("exit test");

        return gradients, loss;


def testFNN(optimizer, initStd = None, initCoef = None, useBatchNormalization = False, weightDecayLambda: float = 0, useDropout = False, dropoutRatio = 0.5):
    mnist = MNIST.MNIST("/media/WindowsE/Data/MNIST", True, True);
    network = MultiLayerNet(mnist.trainX.shape[1], [101, 102, 103, 104, 105, 106], 10, initStd = initStd, initCoef = initCoef, useBatchNormalization = useBatchNormalization, weightDecayLambda = weightDecayLambda, useDropout = useDropout, dropoutRatio = dropoutRatio);

    batchSize = 100;
    iterationNumber = 2000;
    trainX, trainY = mnist.trainX[:300, :], mnist.trainY[:300, :];
    testX, testY = mnist.testX, mnist.testY;
    n = trainX.shape[0];
    epoch = max(n / batchSize, 1);

    trainLossData = [];
    testAccuracyData = [];
    trainAccuracyData = [];

    for i in range(iterationNumber):
        indices = np.random.choice(n, batchSize, False);
        X = trainX[indices, :];
        T = trainY[indices, :];

        # network.gradientCheck(X, T);

        gradients, lossValue = network.gradient(X, T, True);
        optimizer.update(network.params, gradients);

        trainLossData.append(lossValue);
        print("loss value: {0}".format(lossValue));

        if i % epoch == 0:
            testAccuracyData.append(testAccuracy := network.accuracy(testX, testY));
            trainAccuracyData.append(trainAccuracy := network.accuracy(trainX, trainY));
            print("test accuracy: {0}, train accuracy: {1}".format(testAccuracy, trainAccuracy));

    testAccuracyData.append(testAccuracy := network.accuracy(testX, testY));
    trainAccuracyData.append(trainAccuracy := network.accuracy(trainX, trainY));
    print("test accuracy: {0}, train accuracy: {1}".format(testAccuracy, trainAccuracy));

    print("exit.")

    return trainLossData, trainAccuracyData, testAccuracyData;


def testCNN(optimizer, convParams, poolingParams, hiddenSize):
    mnist = MNIST.MNIST("/media/WindowsE/Data/Fashion-MNIST", False, True);
    N, C, H, W = mnist.trainX.shape;
    network = SimpleCNN((C, H, W), convParams, poolingParams, hiddenSize, mnist.trainY.shape[1]);

    batchSize = 100;
    iterationNumber = 4000;
    trainX, trainY = mnist.trainX[:], mnist.trainY[:];
    testX, testY = mnist.testX, mnist.testY;
    n = trainX.shape[0];
    epoch = max(n / batchSize, 1);

    trainLossData = [];
    testAccuracyData = [];
    trainAccuracyData = [];

    for i in range(iterationNumber):
        indices = np.random.choice(n, batchSize, False);
        # indices = np.arange(100);
        X = trainX[indices];
        T = trainY[indices];

        gradients, lossValue = network.gradient(X, T, True);
        optimizer.update(network.params, gradients);

        trainLossData.append(lossValue);
        print("loss value: {0}".format(lossValue));

        if i % epoch == 0:
            testAccuracyData.append(testAccuracy := network.accuracy(testX, testY));
            trainAccuracyData.append(trainAccuracy := network.accuracy(trainX, trainY));
            print("test accuracy: {0}, train accuracy: {1}".format(testAccuracy, trainAccuracy));

    testAccuracyData.append(testAccuracy := network.accuracy(testX, testY));
    trainAccuracyData.append(trainAccuracy := network.accuracy(trainX, trainY));
    print("test accuracy: {0}, train accuracy: {1}".format(testAccuracy, trainAccuracy));

    print("exit.")

    return trainLossData, trainAccuracyData, testAccuracyData;


def trainFNN(net : INetModule, lossFunc : INetLoss, optimizer : INetOptimizer, dataIterator : DataIterator, epoch : int):
    lossData = [];
    lossValues = [];

    net.isTrainingMode = True;

    for i in range(epoch):
        lossData.clear();

        for X, T in dataIterator:
            Y = net.forward(X);
            loss = lossFunc.forward(Y, T);
            lossData.append(loss);

            net.backward(lossFunc.backward());
            optimizer.update(net.params, net.grads);

        lossValues.append(sum(lossData) / len(lossData));
        print(f"epoch {i + 1}, loss: {lossValues[-1]}");

    return lossValues;


def preprocess(text : str):
    text = text.lower();
    text = text.replace(".", " .");
    words = text.split(" ");

    word2ID, id2Word = {}, {};

    for word in words:
        if word in word2ID:
            continue;

        word2ID[word] = (wID := len(word2ID));
        id2Word[wID] = word;

    corpus = np.array([word2ID[w] for w in words]);

    return corpus, word2ID, id2Word;


def createCoMatrix(corpus, vocabSize : int, windowSize : int = 1):
    corpusSize = len(corpus);
    C = np.zeros((vocabSize, vocabSize), dtype = np.int32);

    for index, wordID in enumerate(corpus):
        for offset in range(1, windowSize + 1):
            leftIndex = index - offset;
            rightIndex = index + offset;

            if leftIndex >= 0:
                C[wordID, corpus[leftIndex]] += 1;
            if rightIndex < corpusSize:
                C[wordID, corpus[rightIndex]] += 1;

    return C;


def cosSimilarity1D(x, y, epsilon = 1e-8):
    nx = x / (math.sqrt(np.dot(x, x)) + epsilon);
    ny = y / (math.sqrt(np.dot(y, y)) + epsilon);
    return np.dot(nx, ny);


def cosSimilarity2D(x, Y, epsilon = 1e-8):
    nx = x / (math.sqrt(np.dot(x, x)) + epsilon);
    nY = Y / (np.sqrt((Y ** 2).sum(1, keepdims = True)) + epsilon);
    return nx @ nY.T;


def mostSimilarity(word, word2ID, id2Word, C, top = 5):
    x = C[word2ID[word]];
    similarity = cosSimilarity2D(x, C);
    return [(id2Word[i], similarity[i]) for i in (-similarity).argsort()[1: top + 1]];


def ppmi(C, epsilon = 1e-8):
    N = C.sum();
    S = C.sum(0);

    M = C / S.reshape((len(S), 1));
    M = M / S;
    M = M * N;

    return np.maximum(0, np.log2(M + epsilon));


def createContextsAndTarget(corpus, windowSize = 1):
    contexts = [];
    target = corpus[windowSize: -windowSize];

    for i in range(windowSize, len(corpus) - windowSize, 1):
        cs = corpus[i - windowSize: i + windowSize + 1].tolist();
        cs.pop(windowSize);

        contexts.append(cs);

    return np.array(contexts), target;


# def test():
#     X = np.arange(0, 2 * math.pi + 0.1, 0.1).reshape(-1, 1);
#     T = np.sin(X) + np.random.randn(*X.shape) * 0;
#
#     iterationNumber = 2000;
#     optimizer = Adam(0.01);
#     network = MultiLayerNet(1, [100], 1, lastLayerType=IdentityWithLossLayer, useBatchNormalization=True);
#
#     trainLossData = [];
#
#     for k in range(iterationNumber):
#         gradients, lossValue = network.gradient(X, T, True);
#
#         optimizer.update(network.params, gradients);
#         trainLossData.append(lossValue);
#         print("loss value: {0}".format(lossValue));
#
#     Y, dX = network.predictWithdX(X);
#     dX2 = np.cos(X);
#     print(np.abs(dX.flatten() - dX2.flatten()).sum() / dX.shape[0]);
#
#     plt.figure(1, (12, 8));
#     plt.plot(X.flatten(), T.flatten(), "-xk", label = "sin(x)");
#     plt.plot(X.flatten(), Y.flatten(), "-or", label="fitted value");
#     plt.legend(loc='upper right');
#     plt.show(block=True);
#     plt.close();


# def test():
#     key = "xy";
#     paths = [];
#     gradients = {};
#     params = {key: np.array([-7.0, 2.0])};
#     # optimizer = SGD(0.9);
#     # optimizer = Momentum(0.9, 0.5);
#     # optimizer = Nesterov(0.1, 0.9);
#     # optimizer = AdaGrad(0.9);
#     # optimizer = RMSprop(0.9, 0.5);
#     optimizer = Adam(0.9);
#
#     for i in range(100):
#         paths.append((w := params[key]).tolist());
#         gradients[key] = np.array([w[0] / 10, 2 * w[1]]);
#         optimizer.update(params, gradients);
#     paths.append(params[key].tolist());
#
#     data = np.array(paths);
#
#     plt.figure(1, (12, 8));
#     plt.xlim(-10, 10);
#     plt.ylim(-4, 4);
#     plt.plot(data[:, 0], data[:, 1], "-ok");
#     plt.show(block=True);
#     plt.close();


def loadSpiral(N = 1000, C = 3):
    np.random.seed(int(time.time()));

    X = np.zeros((N * C, 2));
    T = np.zeros((N * C, C));

    for j in range(C):
        for i in range(N):
            r = i * 1.0 / N;
            idx = j * N + i;
            theta = 4.0 * (r + j) + np.random.randn() * 0.2;

            X[idx] = np.array([r * math.cos(theta), r * math.sin(theta)]);
            T[idx, j] = 1;

    return X, T;


# def test():
#     N, C = 100, 3;
#     X, T = loadSpiral(N, C);
#
#     lr = 1;
#     net = SequentialContainer(
#         AffineLayer(2, 10),
#         SigmoidLayer(),
#         AffineLayer(10, 3),
#     );
#
#     lossValues = trainFNN(net, SoftmaxWithCrossEntropyLoss(), SGD(lr), DataIterator([X, T], 30), 1500);
#     plt.figure(1, (12, 8));
#     plt.plot(lossValues, "-");
#     plt.show(block=True);
#     plt.close();
#
#     h = 0.001;
#     xMin, xMax = X[:, 0].min(), X[:, 0].max();
#     yMin, yMax = X[:, 1].min(), X[:, 1].max();
#     xM, yM = np.meshgrid(np.arange(xMin - h, xMax + 2 * h, h), np.arange(yMin - h, yMax + 2 * h, h));
#     scores = net.forward(np.vstack((xM.flatten(), yM.flatten())).T);
#     zM = scores.argmax(1).reshape(xM.shape);
#
#     plt.figure(1, (12, 8));
#     markers = ["o", "*", "+"];
#     plt.contourf(xM, yM, zM);
#     for i in range(C):
#         plt.scatter(X[i * N: (i + 1) * N, 0].flatten(), X[i * N: (i + 1) * N, 1].flatten(), marker = markers[i], label = f"class {i + 1}");
#     plt.legend(loc='upper right');
#     plt.show(block=True);
#     plt.close();
#
#     print("exit.");


def testSpiral():
    C = 3;
    X, T = loadSpiral(10000, C);

    markers, colors = ["x", "*", "+", "s", "d"], ["b", "k", "g", "y", "r"];

    plt.figure(1);
    for j in range(C):
        plt.scatter(X[T[:, j] == 1, 0].get(), X[T[:, j] == 1, 1].get(), marker = markers[j], color = colors[j]);
    plt.show(block = True);
    plt.close();

    model = SequentialContainer(
        AffineLayer(X.shape[1], 10),
        ReluLayer(),
        AffineLayer(10, C),
    );
    lossFunc = SoftmaxWithCrossEntropyLoss();
    optimizer = Adam();
    iterator = DataIterator([X, T]);
    evaluator = ClassifierAccuracyEvaluator();

    trainer = NetTrainer(model, lossFunc, optimizer, evaluator);
    trainer.train(200, iterator);
    trainer.plot();


def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show(block = True);
    plt.close();


def test():
    mnist = MNIST("/media/WindowsE/Data/Fashion-MNIST", flatten = False);

    model = SequentialContainer(
        ConvolutionLayer(30, 1, 5, 5),
        ReluLayer(),
        MaxPoolingLayer(2, 2, 2),
        AffineLayer(30 * 12 * 12, 100, W = math.sqrt(2 / (30 * 12 * 12)) * np.random.randn(30 * 12 * 12, 100)),
        ReluLayer(),
        # DropoutLayer(),
        AffineLayer(100, 10, W = math.sqrt(2 / 100) * np.random.randn(100, 10)),
    );
    lossFunc = SoftmaxWithCrossEntropyLoss();
    optimizer = Adam();
    trainIterator = DataIterator([mnist.trainX, mnist.trainY]);
    testIterator = DataIterator([mnist.testX, mnist.testY], shuffle = False);
    evaluator = ClassifierAccuracyEvaluator();

    filter_show(model.modules[0].weight.get());

    trainer = NetTrainer(model, lossFunc, optimizer, evaluator);
    trainer.train(20, trainIterator, testIterator);
    trainer.plot();

    filter_show(model.modules[0].weight.get());

    # plt.figure(1);
    # for j in range(C):
    #     plt.scatter(X[T[:, j] == 1, 0].get(), X[T[:, j] == 1, 1].get(), marker = markers[j], color = colors[j]);
    # plt.show(block = True);
    # plt.close();


    # text = "you say goodbye and i say hello.";
    # corpus, word2ID, id2Word = preprocess(text);
    # vocabSize = len(word2ID);
    # contexts, target = createContextsAndTarget(corpus);
    # contexts = convert2OneHot(contexts, vocabSize);
    # target = convert2OneHot(target, vocabSize);
    #
    # # ptb = PTB.PTB("/media/WindowsE/Data/PTB");
    # # vocabSize = len(ptb.word2ID);
    # # contexts, target = createContextsAndTarget(ptb.trainCorpus);
    # # contexts = convert2OneHot(contexts, vocabSize);
    # # target = convert2OneHot(target, vocabSize);
    #
    # net = CBOW(1, vocabSize, 5);
    # # net = SkipGram(1, vocabSize, 5);
    #
    # lossValues = trainFNN(net, SoftmaxWithCrossEntropyLoss(), Adam(), DataIterator([contexts, target], 3), 2000);
    # plt.figure(1, (12, 8));
    # plt.plot(lossValues, "-");
    # plt.show(block = True);
    # plt.close();
    #
    # print(net.forward(contexts));

    print("exit.");
