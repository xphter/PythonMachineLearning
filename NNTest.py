import re;
import abc;
import math;
import time;
import pickle;
import collections;

import cupy as cp;
import numpy as np;
import scipy.stats;
import matplotlib.pyplot as plt;
import mpl_toolkits.mplot3d as p3d;

from typing import List, Tuple, Callable, Any;

import torch
from sklearn.utils.extmath import randomized_svd;

import DataHelper
import NN
from ImportNumpy import *;
from NN import *;
from MNIST import *;
from PTB import *;


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


def preprocess(text : str) -> (np.ndarray, dict, dict):
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


def createCoMatrix(corpus : np.ndarray, vocabSize : int, windowSize : int = 1) -> np.ndarray:
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


def mostSimilarity(word, word2ID, id2Word, C, top = 5):
    x = C[word2ID[word]];
    similarity = DataHelper.calcCosine(x, C).flatten();
    return [(id2Word[i], similarity[i]) for i in np.argsort(-similarity)[1: top + 1].tolist()];


def analogy(a, b, c, word2ID, id2Word, C, top = 5):
    x = C[word2ID[b]] - C[word2ID[a]] + C[word2ID[c]];
    similarity = DataHelper.calcCosine(x, C).flatten();
    return [(id2Word[i], similarity[i]) for i in np.argsort(-similarity)[1: top + 1].tolist()];


def ppmi(C : np.ndarray, epsilon = 1e-8) -> np.ndarray:
    N = np.sum(C);
    S = np.sum(C, 1, keepdims = True);
    S = S * S.T;

    M = C / S * N;
    return np.maximum(0, np.log2(M + epsilon));


def createContextsAndTarget(corpus : np.ndarray, windowSize : int = 1) -> (np.ndarray, np.ndarray):
    contexts = [];
    target = corpus[windowSize: -windowSize];

    for i in range(windowSize, len(corpus) - windowSize, 1):
        cs = corpus[i - windowSize: i + windowSize + 1].tolist();
        cs.pop(windowSize);

        contexts.append(cs);

    return np.array(contexts), target;


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


def createMNISTNN() -> INetModel:
    return SequentialContainer(
        ConvolutionLayer(16, 1, 3, 3, 1, 1),
        ReluLayer(),
        ConvolutionLayer(16, 16, 3, 3, 1, 1),
        ReluLayer(),
        MaxPoolingLayer(2, 2, 2),
        ConvolutionLayer(32, 16, 3, 3, 1, 1),
        ReluLayer(),
        ConvolutionLayer(32, 32, 3, 3, 1, 2),
        ReluLayer(),
        MaxPoolingLayer(2, 2, 2),
        ConvolutionLayer(64, 32, 3, 3, 1, 1),
        ReluLayer(),
        ConvolutionLayer(64, 64, 3, 3, 1, 1),
        ReluLayer(),
        MaxPoolingLayer(2, 2, 2),
        ReshapeLayer((-1, 64 * 4 * 4)),
        AffineLayer(64 * 4 * 4, 50),
        ReluLayer(),
        DropoutLayer(),
        AffineLayer(50, 10),
        DropoutLayer(),
    );


def testMNIST():
    mnist = MNIST("/media/WindowsE/Data/MNIST", normalize = True, flatten = False);

    model = createMNISTNN();
    lossFunc = SoftmaxWithCrossEntropyLoss();
    optimizer = Adam();
    trainIterator = DataIterator([mnist.trainX, mnist.trainY], batchSize = 2 ** 9);
    testIterator = DataIterator([mnist.testX, mnist.testY], batchSize = 2 ** 9, shuffle = False);
    evaluator = ClassifierAccuracyEvaluator();

    # filter_show(model.modules[0].weight.get());

    trainer = NetTrainer(model, lossFunc, optimizer, evaluator);
    trainer.train(20, trainIterator, testIterator);
    trainer.plot();

    # filter_show(model.modules[0].weight.get());


def testWord2Vec():
    ptb = PTB("/media/WindowsE/Data/PTB");
    corpus, word2ID, id2Word = ptb.trainCorpus, ptb.word2ID, ptb.id2Word;
    windowSize, hiddenSize, batchSize, negativeSize, maxEpoch = 5, 100, 2 ** 7, 5, 10;

    # with open("ptb_cbow.weights", "br") as file:
    #     wordVec = pickle.load(file)[0];
    #
    # for word in ["you", "year", "car", "toyota"]:
    #     for w, similarity in mostSimilarity(word, word2ID, id2Word, wordVec):
    #         print(f"{w}: {similarity}");
    #     print("");
    #
    #
    # for a, b, c in [("man", "king", "queen"), ("take", "took", "go"), ("car", "cars", "child"), ("good", "better", "bad")]:
    #     print(f"{a} -> {b} = {c} -> ?");
    #     for w, similarity in analogy(a, b, c, word2ID, id2Word, wordVec):
    #         print(f"{w}: {similarity}");
    #     print("");

    vocabSize = len(word2ID);
    contexts, target = createContextsAndTarget(corpus, windowSize);
    negativeSampler = CorpusNegativeSampler(corpus, negativeSize);

    # model = CBOWModel(windowSize, vocabSize, hiddenSize, negativeSampler);
    # data = [contexts, target, np.ones_like(target)];
    # filename = "ptb_cbow.weights";

    model = SkipGramModel(windowSize, vocabSize, hiddenSize, negativeSampler);
    data = [target, contexts, np.ones_like(contexts)];
    filename = "ptb_skipgram.weights";

    lossFunc = SigmoidWithCrossEntropyLoss();
    optimizer = Adam();
    trainIterator = SequentialDataIterator(data, batchSize = batchSize);
    evaluator = ClassifierAccuracyEvaluator();

    trainer = NetTrainer(model, lossFunc, optimizer, evaluator);
    trainer.train(maxEpoch, trainIterator);
    with open(filename, "bw") as file:
        pickle.dump(model.weights, file);
    trainer.plot();


def test():
    # x = np.random.randn(12);
    #
    # data = np.load("/media/WindowsE/Data/PARS/JNLH/AllYiCuiHua/ISYS_history_20210422_20210629/__JNRTDB_TIC6201.PV.npy");
    # # data = np.load("/media/WindowsE/Data/PARS/JNLH/AllYiCuiHua/PI_history_20190101_20201101/__JNRTDB_YCH_TIC6201.PV.npy");
    # X = data[:, 0];
    # q1, q3 = np.quantile(X, 0.25), np.quantile(X, 0.75);
    # IQR = 1.5 * (q3 - q1);
    #
    # c1 = np.sum(X <= q1 - IQR);
    # c2 = np.sum(X >= q3 + IQR);
    #
    # plt.figure();
    # plt.hist(X[X >=0], bins = 1000);
    # plt.show(block = True);
    # plt.close();
    #
    # print("exit");
    # return;

    # plt.figure(1);
    # for j in range(C):
    #     plt.scatter(X[T[:, j] == 1, 0].get(), X[T[:, j] == 1, 1].get(), marker = markers[j], color = colors[j]);
    # plt.show(block = True);
    # plt.close();

    # text = "you say goodbye and i say hello.";
    # corpus, word2ID, id2Word = preprocess(text);
    # windowSize, hiddenSize, batchSize, negativeSize, maxEpoch = 1, 5, 3, 2, 1000;

    ptb = PTB("/media/WindowsE/Data/PTB");
    trainingCorpus, testCorpus, word2ID, id2Word = ptb.trainCorpus, ptb.testCorpus, ptb.word2ID, ptb.id2Word;
    vocabSize = len(word2ID);
    vecSize, hiddenSize, batchSize, timeSize, maxEpoch = 120, 100, 20, 35, 4;

    # trainingCorpus = trainingCorpus[:10000];
    # vocabSize = int(np.amax(trainingCorpus)) + 1;

    model = SequentialContainer(
        EmbeddingLayer(vocabSize, vecSize),
        LstmLayer(vecSize, hiddenSize),
        ReshapeLayer((-1, hiddenSize)),
        AffineLayer(hiddenSize, vocabSize),
        ReshapeLayer((batchSize, -1, vocabSize)),
    );

    lossFunc = SoftmaxWithCrossEntropy1DLoss();
    optimizer = GradientsClipping(0.25, SGD(20));
    # optimizer = GradientsClipping(0.25, Adam(1.0));
    # optimizer = SGD(0.1);
    # trainIterator = SequentialDataIterator([corpus[:-1], corpus[1:]], batchSize = timeSize);
    trainingIterator = PartitionedDataIterator([trainingCorpus[:-1], trainingCorpus[1:]], partitionNumber = batchSize, batchSize = timeSize);
    testIterator = PartitionedDataIterator([testCorpus[:-1], testCorpus[1:]], partitionNumber = batchSize, batchSize = timeSize);
    evaluator = PerplexityAccuracyEvaluator();

    trainer = NetTrainer(model, lossFunc, optimizer, evaluator);
    trainer.train(maxEpoch, trainingIterator, testIterator = testIterator, evalIterations = 20, evalEpoch = True, evalTrainingData = True);
    trainer.plot();

    # U, S, V = U.get(), S.get(), V.get();
    #
    # plt.figure();
    # for word, wordID in word2ID.items():
    #     plt.annotate(word, (U[wordID, 0], U[wordID, 1]));
    # plt.scatter(U[:, 0], U[:, 1], alpha = 0.5);
    # plt.show(block = True);
    # plt.close();
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
