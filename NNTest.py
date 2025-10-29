import json;
import re;
import os;
import os.path;
import abc;
import math;
import time;
import datetime;
import pickle;
import random;
import collections;
import urllib;

import pandas as pd;
import scipy.stats;
import matplotlib.pyplot as plt;
import mpl_toolkits.mplot3d as p3d;

from typing import List, Tuple, Dict, Callable, Any;

import DeviceConfig;
DeviceConfig.floatLength = 32;
# DeviceConfig.enableGPU = True;

import DataHelper;
# from IsolationForest import *;
from ImportNumpy import *;
# from GMM import *;
from NN import *;
# from MNIST import *;
# from PTB import *;
import torch;
import torch.nn as nn;


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
    target = corpus[windowSize: -windowSize];
    contexts = [np.concatenate((corpus[i - windowSize: i], corpus[i + 1: i + 1 + windowSize])) for i in range(windowSize, len(corpus) - windowSize, 1)];

    return np.array(contexts, dtype = np.int32), target;


def loadSpiral(N = 1000, C = 3) -> Tuple[np.ndarray, np.ndarray]:
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
    X, T = loadSpiral(100, C);

    markers, colors = ["x", "*", "+", "s", "d", "o"], ["b", "k", "g", "m", "y", "r"];

    plt.figure(1);
    for j in range(C):
        plt.scatter(X[T[:, j] == 1, 0], X[T[:, j] == 1, 1], marker = markers[j], color = colors[j]);
    plt.show(block = True);
    plt.close();

    lr ,batchSize, maxEpoch = 1.0, 30, 300;
    trainIterator = SequentialDataIterator([X, T], batchSize = batchSize, shuffle = True);
    lossFunc = SoftmaxWithCrossEntropyLoss();
    optimizer = SGD(lr);
    evaluator = ClassifierAccuracyEvaluator();

    model = SequentialContainer(
        AffineLayer(X.shape[1], 10),
        ReluLayer(),
        AffineLayer(10, C),
    );
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, evaluator = evaluator, plot = True);

    minX, maxX = np.amin(X[:, 0]), np.amax(X[:, 0]);
    minY, maxY = np.amin(X[:, 1]), np.amax(X[:, 1]);
    x, y = np.arange(minX - 1, maxX + 1, 0.01), np.arange(minY - 1, maxY + 1, 0.01);
    xx, yy = np.meshgrid(x, y);
    features = np.stack((xx, yy), axis = -1).reshape(-1, 2);
    labels = np.argmax(model.predictOne(features)[0], axis = -1);

    plt.figure(1);
    for j in range(C):
        plt.scatter(features[labels == j, 0], features[labels == j, 1], marker = ".", color = colors[-(j + 1)]);
    for j in range(C):
        plt.scatter(X[T[:, j] == 1, 0], X[T[:, j] == 1, 1], marker = markers[j], color = colors[j]);
    plt.show(block = True);
    plt.close();


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


def createMNIST_MLP() -> INetModel:
    D = 784;
    return SequentialContainer(
        AffineLayer(D, D * 2),
        ReluLayer(),
        AffineLayer(D * 2, D // 2),
        ReluLayer(),
        AffineLayer(D // 2, 10),
    );


def createMNIST_CNN() -> INetModel:
    return SequentialContainer(
        Convolution2DLayer(16, 1, 3, 3, 1, 1),
        ReluLayer(),
        Convolution2DLayer(16, 16, 3, 3, 1, 1),
        ReluLayer(),
        MaxPooling2DLayer(2, 2, 2),
        Convolution2DLayer(32, 16, 3, 3, 1, 1),
        ReluLayer(),
        Convolution2DLayer(32, 32, 3, 3, 1, 2),
        ReluLayer(),
        MaxPooling2DLayer(2, 2, 2),
        Convolution2DLayer(64, 32, 3, 3, 1, 1),
        ReluLayer(),
        Convolution2DLayer(64, 64, 3, 3, 1, 1),
        ReluLayer(),
        MaxPooling2DLayer(2, 2, 2),
        ReshapeLayer((-1, 64 * 4 * 4)),
        AffineLayer(64 * 4 * 4, 50),
        ReluLayer(),
        DropoutLayer(),
        AffineLayer(50, 10),
        DropoutLayer(),
    );


def testMNIST():
    # mnist = MNIST("/media/WindowsE/Data/MNIST", normalize = True, flatten = False);
    mnist = MNIST("data/FashionMNIST/raw/", normalize = True, flatten = False);
    batchSize, maxEpoch = 32, 100;

    # model = createMNIST_MLP();
    model = createMNIST_CNN();
    lossFunc = SoftmaxWithCrossEntropy1DLoss();
    # optimizer = SGD(0.01);
    optimizer = Adam(0.01);
    trainIterator = SequentialDataIterator([mnist.trainX, mnist.trainY], batchSize = batchSize);
    testIterator = SequentialDataIterator([mnist.testX, mnist.testY], batchSize = batchSize, shuffle = False);
    evaluator = ClassifierAccuracyEvaluator();

    # filter_show(model.modules[0].weight.get());

    # trainer = NetTrainer(model, lossFunc, optimizer, evaluator);
    # trainer.train(20, trainIterator, testIterator);
    # trainer.plot();
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = True);
    # with open("adam.lr", "wb") as file:
    #     pickle.dump(optimizer.learningRate, file);

    # filter_show(model.modules[0].weight.get());


def testWord2VecPTB():
    ptb = PTB("/media/WindowsE/Data/PTB");
    corpus, word2ID, id2Word = ptb.trainCorpus, ptb.word2ID, ptb.id2Word;
    windowSize, hiddenSize, batchSize, negativeSize, maxEpoch = 5, 100, 2 ** 8, 5, 20;

    # with open("ptb_skipgram.weights", "br") as file:
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
    # # data = [contexts, target, np.ones_like(target)];
    # data = [contexts, target];
    # filename = "data/ptb_cbow.weights";

    model = SkipGramModel(windowSize, vocabSize, hiddenSize, negativeSampler);
    # data = [target, contexts, np.ones_like(contexts)];
    data = [target, contexts];
    filename = "data/ptb_skipgram.weights";

    lossFunc = SigmoidWithCrossEntropyLoss();
    optimizer = Adam();
    trainIterator = SequentialDataIterator(data, batchSize = batchSize);
    evaluator = ClassifierAccuracyEvaluator();
    model.fit(trainIterator, lossFunc, optimizer, maxEpoch, evaluator = evaluator, plot = True);

    with open(filename, "bw") as file:
        pickle.dump(model.weights, file);


def testPTB():
    # text = "you say goodbye and i say hello.";
    # corpus, word2ID, id2Word = preprocess(text);
    # windowSize, hiddenSize, batchSize, negativeSize, maxEpoch = 1, 5, 3, 2, 1000;

    filename = "ptb_lm.weights";
    ptb = PTB("/media/WindowsE/Data/PTB");
    trainingCorpus, testCorpus, word2ID, id2Word = ptb.trainCorpus, ptb.testCorpus, ptb.word2ID, ptb.id2Word;
    vocabSize = len(word2ID);
    vecSize, hiddenSize, batchSize, timeSize, maxEpoch = 650, 650, 20, 35, 10;

    # trainingCorpus = trainingCorpus[:10000];
    # vocabSize = int(np.amax(trainingCorpus)) + 1;

    embedding = EmbeddingLayer(vocabSize, hiddenSize);

    model = SequentialContainer(
        embedding,
        VariationalDropoutLayer(),
        LstmLayer(vecSize, hiddenSize, returnSequences = True),
        VariationalDropoutLayer(),
        LstmLayer(hiddenSize, hiddenSize, returnSequences = True),
        VariationalDropoutLayer(),
        ReshapeLayer((-1, hiddenSize)),
        AffineLayer(hiddenSize, vocabSize, W = embedding.weight.T),
        ReshapeLayer((batchSize, -1, vocabSize)),
    );

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params = pickle.load(file);
        model.params = params;

    lossFunc = SoftmaxWithCrossEntropy1DLoss();
    optimizer = ParametersShare(GradientsClipping(1, Adam()));
    # optimizer = GradientsClipping(0.25, Adam(1.0));
    # optimizer = SGD(0.1);
    # trainIterator = SequentialDataIterator([corpus[:-1], corpus[1:]], batchSize = timeSize);
    trainingIterator = PartitionedDataIterator([trainingCorpus[:-1], trainingCorpus[1:]], batchSize = batchSize,
                                               stepSize = timeSize);
    testIterator = PartitionedDataIterator([testCorpus[:-1], testCorpus[1:]], batchSize = batchSize,
                                           stepSize = timeSize);
    evaluator = PerplexityAccuracyEvaluator();

    trainer = NetTrainer(model, lossFunc, optimizer, evaluator);
    trainer.train(maxEpoch, trainingIterator, testIterator = testIterator, evalIterations = 20, evalEpoch = True,
                  evalTrainingData = True);
    trainer.plot();

    with open(filename, "bw") as file:
        pickle.dump(model.params, file);


def testKeras():
    def getData(X : np.ndarray, size : int) -> np.ndarray:
        return list2OneHot([[i if i < size else 0 for i in x] for x in X], size, defaultDType);


    vocabSize = 10000;
    filePath = "/media/WindowsE/Data/IMDB/imdb.npz";
    with np.load(filePath, allow_pickle = True) as file:
        X_train, X_test = file["x_train"], file["x_test"];
        Y_train, Y_test = file["y_train"], file["y_test"];

    X_train, X_test = getData(X_train, vocabSize), getData(X_test, vocabSize);
    Y_train, Y_test = Y_train.reshape(-1, 1), Y_test.reshape(-1, 1);

    inputSize, hiddenSize, outputSize, batchSize, maxEpoch = X_train.shape[-1], 16, 1, 32, 20;
    model = SequentialContainer(
        AffineLayer(inputSize, hiddenSize),
        ReluLayer(),
        # DropoutLayer(0.2),
        AffineLayer(hiddenSize, hiddenSize),
        ReluLayer(),
        # DropoutLayer(0.2),
        AffineLayer(hiddenSize, outputSize),
    );

    lossFunc = SigmoidWithCrossEntropyLoss();
    optimizer = RMSProp();
    trainingIterator = SequentialDataIterator([X_train, Y_train], batchSize);
    testIterator = SequentialDataIterator([X_test, Y_test], batchSize);
    evaluator = ClassifierAccuracyEvaluator();
    model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, minEpoch = 0, testIterator = testIterator, evaluator = evaluator, plot = True);

    # Y_test_hat = np.concatenate(tuple([sigmoid(item[0]) for item in model.predict(SequentialDataIterator([X_test, Y_test], batchSize, shuffle = False))]), axis = 0);


    print("exit.");


def test():
    # x0 = np.array([30, 34, 38, 42, 46, 50, 54, 58], dtype = np.float32);
    # # x0 = np.arange(0, 48, 4, dtype = np.float32);
    # y0 = np.array([11, 14, 19, 32, 42, 48, 50, 52], dtype = np.float32);
    # L = 60.0;
    # x = x0;
    # y = np.log((L - y0) / y0);
    # n = len(x);
    # xMean, yMean = np.mean(x), np.mean(y);
    # beta1 = np.dot(x - xMean, y) / np.sum((x - xMean) ** 2);
    # beta0 = yMean - beta1 * xMean;
    # yHat = beta0 + beta1 * x;
    # yHat0 = L / (1 + np.exp(beta0 + beta1 * x));
    #
    # # beta1 = (np.sum(x * y) - 100 * np.sum(x)) / np.sum(x ** 2);
    # # yHat = 100 + beta1 * x;
    #
    # # maskX, maskY = x < xMean, y < yMean;
    # # B = np.array([
    # #     [np.sum(maskX & maskY), np.sum(~maskX & maskY)],
    # #     [np.sum(maskX & ~maskY), np.sum(~maskX & ~maskY)],
    # # ]);
    # # E = n * np.array([np.sum(maskY) / n, 1 - np.sum(maskY) / n]).reshape(2, 1) * np.array([np.sum(maskX) / n, 1 - np.sum(maskX) / n]).reshape(1, 2);
    # # D = np.sum((B - E) ** 2 / E);
    # # pValue = 1 - scipy.stats.chi2.cdf(D, df = 1);
    #
    # plt.figure();
    # # plt.scatter(x, y);
    # # plt.plot(x, yHat, "-r");
    # plt.scatter(x0, y0);
    # plt.plot(np.sort(x0), yHat0, "-r");
    # plt.show(block = True);
    #
    # plt.figure();
    # # plt.scatter(x, y - yHat);
    # plt.scatter(x0, y0 - yHat0);
    # plt.show(block = True);

    print("go\n");

    # testSpiral();
    # testWord2VecPTB();
    # testMNIST();
    # testKeras();
    unitTest();
    # testSeq2Seq();
    # testAddition();
    # testTS_Many2One();
    # testTS_Many2Many();
    # testTS1_Seq2Seq();
    # testTS2_BoostingSeq2Seq();
    # testTS3_LM_11_1step();
    # testTS3_LM_N1_1step();
    # testTS4_LM_Nstep();
    # testTS5_LM_1step_All();
    # testAE_NASABearing();
    # testAE_TagData();
    # testIF_TagData();
    # testGMM_TagData();
    # testGaussianVAE_TagData();
    # testBernoulliVAE_TagData();

    return;


def readTimeMachine():
    with open("/media/WindowsE/Data/timemachine.txt", "rt") as file:
        lines = file.readlines();

    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines];


def tokenize(lines : List[str], token : str = "word") -> List[List[str]]:
    if token == "word":
        return [line.split() for line in lines];
    elif token == "char":
        return [list(line) for line in lines];
    else:
        print(f"error: unknown token type {token}");


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
            return self._token2ID[tokens];


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


def loadCorpusTimeMachine(maxTokens : int = -1):
    lines = readTimeMachine();
    tokens = tokenize(lines, "char");
    vocab = Vocab(tokens);
    corpus = [vocab[token] for line in tokens for token in line];

    if maxTokens > 0:
        corpus = corpus[:maxTokens];

    return corpus, vocab;


def loadDataTimeMachine(batchSize : int, stepSize : int, shuffle : bool = False, maxTokens : int = -1):
    corpus, vocab = loadCorpusTimeMachine(maxTokens);
    X, Y = expand2OneHot(np.array(corpus[:-1]), len(vocab)).astype(defaultDType), np.array(corpus[1:]);
    iterator = PartitionedDataIterator([X, Y], batchSize, stepSize, shuffle);

    return iterator, vocab;


def predict(prefix : str, count : int, model : INetModel, vocab : Vocab) -> str:
    vocabSize = len(vocab);
    X = expand2OneHot(np.array([vocab[c] for c in prefix[:-1]]).reshape((1, -1)), vocabSize).astype(defaultDType);

    model.reset();
    Y, = model.forward(X);

    outputs = prefix;
    for _ in range(count):
        X = expand2OneHot(np.array([vocab[outputs[-1]]]).reshape((1, -1)), vocabSize).astype(defaultDType);
        Y, = model.forward(X);

        idx = int(Y.flatten().argmax());
        if idx == vocab.unk:
            break;

        outputs += vocab.id2Token[idx];

    return outputs;


def testPyTorch():
    testTS2();
    return;

    batchSize, stepSize, maxTokens, maxEpoch = 32, 35, 10000, 500;
    iterator, vocab = loadDataTimeMachine(batchSize, stepSize, maxTokens = maxTokens);

    vocabSize, hiddenSize = len(vocab), 256;
    model = SequentialContainer(
        GruLayer(vocabSize, hiddenSize),
        GruLayer(hiddenSize, hiddenSize),
        ReshapeLayer((-1, hiddenSize)),
        AffineLayer(hiddenSize, vocabSize),
    );

    lossFunc = SoftmaxWithCrossEntropy1DLoss();
    optimizer = GradientsClipping(1.0, SGD(1.0));
    trainingIterator = iterator;
    testIterator = None;
    evaluator = PerplexityAccuracyEvaluator();

    trainer = NetTrainer(model, lossFunc, optimizer, evaluator);
    trainer.train(maxEpoch, trainingIterator, testIterator = testIterator, evalEpoch = True, evalTrainingData = True);
    trainer.plot();

    print(predict('time traveller', 50, model, vocab))
    print(predict('traveller', 50, model, vocab));

    print("exit.");


# find Longest-Continuous-Subsequence
def findLCS(X : np.ndarray, predicate : Callable, startIndex : int = None, endIndex : int = None) -> Tuple[int, int]:
    if len(X) == 0:
        return 0, 0;

    if startIndex is None:
        startIndex = 0;
    if endIndex is None:
        endIndex = len(X);

    r1, r2 = 0, 0;
    j1, j2 = None, None;
    X = X[startIndex: endIndex];

    for i in range(len(X)):
        if predicate(X[i], X, i):
            if j1 is None:
                j1 = i;

            j2 = i + 1;
        else:
            if j1 is not None and j2 is not None and j2 - j1 > r2 - r1:
                r1, r2 = j1, j2;

            j1, j2 = None, None;

    if j1 is not None and j2 is not None and j2 - j1 > r2 - r1:
        r1, r2 = j1, j2;

    return r1 + startIndex, r2;


def findSegments(X : np.ndarray, predicate : Callable, minLength : int = 0) -> List[Tuple[int, int]]:
    if len(X) == 0:
        return [(0, 0)];

    j = None;
    segments = [];
    for i in range(len(X)):
        if predicate(X[i], X, i):
            if j is None:
                j = i;
        else:
            if j is not None:
                segments.append((j, i));
                j = None;

    if j is not None:
        segments.append((j, len(X)));

    return [(r1, r2) for r1, r2 in segments if r2 - r1 > minLength];


def loadDataset(X : np.ndarray, M : np.ndarray, minLength : int = 0) -> List[np.ndarray]:
    return [X[r1: r2] for r1, r2 in findSegments(M, lambda v, x, i: np.all(v == 1), minLength)];


def lagAndForecast(dataset : np.ndarray, LN : int, FN : int) -> (np.ndarray, np.ndarray):
    N, D = len(dataset) - LN - FN + 1, dataset.shape[-1];
    X, Y = np.zeros((N, LN, D)), np.zeros((N, FN, D));

    for i in range(N):
        X[i] = dataset[i: i + LN];
        Y[i] = dataset[i + LN: i + LN + FN];

    return X, Y;


def loadRawTSData(targetTags : List[str]) -> Tuple[np.ndarray, np.ndarray]:
    data, marks = [], [];
    folders = ["1CH", "2CH", "2JQ", "3JQ", "BW", "CA", "CA_GYGQ", "CY", "DL", "GD", "HMJQ", "JBX", "JHS", "JLSC", "LH", "QF", "RHY", "RHYJQ", "SZORB", "WUX", "XHS", "YCL", "YP"];

    for name in targetTags:
        for folder in folders:
            filePath = f"/media/WindowsE/Data/PARS/JNLH/AiModel/isys_data_20210701_20220516_180/{folder}/__JNRTDB_{name}.npy";
            if os.path.isfile(filePath):
                break;

            filePath = None;

        if filePath is None:
            raise FileNotFoundError(name);

        d = np.load(filePath);
        data.append(d[:, 0]);
        marks.append(d[:, 1]);
    data, marks = np.column_stack(tuple(data)), np.column_stack(tuple(marks));

    return data.astype(defaultDType), marks.astype(defaultDType);


def loadTSData1AndMany(targetTag : str, negativeTags : List[str], positiveTags : List[str]) -> Tuple[np.ndarray, np.ndarray]:
    data, marks = loadRawTSData([item for item in negativeTags + positiveTags + [targetTag] if not item.endswith(".SV")]);

    # remove status value, setup value
    idx = [];
    for j in range(data.shape[-1] - 1):
        if len(set(data[marks[:, j] == 1, j].tolist())) > 100:
            idx.append(j);
    idx.append(data.shape[-1] - 1);
    data, marks = data[:, idx], marks[:, idx];

    # remove break points
    breakPoints, M = [], np.all(marks == 1, axis = -1);
    for i in range(1, len(data) - 1):
        if M[i - 1] and not M[i] and M[i + 1]:
            breakPoints.append(i);

    for i in breakPoints:
        idx = marks[i] != 1;
        data[i, idx] = (data[i - 1, idx] + data[i + 1, idx]) * 0.5;
        marks[i] = 1;

    return data, marks;


def testTS_Many2One():
    targetTag = "TI6116A.PV";
    negativeTags = [];   # ['XIP301A.PV', 'DI6103.PV', 'FIC6031.SV', 'PDI6105.PV', 'DI6104.PV', 'JI6601.PV', 'TIC6522.SV'];
    # positiveTags = [];
    positiveTags = ['TI6116B.PV', 'TIC6115.PV', 'TI6106.PV', 'TI6901.PV', 'TI6166A.PV', 'TE6658.PV', 'TI6107D.PV', 'TI6107C.PV', 'TI6107H.PV', 'TI6107G.PV'];

    data, marks = loadTSData1AndMany(targetTag, negativeTags, positiveTags);
    segments = loadDataset(data, marks);
    segmentIndex = -2;
    dataSet = segments[segmentIndex];
    trainSize, predictSize = int(len(dataSet) * 0.8), 10;
    stepSize = 10 * predictSize;

    scaler = StandardScaler();
    scaler.fit(dataSet[:trainSize]);

    X = scaler.transform(dataSet);
    X_train, Y_train = lagAndForecast(X[: trainSize], stepSize, predictSize);
    X_test, Y_test = lagAndForecast(X[trainSize:], stepSize, predictSize);

    Y_train, Y_test = Y_train[..., -1], Y_test[..., -1];

    batchSize, inputSize, hiddenSize, outputSize, maxEpoch = 32, X_train.shape[-1], 32, Y_train.shape[-1], 100;
    filename = f"many2One_{targetTag}_{segmentIndex}_{inputSize}f_{stepSize}t_{predictSize}p.weights";
    model = SequentialContainer(
        LstmLayer(inputSize, hiddenSize),
        AffineLayer(hiddenSize, outputSize),
    );

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params = pickle.load(file);
        if isinstance(params[0], cp.ndarray):
            params = [cp.asnumpy(p) for p in params];
        model.params = params;

    model.reset();
    Y_train_hat = np.concatenate(tuple([model.forward(*item)[0] for item in SequentialDataIterator([X_train, Y_train], batchSize, shuffle = False)]), axis = 0);
    Y_train_hat = scaler.inverse(Y_train_hat, index = -1);
    Y_train_real = scaler.inverse(Y_train, index = -1);
    print(f"train {predictSize}-step MAE: {meanAbsoluteError(Y_train_real, Y_train_hat)}");

    idx = np.random.randint(0, len(X_train), 20);
    for i in idx.tolist():
        plt.figure(1, (14, 8));
        plt.plot(np.arange(predictSize), Y_train_real[i], "D-b", label = "real data");
        plt.plot(np.arange(predictSize), Y_train_hat[i], "s-r", label = "predict data");
        plt.legend(loc = 'upper right');
        plt.show(block = True);
        plt.close();

    model.reset();
    Y_test_hat = np.concatenate(tuple([model.forward(*item)[0] for item in SequentialDataIterator([X_test, Y_test], batchSize, shuffle = False)]), axis = 0);
    Y_test_hat = scaler.inverse(Y_test_hat, index = -1);
    Y_test_real = scaler.inverse(Y_test, index = -1);
    print(f"test {predictSize}-step MAE: {meanAbsoluteError(Y_test_real, Y_test_hat)}");

    idx = np.random.randint(0, len(X_test), 20);
    for i in idx.tolist():
        plt.figure(1, (14, 8));
        plt.plot(np.arange(predictSize), Y_test_real[i], "D-b", label = "real data");
        plt.plot(np.arange(predictSize), Y_test_hat[i], "s-r", label = "predict data");
        plt.legend(loc = 'upper right');
        plt.show(block = True);
        plt.close();

    # lossFunc = IdentityWithMeanSquareLoss();
    # optimizer = GradientsClipping(5.0, Adam());
    # trainingIterator = SequentialDataIterator([X_train, Y_train], batchSize, shuffle = False);
    # testIterator = SequentialDataIterator([X_test, Y_test], batchSize, shuffle = False);
    # evaluator = MaeAccuracyEvaluator();
    # model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, minEpoch = 1, testIterator = testIterator, evaluator = evaluator, plot = True);
    #
    # with open(filename, "wb") as file:
    #     pickle.dump(model.params, file);

    print("exit.");


def testTS_Many2Many():
    targetTag = "TI6116A.PV";
    negativeTags = [];  #['XIP301A.PV', 'DI6103.PV', 'FIC6031.SV', 'PDI6105.PV', 'DI6104.PV', 'JI6601.PV', 'TIC6522.SV'];
    positiveTags = [];
    # positiveTags = ['TI6116B.PV', 'TIC6115.PV', 'TI6106.PV', 'TI6901.PV', 'TI6166A.PV', 'TE6658.PV', 'TI6107D.PV', 'TI6107C.PV', 'TI6107H.PV', 'TI6107G.PV'];

    data, marks = loadTSData1AndMany(targetTag, negativeTags, positiveTags);
    segments = loadDataset(data, marks);
    segmentIndex = -2;
    dataSet = segments[segmentIndex];
    trainSize, predictSize = int(len(dataSet) * 0.8), 10;
    stepSize = 10 * predictSize;

    scaler = StandardScaler();
    scaler.fit(dataSet[:trainSize]);

    X = scaler.transform(dataSet);
    X_train, Y_train = lagAndForecast(X[: trainSize], stepSize, predictSize);
    X_test, Y_test = lagAndForecast(X[trainSize:], stepSize, predictSize);

    Y_train, Y_test = Y_train[..., [-1]], Y_test[..., [-1]];

    batchSize, inputSize, hiddenSize, outputSize, maxEpoch = 32, X_train.shape[-1], 32, Y_train.shape[-1], 100;
    filename = f"many2Many_{targetTag}_{segmentIndex}_{inputSize}f_{stepSize}t_{predictSize}p.weights";
    model = Seq2SeqTSModel_Seq2Seq_OnlyStateInput(inputSize, hiddenSize, outputSize);

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params = pickle.load(file);
        if isinstance(params[0], cp.ndarray):
            params = [cp.asnumpy(p) for p in params];
        model.params = params;

    model.reset();
    Y_train_hat = np.concatenate(tuple([model.forward(*item)[0] for item in SequentialDataIterator([X_train, Y_train], batchSize, shuffle = False)]), axis = 0);
    Y_train_hat = scaler.inverse(Y_train_hat, index = -1);
    Y_train_real = scaler.inverse(Y_train, index = -1);
    X_train_real = scaler.inverse(X_train, index = -1);
    print(f"train {predictSize}-step MAE: {meanAbsoluteError(Y_train_real, Y_train_hat)}");

    idx = np.random.randint(0, len(X_train_real), 100);
    for i in idx.tolist():
        plt.figure(1, (14, 8));
        plt.plot(np.arange(stepSize + predictSize), np.concatenate((X_train_real[i], Y_train_real[i]), axis = 0).flatten(), "D-b", label = "real data");
        plt.plot(np.arange(stepSize, stepSize + predictSize), Y_train_hat[i].flatten(), "s-r", label = "predict data");
        plt.legend(loc = 'upper right');
        plt.show(block = True);
        plt.close();

    model.reset();
    Y_test_hat = np.concatenate(tuple([model.forward(*item)[0] for item in SequentialDataIterator([X_test, Y_test], batchSize, shuffle = False)]), axis = 0);
    Y_test_hat = scaler.inverse(Y_test_hat, index = -1);
    Y_test_real = scaler.inverse(Y_test, index = -1);
    print(f"test {predictSize}-step MAE: {meanAbsoluteError(Y_test_real, Y_test_hat)}");

    idx = np.random.randint(0, len(X_test), 20);
    for i in idx.tolist():
        plt.figure(1, (14, 8));
        plt.plot(np.arange(predictSize), Y_test_real[i].flatten(), "D-b", label = "real data");
        plt.plot(np.arange(predictSize), Y_test_hat[i].flatten(), "s-r", label = "predict data");
        plt.legend(loc = 'upper right');
        plt.show(block = True);
        plt.close();

    # lossFunc = IdentityWithMeanSquareLoss();
    # optimizer = GradientsClipping(5.0, Adam());
    # trainingIterator = SequentialDataIterator([X_train, Y_train], batchSize, shuffle = False);
    # testIterator = SequentialDataIterator([X_test, Y_test], batchSize, shuffle = False);
    # evaluator = MaeAccuracyEvaluator();
    # model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, minEpoch = 1, testIterator = testIterator, evaluator = evaluator, plot = True);
    #
    # with open(filename, "wb") as file:
    #     pickle.dump(model.params, file);

    print("exit.");


def testTS1_Seq2Seq():
    # TI6116A = np.load("/media/WindowsE/Data/PARS/JNLH/AiModel/isys_data_20210701_20220401_180/__JNRTDB_TI6116A.PV.npy");
    # datasets = loadDataset(TI6116A[:, 0].reshape(-1, 1), TI6116A[:, 1].reshape(-1, 1), int(1440 / 3 * 7));
    # datasets = [np.load("TI6116A.npy").reshape(-1, 1)];
    # scaler = StandardScaler();
    # scaler.fit(np.concatenate(tuple(datasets), axis = 0));
    #
    # filename = "lstm1.weights";
    # stepSize, predictSize = 100, 10;
    #
    # X_train, Y_train, X_test, Y_test = [], [], [], [];
    # for dataset in datasets:
    #     X = dataset;
    #     X = scaler.transform(X);
    #     trainSize = int(len(X) * 0.8);
    #
    #     x, y = lagAndForecast(X[:trainSize], stepSize, predictSize);
    #     X_train.append(x);
    #     Y_train.append(y);
    #
    #     x, y = lagAndForecast(X[trainSize:], stepSize, predictSize);
    #     X_test.append(x);
    #     Y_test.append(y);
    #
    # X_train = np.concatenate(tuple(X_train), axis = 0);
    # Y_train = np.concatenate(tuple(Y_train), axis = 0);
    # X_test = np.concatenate(tuple(X_test), axis = 0);
    # Y_test = np.concatenate(tuple(Y_test), axis = 0);

    TI6116A = np.load("TI6116A.npy");
    # scaler = StandardScaler();
    # scaler.fit(TI6116A);

    filename = "lstm1.weights";
    trainSize, stepSize, predictSize = int(len(TI6116A) * 0.8), 30, 3;

    # X = scaler.transform(TI6116A);
    X = np.diff(TI6116A, axis = 0);
    X_train, Y_train = lagAndForecast(X[:trainSize], stepSize, predictSize);
    X_test, Y_test = lagAndForecast(X[trainSize:], stepSize, predictSize);

    X = np.concatenate((X_train, Y_train), axis = 1).reshape(-1, stepSize + predictSize);
    Y = np.repeat(np.mean(X[:, -2 * predictSize: -predictSize], axis = -1, keepdims = True), predictSize, axis = -1);
    print(f"train base error: {meanAbsoluteError(X[:, -predictSize:], Y)}");

    X = np.concatenate((X_test, Y_test), axis = 1).reshape(-1, stepSize + predictSize);
    Y = np.repeat(np.mean(X[:, -2 * predictSize: -predictSize], axis = -1, keepdims = True), predictSize, axis = -1);
    print(f"train base error: {meanAbsoluteError(X[:, -predictSize:], Y)}");

    # for i in np.random.randint(0, len(X_train), 20).tolist():
    #     plt.figure();
    #     plt.plot(X_train[i].flatten(), "x-");
    #     plt.show(block = True);
    #     plt.close();

    batchSize, inputSize, hiddenSize, maxEpoch = 32, X_train.shape[-1], 128, 40;
    model = Seq2SeqTSModel_Seq2Seq_OnlyStateInput(inputSize, hiddenSize, inputSize, layerNum = 2, inputDropout = 0.2, recurrentDropout = 0.2);

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params = pickle.load(file);
        if isinstance(params[0], cp.ndarray):
            params = [cp.asnumpy(p) for p in params];
        model.params = params;

    model.reset();
    Y_hat = np.concatenate(tuple([item[0] for item in list(model.predict(SequentialDataIterator([X_test, Y_test], batchSize, shuffle = False)))]), axis = 0);
    Y_real = model.getFinalTag(Y_test);

    # X_test, Y_test, Y_real, Y_hat = scaler.inverse(X_test), scaler.inverse(Y_test), scaler.inverse(Y_real), scaler.inverse(Y_hat),
    print(f"MAE: {meanAbsoluteError(Y_real, Y_hat)}");

    idx = np.random.randint(0, len(X_test), 20);
    for i in idx.tolist():
        x = np.concatenate((X_test[i], Y_test[i]), axis = None);

        plt.figure(1);
        plt.plot(np.arange(len(x)), x, "D-b", label = "real data");
        plt.plot(np.arange(len(x) - predictSize, len(x)), Y_hat[i].flatten(), "s-r", label = "predict data");
        plt.legend(loc = 'upper left');
        plt.show(block = True);
        plt.close();

    # lossFunc = IdentityWithMeanSquareLoss();
    # optimizer = GradientsClipping(5.0, Adam());
    # trainingIterator = SequentialDataIterator([X_train, Y_train], batchSize);
    # testIterator = SequentialDataIterator([X_test, Y_test], batchSize);
    # evaluator = MaeAccuracyEvaluator();
    # model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, minEpoch = 5, plot = True);
    #
    # with open(filename, "wb") as file:
    #     pickle.dump(model.params, file);

    print("exit.");


def testTS2_BoostingSeq2Seq():
    # X = TI6116A[: 1000 * 480].flatten();
    # X = np.diff(X);
    # acf = [DataHelper.calcAcf(X, k + 1) for k in range(100)];
    # plt.figure();
    # plt.plot(acf, "x-");
    # plt.show(block = True);
    # plt.close();
    #
    #
    # return;
    TI6116A = np.load("TI6116A.npy");
    scaler = StandardScaler();
    scaler.fit(TI6116A);

    filename = "lstm2.weights";
    trainSize, stepSize, predictSize = int(len(TI6116A) * 0.8), 100, 10;

    X = scaler.transform(TI6116A);
    X_train, Y_train = lagAndForecast(X[:trainSize], stepSize, predictSize);
    X_test, Y_test = lagAndForecast(X[trainSize:], stepSize, predictSize);

    # for i in np.random.randint(0, len(X_train), 20).tolist():
    #     plt.figure();
    #     plt.plot(np.concatenate((X_train[i], Y_train[i]), axis = None), "x-");
    #     plt.show(block = True);
    #     plt.close();

    maxModelNumber, batchSize, inputSize, hiddenSize, maxEpoch = 5, 32, X_train.shape[-1], 128, 20;
    models = [Seq2SeqTSModel_Seq2Seq_OnlyStateInput(inputSize, hiddenSize, inputSize, inputDropout = 0.5, recurrentDropout = 0.5) for _ in range(maxModelNumber)];

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params = pickle.load(file);
        for i in range(len(params)):
            models[i].params = params[i];

    for m in models:
        m.reset();

    Y_hat = np.zeros_like(Y_test);
    for m in models:
        Y_hat += np.concatenate(tuple([item[0] for item in list(m.predict(SequentialDataIterator([X_test, Y_test], batchSize, shuffle = False)))]), axis = 0);
    Y_real = models[0].getFinalTag(Y_test);

    X_test, Y_test, Y_real, Y_hat = scaler.inverse(X_test), scaler.inverse(Y_test), scaler.inverse(Y_real), scaler.inverse(Y_hat),
    print(f"MAE: {meanAbsoluteError(Y_real, Y_hat)}");

    idx = np.random.randint(0, len(X_test), 20);
    for i in idx.tolist():
        x = np.concatenate((X_test[i], Y_test[i]), axis = None);

        plt.figure(1);
        plt.plot(np.arange(len(x)), x, "D-b", label = "real data");
        plt.plot(np.arange(len(x) - predictSize, len(x)), Y_hat[i].flatten(), "s-r", label = "predict data");
        plt.legend(loc = 'upper left');
        plt.show(block = True);
        plt.close();

    # R_train, R_test = Y_train, Y_test;
    # Y_train_hat, Y_test_hat = np.zeros_like(Y_train), np.zeros_like(Y_test);
    # trainError, testError = [], [];
    # for i, model in enumerate(models):
    #     lossFunc = IdentityWithMeanSquareLoss();
    #     optimizer = GradientsClipping(5.0, Adam());
    #     trainingIterator = SequentialDataIterator([X_train, R_train], batchSize);
    #     testIterator = SequentialDataIterator([X_test, R_test], batchSize);
    #     evaluator = MaeAccuracyEvaluator();
    #     model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, minEpoch = 1, plot = False);
    #
    #     R_train_hat = np.concatenate(tuple([item[0] for item in list(model.predict(SequentialDataIterator([X_train, R_train], batchSize, shuffle = False)))]), axis = 0);
    #     R_test_hat = np.concatenate(tuple([item[0] for item in list(model.predict(SequentialDataIterator([X_test, R_test], batchSize, shuffle = False)))]), axis = 0);
    #     R_train = R_train - R_train_hat;
    #     R_test = R_test - R_test_hat;
    #
    #     Y_train_hat += R_train_hat;
    #     Y_test_hat += R_test_hat;
    #     trainError.append(meanAbsoluteError(Y_train, Y_train_hat));
    #     testError.append(meanAbsoluteError(Y_test, Y_test_hat));
    #     print(f"{i + 1} boosting train MAE: {trainError[-1]},test MAE: {testError[-1]}");
    #     print("\n\n");
    #
    # plt.figure(1);
    # plt.plot(np.arange(len(trainError)), trainError, "D-b", label = "train MAE");
    # plt.plot(np.arange(len(testError)), testError, "s-r", label = "test MAE");
    # plt.legend(loc = 'upper right');
    # plt.show(block = True);
    # plt.close();
    #
    # with open(filename, "wb") as file:
    #     pickle.dump([m.params for m in models], file);

    print("exit.");


def testTS3_LM_11_1step():
    TI6116A = np.load("TI6116A.npy");
    scaler = StandardScaler();
    scaler.fit(TI6116A);

    filename = "lstm31.weights";
    trainSize, stepSize, predictSize = int(len(TI6116A) * 0.8), 240, 10;

    X = scaler.transform(TI6116A);
    X_train, Y_train = X[:trainSize - 1], X[1: trainSize];
    X_test, Y_test = X[trainSize: -1], X[trainSize + 1:];

    batchSize, inputSize, hiddenSize, maxEpoch = 4, X.shape[-1], 128, 200;
    model = LMTSModel(inputSize, hiddenSize, inputSize, layerNum = 2, inputDropout = 0, recurrentDropout = 0);

    # if os.path.isfile(filename):
    #     with open(filename, "br") as file:
    #         params = pickle.load(file);
    #     if isinstance(params[0], cp.ndarray):
    #         params = [cp.asnumpy(p) for p in params];
    #     model.params = params;
    #
    # model.reset();
    # testSize = len(X_train) // stepSize * stepSize;
    # Y_train_hat = np.concatenate(tuple([model.forward(*item)[0] for item in PartitionedDataIterator([X_train[: testSize]], 1, stepSize, randomOffset = False)]), axis = 1);
    # Y_train_hat = scaler.inverse(np.squeeze(Y_train_hat, axis = 0));
    # Y_train_real = scaler.inverse(Y_train[: testSize]);
    # print(f"train 1-step MAE: {meanAbsoluteError(Y_train_real, Y_train_hat)}");

    # idx = np.random.randint(0, testSize - predictSize, 20);
    # for i in idx.tolist():
    #     plt.figure(1, (14, 8));
    #     plt.plot(np.arange(predictSize), Y_train_real[i: i + predictSize].flatten(), "D-b", label = "real data");
    #     plt.plot(np.arange(predictSize), Y_train_hat[i: i + predictSize].flatten(), "s-r", label = "predict data");
    #     plt.legend(loc = 'upper right');
    #     plt.show(block = True);
    #     plt.close();

    # model.reset();
    # testSize = len(X_test) // stepSize * stepSize;
    # Y_test_hat = np.concatenate(tuple([model.forward(*item)[0] for item in PartitionedDataIterator([X_test[: testSize]], 1, stepSize, randomOffset = False)]), axis = 1);
    # Y_test_hat = scaler.inverse(np.squeeze(Y_test_hat, axis = 0));
    # Y_test_real = scaler.inverse(Y_test[: testSize]);
    # print(f"test 1-step MAE: {meanAbsoluteError(Y_test_real, Y_test_hat)}");

    # idx = np.random.randint(0, testSize - predictSize, 20);
    # for i in idx.tolist():
    #     plt.figure(1, (14, 8));
    #     plt.plot(np.arange(predictSize), Y_test_real[i: i + predictSize].flatten(), "D-b", label = "real data");
    #     plt.plot(np.arange(predictSize), Y_test_hat[i: i + predictSize].flatten(), "s-r", label = "predict data");
    #     plt.legend(loc = 'upper right');
    #     plt.show(block = True);
    #     plt.close();

    # model.reset();
    # heatSize = 1 * stepSize;
    # X_predict, Y_predict = lagAndForecast(X_test, heatSize, predictSize);
    # Y_predict_hat = np.concatenate(tuple(model.predict(SequentialDataIterator([X_predict], 128, shuffle = False), predictSize)), axis = 0);
    # X_predict, Y_predict, Y_predict_hat = scaler.inverse(X_predict), scaler.inverse(Y_predict), scaler.inverse(Y_predict_hat);
    # print(f"test {predictSize}-step MAE: {meanAbsoluteError(Y_predict, Y_predict_hat)}");
    #
    # idx = np.random.randint(0, len(X_predict), 20);
    # for i in idx.tolist():
    #     x = np.concatenate((X_predict[i], Y_predict[i]), axis = None)[-100:];
    #
    #     plt.figure(1, (14, 8));
    #     plt.plot(np.arange(len(x)), x, "D-b", label = "real data");
    #     plt.plot(np.arange(len(x) - predictSize, len(x)), Y_predict_hat[i].flatten(), "s-r", label = "predict data");
    #     plt.legend(loc = 'upper left');
    #     plt.show(block = True);
    #     plt.close();

    # lossFunc = IdentityWithMeanSquareLoss();
    # optimizer = GradientsClipping(5.0, Adam());
    # trainingIterator = PartitionedDataIterator([X_train, Y_train], batchSize, stepSize);
    # testIterator = PartitionedDataIterator([X_test, Y_test], 1, stepSize, randomOffset = False);
    # evaluator = MaeAccuracyEvaluator();
    # model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, minEpoch = 5, testIterator = testIterator, evaluator = evaluator, plot = True);
    #
    # with open(filename, "wb") as file:
    #     pickle.dump(model.params, file);

    print("exit.");


def testTS3_LM_N1_1step():
    targetTag = "TI6116A.PV";
    negativeTags = [];  # ['XIP301A.PV', 'DI6103.PV', 'FIC6031.SV', 'PDI6105.PV', 'DI6104.PV', 'JI6601.PV', 'TIC6522.SV'];
    positiveTags = ['TI6116B.PV', 'TIC6115.PV', 'TI6106.PV', 'TI6901.PV', 'TI6166A.PV', 'TE6658.PV', 'TI6107D.PV', 'TI6107C.PV', 'TI6107H.PV', 'TI6107G.PV'];

    data, marks = loadTSData1AndMany(targetTag, negativeTags, positiveTags);
    segments = loadDataset(data, marks);
    dataSet = segments[-1];

    scaler = StandardScaler();
    scaler.fit(dataSet);

    filename = "lstm3_N1.weights";
    trainSize, stepSize, predictSize = int(len(dataSet) * 0.8), 240, 10;

    X = scaler.transform(dataSet);
    X_train, Y_train = X[:trainSize - 1], X[1: trainSize, [-1]];
    X_test, Y_test = X[trainSize: -1], X[trainSize + 1:, [-1]];

    batchSize, inputSize, hiddenSize, outputSize, maxEpoch = math.ceil(len(X_train) / (7 * 480)), X_train.shape[-1], 128, Y_train.shape[-1], 200;
    model = LMTSModel(inputSize, hiddenSize, outputSize, layerNum = 2, inputDropout = 0, recurrentDropout = 0);

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params = pickle.load(file);
        if isinstance(params[0], cp.ndarray):
            params = [cp.asnumpy(p) for p in params];
        model.params = params;

    model.reset();
    testSize = len(X_train) // stepSize * stepSize;
    Y_train_hat = np.concatenate(tuple([model.forward(*item)[0] for item in PartitionedDataIterator([X_train[: testSize]], 1, stepSize, randomOffset = False)]), axis = 1);
    Y_train_hat = scaler.inverse(np.squeeze(Y_train_hat, axis = 0), index = -1);
    Y_train_real = scaler.inverse(Y_train[: testSize], index = -1);
    print(f"train 1-step MAE: {meanAbsoluteError(Y_train_real, Y_train_hat)}");

    # idx = np.random.randint(0, testSize - predictSize, 20);
    # for i in idx.tolist():
    #     plt.figure(1, (14, 8));
    #     plt.plot(np.arange(predictSize), Y_train_real[i: i + predictSize].flatten(), "D-b", label = "real data");
    #     plt.plot(np.arange(predictSize), Y_train_hat[i: i + predictSize].flatten(), "s-r", label = "predict data");
    #     plt.legend(loc = 'upper right');
    #     plt.show(block = True);
    #     plt.close();

    model.reset();
    testSize = len(X_test) // stepSize * stepSize;
    Y_test_hat = np.concatenate(tuple([model.forward(*item)[0] for item in PartitionedDataIterator([X_test[: testSize]], 1, stepSize, randomOffset = False)]), axis = 1);
    Y_test_hat = scaler.inverse(np.squeeze(Y_test_hat, axis = 0), index = -1);
    Y_test_real = scaler.inverse(Y_test[: testSize], index = -1);
    print(f"test 1-step MAE: {meanAbsoluteError(Y_test_real, Y_test_hat)}");

    # idx = np.random.randint(0, testSize - predictSize, 20);
    # for i in idx.tolist():
    #     plt.figure(1, (14, 8));
    #     plt.plot(np.arange(predictSize), Y_test_real[i: i + predictSize].flatten(), "D-b", label = "real data");
    #     plt.plot(np.arange(predictSize), Y_test_hat[i: i + predictSize].flatten(), "s-r", label = "predict data");
    #     plt.legend(loc = 'upper right');
    #     plt.show(block = True);
    #     plt.close();

    model.reset();
    heatSize = 1 * stepSize;
    X_predict, Y_predict = lagAndForecast(X_test, heatSize, predictSize);
    Y_predict_hat = np.concatenate(tuple(model.predict(SequentialDataIterator([X_predict], 128, shuffle = False), predictSize)), axis = 0);
    X_predict, Y_predict, Y_predict_hat = scaler.inverse(X_predict, index = -1), scaler.inverse(Y_predict, index = -1), scaler.inverse(Y_predict_hat, index = -1);
    print(f"test {predictSize}-step MAE: {meanAbsoluteError(Y_predict, Y_predict_hat)}");

    idx = np.random.randint(0, len(X_predict), 20);
    for i in idx.tolist():
        x = np.concatenate((X_predict[i], Y_predict[i]), axis = None)[-100:];

        plt.figure(1, (14, 8));
        plt.plot(np.arange(len(x)), x, "D-b", label = "real data");
        plt.plot(np.arange(len(x) - predictSize, len(x)), Y_predict_hat[i].flatten(), "s-r", label = "predict data");
        plt.legend(loc = 'upper left');
        plt.show(block = True);
        plt.close();

    # lossFunc = IdentityWithMeanSquareLoss();
    # optimizer = GradientsClipping(5.0, Adam());
    # trainingIterator = PartitionedDataIterator([X_train, Y_train], batchSize, stepSize);
    # testIterator = PartitionedDataIterator([X_test, Y_test], 1, stepSize, randomOffset = False);
    # evaluator = MaeAccuracyEvaluator();
    # model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, minEpoch = 5, testIterator = testIterator, evaluator = evaluator, plot = True);
    #
    # with open(filename, "wb") as file:
    #     pickle.dump(model.params, file);

    print("exit.");


def testTS4_LM_Nstep():
    TI6116A = np.load("TI6116A.npy");
    scaler = StandardScaler();
    scaler.fit(TI6116A);

    filename = "lstm4.weights";
    trainSize, stepSize, predictSize = int(len(TI6116A) * 0.8), 240, 10;

    X = scaler.transform(TI6116A);
    batchSize, inputSize, hiddenSize, maxEpoch = 4, X.shape[-1], 128, 100;
    models = [LMTSModel(inputSize, hiddenSize, inputSize, layerNum = 1, inputDropout = 0, recurrentDropout = 0) for _ in range(predictSize)];

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params = pickle.load(file);
        for m, ps in zip(models, params):
            m.params = ps;

    heatSize = 1 * stepSize;
    X_test = X[trainSize:];
    X_predict, Y_predict = lagAndForecast(X_test, heatSize, predictSize);
    Y_predict_hat = np.concatenate(tuple([np.concatenate(tuple(m.predict(SequentialDataIterator([X_predict], 128, shuffle = False), 1)), axis = 0) for m in models]), axis = 1);
    X_predict, Y_predict, Y_predict_hat = scaler.inverse(X_predict), scaler.inverse(Y_predict), scaler.inverse(Y_predict_hat);
    for i in range(1, predictSize + 1):
        print(f"test {i}-step MAE: {meanAbsoluteError(Y_predict[:, : i], Y_predict_hat[:, : i])}");

    idx = np.random.randint(0, len(X_predict), 40);
    for i in idx.tolist():
        x = np.concatenate((X_predict[i], Y_predict[i]), axis = None)[-50:];

        plt.figure(1, (14, 8));
        plt.plot(np.arange(len(x)), x, "D-b", label = "real data");
        plt.plot(np.arange(len(x) - predictSize, len(x)), Y_predict_hat[i].flatten(), "s-r", label = "predict data");
        plt.legend(loc = 'upper left');
        plt.show(block = True);
        plt.close();

    # for i, model in enumerate(models):
    #     gap = i + 1;
    #     X_train, Y_train = X[:trainSize - gap], X[gap: trainSize];
    #     X_test, Y_test = X[trainSize: -gap], X[trainSize + gap:];
    #
    #     lossFunc = IdentityWithMeanSquareLoss();
    #     optimizer = GradientsClipping(5.0, Adam());
    #     trainingIterator = PartitionedDataIterator([X_train, Y_train], batchSize, stepSize);
    #     testIterator = PartitionedDataIterator([X_test, Y_test], 1, stepSize, randomOffset = False);
    #     evaluator = MaeAccuracyEvaluator();
    #     model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, minEpoch = 5, testIterator = testIterator, evaluator = evaluator, plot = True);
    #     print(f"complete to train model {i}\n\n");
    #
    # with open(filename, "wb") as file:
    #     pickle.dump([m.params for m in models], file);

    print("exit.");


def testTS5_LM_1step_All():
    batchStepSize = int(1440 / 3 * 7);
    TI6116A = np.load("/media/WindowsE/Data/PARS/JNLH/AiModel/isys_data_20210701_20220401_180/__JNRTDB_TI6116A.PV.npy");
    datasets = loadDataset(TI6116A[:, 0].reshape(-1, 1), TI6116A[:, 1].reshape(-1, 1), batchStepSize);
    scaler = StandardScaler();
    scaler.fit(np.concatenate(tuple(datasets), axis = 0));

    filename = "lstm5.weights";
    stepSize, predictSize = 240, 10;

    inputSize, hiddenSize, minEpoch, maxEpoch = 1, 128, 5, 50;
    trainSets, testSet = [scaler.transform(d) for d in datasets[:-1]], scaler.transform(datasets[-1]);
    model = LMTSModel(inputSize, hiddenSize, inputSize, layerNum = 1, inputDropout = 0, recurrentDropout = 0);

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params = pickle.load(file);
        if isinstance(params[0], cp.ndarray):
            params = [cp.asnumpy(p) for p in params];
        model.params = params;

    X_test, Y_test = testSet[: -1], testSet[1:];

    model.reset();
    testSize = len(X_test) // stepSize * stepSize;
    Y_test_hat = np.concatenate(tuple([model.forward(*item)[0] for item in PartitionedDataIterator([X_test[: testSize]], 1, stepSize, randomOffset = False)]), axis = 1);
    Y_test_hat = scaler.inverse(np.squeeze(Y_test_hat, axis = 0));
    Y_test_real = scaler.inverse(Y_test[: testSize]);
    print(f"test 1-step MAE: {meanAbsoluteError(Y_test_real, Y_test_hat)}");

    # idx = np.random.randint(0, testSize - predictSize, 20);
    # for i in idx.tolist():
    #     plt.figure(1, (14, 8));
    #     plt.plot(np.arange(predictSize), Y_test_real[i: i + predictSize].flatten(), "D-b", label = "real data");
    #     plt.plot(np.arange(predictSize), Y_test_hat[i: i + predictSize].flatten(), "s-r", label = "predict data");
    #     plt.legend(loc = 'upper right');
    #     plt.show(block = True);
    #     plt.close();

    model.reset();
    heatSize = 1 * stepSize;
    X_predict, Y_predict = lagAndForecast(X_test, heatSize, predictSize);
    Y_predict_hat = np.concatenate(tuple(model.predict(SequentialDataIterator([X_predict], 128, shuffle = False), predictSize)), axis = 0);
    X_predict, Y_predict, Y_predict_hat = scaler.inverse(X_predict), scaler.inverse(Y_predict), scaler.inverse(Y_predict_hat);
    print(f"test {predictSize}-step MAE: {meanAbsoluteError(Y_predict, Y_predict_hat)}");

    idx = np.random.randint(0, len(X_predict), 40);
    for i in idx.tolist():
        x = np.concatenate((X_predict[i], Y_predict[i]), axis = None)[-100:];

        plt.figure(1, (14, 8));
        plt.plot(np.arange(len(x)), x, "D-b", label = "real data");
        plt.plot(np.arange(len(x) - predictSize, len(x)), Y_predict_hat[i].flatten(), "s-r", label = "predict data");
        plt.legend(loc = 'upper left');
        plt.show(block = True);
        plt.close();

    # lossFunc = IdentityWithMeanSquareLoss();
    # optimizer = GradientsClipping(5.0, Adam());
    # testIterator = PartitionedDataIterator([testSet[: -1], testSet[1:]], 1, stepSize, randomOffset = False);
    # evaluator = MaeAccuracyEvaluator();
    # params, accuracyData, lossValues, lossData = [], [], [], [];
    # for epoch in range(maxEpoch):
    #     lossValues.clear();
    #
    #     for X in trainSets:
    #         X_train, Y_train = X[: -1], X[1:];
    #         trainingIterator = PartitionedDataIterator([X_train, Y_train], len(X_train) // batchStepSize, stepSize);
    #         lossValues.extend(model.fit(trainingIterator, lossFunc, optimizer, 1, plot = False)[0]);
    #
    #     params.append([np.copy(p) for p in model.params]);
    #     lossData.append(sum(lossValues) / len(lossValues));
    #     accuracyData.append(model.eval(lossFunc, evaluator, iterator = testIterator));
    #     print(f"epoch {epoch}, total average loss: {lossData[-1]}, test {evaluator.name}: {accuracyData[-1]}\n\n");
    #
    # model.params = params[accuracyData[minEpoch:].index(min(accuracyData[minEpoch:])) + minEpoch];
    # with open(filename, "wb") as file:
    #     pickle.dump(model.params, file);
    #
    # fig = plt.figure(1);
    #
    # ax1 = fig.add_subplot(111);
    # ax1.set_xlabel("epoch");
    # ax1.set_ylabel('loss');
    # ax1.plot(lossData, "o-k", label = "loss");
    #
    # ax2 = ax1.twinx();
    # ax2.set_ylabel('accuracy');
    # ax2.plot(accuracyData, "s-r", label = f"test {evaluator.name}");
    #
    # fig.legend(loc = "upper right", bbox_to_anchor = (1, 1), bbox_transform = ax1.transAxes)
    # plt.show(block = True);
    # plt.close();

    print("exit.");


def testAE_NASABearing():
    # data_dir = '/media/WindowsE/Data/Kaggle/NASA Bearing Dataset/2nd_test';
    # merged_data = pd.DataFrame()
    #
    # for filename in os.listdir(data_dir):
    #     dataset = pd.read_csv(os.path.join(data_dir, filename), sep = '\t')
    #     dataset_mean_abs = np.array(dataset.abs().mean())
    #     dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1, 4))
    #     dataset_mean_abs.index = [filename]
    #     merged_data = merged_data.append(dataset_mean_abs)
    # merged_data.reset_index(inplace = True)  # reset index to get datetime as columns
    # merged_data.columns = ['Datetime', 'Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']  # rename columns
    # merged_data.sort_values(by = 'Datetime', inplace = True)
    # merged_data.to_csv('2nd_test_resmaple_10minutes.csv')

    merged_data = pd.read_csv("2nd_test_resmaple_10minutes.csv", index_col='Datetime', usecols = ['Datetime','Bearing 1','Bearing 2','Bearing 3','Bearing 4']);
    merged_data.index = pd.to_datetime(merged_data.index, format = "%Y.%m.%d.%H.%M.%S");
    print(merged_data.head());

    dataset_train = merged_data["2004-02-12 11:02:39" : "2004-02-13 23:52:39"];
    dataset_test = merged_data["2004-02-13 23:52:39": ];
    # dataset_train.plot(figsize = (12,6));
    # dataset_test.plot(figsize = (12, 6));

    trainSize = int(len(dataset_train) * 0.95);
    X_train, X_validate, X_test = dataset_train[: trainSize].to_numpy(), dataset_train[trainSize: ].to_numpy(), dataset_test.to_numpy();
    scaler = MinMaxScaler();
    scaler.fit(X_train);
    X_train, X_validate, X_test = scaler.transform(X_train), scaler.transform(X_validate), scaler.transform(X_test);

    filename = "2nd_test_resmaple_10minutes.weights";
    inputSize, hiddenSize1, hiddenSize2, batchSize, maxEpoch = X_train.shape[-1], 10, 2, 10, 100;
    model = SequentialContainer(
        AffineLayer(inputSize, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, hiddenSize2),
        ReluLayer(),
        AffineLayer(hiddenSize2, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, inputSize),
        SigmoidLayer(),
    );

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params = pickle.load(file);
        if isinstance(params[0], cp.ndarray):
            params = [cp.asnumpy(p) for p in params];
        model.params = params;

    X_train_hat = np.concatenate(tuple([item[0] for item in model.predict(SequentialDataIterator([X_train, X_train], batchSize, shuffle = False))]), axis = 0);
    X_train_score = np.mean(np.abs(X_train_hat - X_train), axis = -1);
    plt.hist(X_train_score);
    plt.show(block = True);

    threshold = 0.3;
    X_test_hat = np.concatenate(tuple([item[0] for item in model.predict(SequentialDataIterator([X_test, X_test], batchSize, shuffle = False))]), axis = 0);
    X_test_score = np.mean(np.abs(X_test_hat - X_test), axis = -1);
    plt.plot(np.log(X_test_score));
    plt.axhline(y = math.log(threshold), color = "red");
    plt.show(block = True);

    # lossFunc = IdentityWithMeanSquareLoss();
    # optimizer = Adam();
    # trainingIterator = SequentialDataIterator([X_train, X_train], batchSize);
    # testIterator = SequentialDataIterator([X_validate, X_validate], 1);
    # evaluator = MaeAccuracyEvaluator();
    # model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, testIterator = testIterator, evaluator = evaluator, plot = True);
    #
    # with open(filename, "wb") as file:
    #     pickle.dump(model.params, file);

    print("exit.");


def testAE_TagData():
    # tagNames = ["TE6658.PV", "PT6621.PV", "JI6601.PV", "TE6648A.PV", "ZE6603B.PV", "ZE6603A.PV", "VE6605X.PV", "VE6605Y.PV", "VE6606Y.PV", "VE6606X.PV", "TE6649A.PV"];
    tagNames = ["TI6113A.PV", "TI6116A.PV", "TIC6115.PV", "AI6103.PV"];
    data, marks = loadRawTSData(tagNames);
    X = data[np.all(marks == 1, axis = -1)];
    np.random.shuffle(X);

    # Q1, Q3 = np.quantile(X, 0.25, axis = 0), np.quantile(X, 0.75, axis = 0);
    # mask = np.logical_and(X >= Q1 - 1.5 * (Q3 - Q1), X <= Q3 + 1.5 * (Q3 - Q1));
    # X_normal = X[np.all(mask, axis = -1)];
    # trainSize, anomalySize = int(len(X_normal) * 0.8), np.sum(~np.all(mask, axis = -1));
    # X_train, X_test = X_normal[: trainSize], np.concatenate((X_normal[trainSize: ], X[~np.all(mask, axis = -1)]), axis = 0);
    X_train, X_test = X, None;

    scaler = MinMaxScaler();
    scaler.fit(X_train);
    X_train, X_test = scaler.transform(X_train), (scaler.transform(X_test) if X_test is not None else None);

    filename = "AE_tagData.weights";
    D = X_train.shape[-1];
    inputSize, hiddenSize1, hiddenSize2, batchSize, maxEpoch = D, 2 * D, math.ceil(0.3 * D), 32, 100;
    model = SequentialContainer(
        AffineLayer(inputSize, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, hiddenSize2),
        ReluLayer(),
        AffineLayer(hiddenSize2, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, inputSize),
        SigmoidLayer(),
    );

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params = pickle.load(file);
        if isinstance(params[0], cp.ndarray):
            params = [cp.asnumpy(p) for p in params];
        model.params = params;

    X_train_hat = np.concatenate(tuple([item[0] for item in model.predict(SequentialDataIterator([X_train, X_train], batchSize, shuffle = False))]), axis = 0);
    X_train_score = np.mean(np.abs(X_train_hat - X_train), axis = -1);
    plt.hist(X_train_score, bins = 1000);
    plt.show(block = True);
    plt.plot(X_train_score);
    plt.show(block = True);
    #
    # threshold = 0.2;
    # X_test_hat = np.concatenate(tuple([item[0] for item in model.predict(SequentialDataIterator([X_test, X_test], batchSize, shuffle = False))]), axis = 0);
    # X_test_score = np.mean(np.abs(X_test_hat - X_test), axis = -1);
    # plt.plot(np.log(X_test_score));
    # plt.axhline(y = math.log(threshold), color = "red");
    # plt.show(block = True);

    # lossFunc = IdentityWithMeanSquareLoss();
    # optimizer = Adam();
    # trainingIterator = SequentialDataIterator([X_train, X_train], batchSize);
    # testIterator = SequentialDataIterator([X_test, X_test], batchSize) if X_test is not None else None;
    # evaluator = MaeAccuracyEvaluator();
    # model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, testIterator = testIterator, evaluator = evaluator, plot = True);
    #
    # with open(filename, "wb") as file:
    #     pickle.dump(model.params, file);

    print("exit.");


def testIF_TagData():
    # dataName = "主反原料带水";
    # tagNames = ["FIC6421.PV", "FI6241.PV", "TIC6421.PV", "PI6410.PV", "FIC6419.PV", "TIC6201.PV", "TIC6414.PV"];
    # dataName = "副反原料带水";
    # tagNames = ["FIC6451.PV", "TIC6208.PV", "FIQ6243.PV", "FIC6201.PV", "PI6102B.PV", "TIC6101.PV"];
    # dataName = "一再尾燃";
    # tagNames = ["TI6113A.PV", "TI6116A.PV", "TIC6115.PV", "AI6103.PV"];
    # dataName = "增压机停机";
    # tagNames = ["TIC6102.PV", "LIC6101.PV", "FIC6121.PV", "FIC6104.PV", "TI6131A.PV", "PI6139.PV", "PI6133.PV", "TI6110.PV", "FT6702.PV", "TI6109.PV", "TI6192.PV", "TI6113A.PV", "PI6138.PV", "FIC6122.PV", "PT6707.PV", "LIC6102.PV", "TI6132.PV"];
    # dataName = "二再碳堆积";
    # tagNames = ["TIC6102.PV", "FIC6101D.PV", "FIC6418.PV", "FIQ6316.PV", "FI6241.PV", "LI6219.PV", "FIC6101C.PV", "FIC6328.PV", "FIC6211.PV", "LIC6270.PV", "TIC6201.PV", "FIQ6315.PV", "FIC6270.PV", "TI6105.PV", "TI6131A.PV", "FIQ6317.PV", "FIC6101B.PV", "FIC6309.PV", "FIC6101A.PV", "TIC6414.PV", "TI6414D.PV", "LIC6206.PV", "TIC6208.PV", "FI6313.PV", "TIC6101.PV", "FI6802.PV", "FIC6276.PV", "LI6212.PV", "FI6312.PV", "LIC6307.PV", "FIC6417.PV", "FIC6416.PV"];
    # dataName = "主原料中断";
    # tagNames = ["FI6180C.PV", "TI6417E.PV", "FI6180A.PV", "FIC6421.PV", "TIC6421.PV", "TI6410.PV", "TI6113A.PV", "FIC6419.PV", "TI6417F.PV", "TI6417C.PV", "TI6417B.PV", "FI6180D.PV", "FT6801.PV", "TI6255.PV", "TI6417A.PV", "TI6417D.PV", "PI6410.PV", "TI6426.PV", "TIC6414.PV", "TI6420.PV", "FI6180B.PV"];
    # dataName = "副原料中断";
    # tagNames = ["TI6103.PV", "FI6453D.PV", "FT6801.PV", "FI6453C.PV", "PI6102B.PV", "FIC6451.PV", "TI6113A.PV", "TI6454.PV", "TI6108.PV", "TIC6101.PV", "FI6453A.PV", "FI6453B.PV", "FIC6452.PV"];
    # dataName = "烟机异常振动";
    # tagNames = ["ZE6603B.PV", "TE6658.PV", "VE6606Y.PV", "ZE6603A.PV", "TE6648A.PV", "VE6606X.PV", "VE6605X.PV", "VE6605Y.PV", "TE6649A.PV", "PT6620.PV", "JI6601.PV", "PT6621.PV"];
    dataName = "PK301停机";
    tagNames = ["9PIC341.PV", "9PIC311.PV"];
    data, marks = loadRawTSData(tagNames);
    dataset = data[np.all(marks == 1, axis = -1)];

    trainAllData = True;
    filename = f"IF_tagData_{dataName}_{trainAllData}.result";
    batchSize = 32;
    model = IsolationForest(200, 512, ProportionThresholdFinder(1/365.0));

    # Q = np.quantile(dataset, [0.25, 0.75], axis = 0);
    # Q1, Q3 = Q[0], Q[1];
    # IQR = Q3 - Q1;
    # mask = np.all(np.logical_and(dataset >= Q1 - 5 * IQR, dataset <= Q3 + 5 * IQR), axis = -1);
    # X = dataset[mask];
    # np.random.shuffle(X);
    # trainSize = int(len(X) * 0.8);
    # X_train, X_test_normal, X_test_anomaly = X[: trainSize], X[trainSize:], dataset[~mask];
    #
    # if trainAllData:
    #     X_train = np.concatenate((X_train, X_test_normal, X_test_anomaly), axis = 0);
    #
    # print(f"train IF: {dataName}");
    # model.fill(np.mat(X_train));
    # with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
    #     X_train_rp = -np.array(pool.map(model.getAnomalyScore, [item for item in np.mat(X_train)]));
    # with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
    #     X_test_normal_rp = -np.array(pool.map(model.getAnomalyScore, [item for item in np.mat(X_test_normal)]));
    # with multiprocessing.Pool(max(1, psutil.cpu_count(False) - 2)) as pool:
    #     X_test_anomaly_rp = -np.array(pool.map(model.getAnomalyScore, [item for item in np.mat(X_test_anomaly)]));
    # with open(filename, "wb") as file:
    #     pickle.dump((model, X_train, X_test_normal, X_test_anomaly, X_train_rp, X_test_normal_rp, X_test_anomaly_rp), file);

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            model, X_train, X_test_normal, X_test_anomaly, X_train_rp, X_test_normal_rp, X_test_anomaly_rp = pickle.load(file);

    ratio = 1 / 365.0;
    X_all_rp = X_train_rp if trainAllData else np.concatenate((X_test_anomaly_rp, X_test_normal_rp, X_train_rp), axis = 0);
    X_all = X_train if trainAllData else np.concatenate((X_test_anomaly, X_test_normal, X_train), axis = 0);
    threshold = np.quantile(X_all_rp, ratio);
    X_anomaly = X_all[np.argwhere(X_all_rp <= threshold)[:, 0]];

    filename = f"IF_tagData_{dataName}_{trainAllData}.csv";
    content = [",".join(tagNames)];
    for item in X_anomaly:
        content.append(",".join([str(v) for v in item]));
    with open(filename, "wt", encoding = "utf-8") as file:
        file.write("\n".join(content));
    print(f"anomaly ratio is {ratio}, the anomaly data saved to file: {filename}");

    plt.figure(figsize = (12, 14));
    bins = plt.hist(X_train_rp, bins = 1000, color = "k");
    plt.hist(X_test_normal_rp, bins = 1000, label = "normal", color = "b");
    # plt.hist(X_test_anomaly_rp, bins = 1000, label = "normal", color = "r");
    plt.show(block = True);

    fprX, tprY = [0], [0];
    for alpha in np.arange(0.001, 1, 0.001):
        threshold = np.quantile(X_train_rp, alpha);
        X_test_normal_class = X_test_normal_rp < threshold;
        X_test_anomaly_class = X_test_anomaly_rp < threshold;

        TP, FP = np.sum(X_test_anomaly_class == True), np.sum(X_test_normal_class == True);
        FN, TN = np.sum(X_test_anomaly_class == False), np.sum(X_test_normal_class == False);
        precision, recall = TP / (TP + FP), TP / (TP + FN);
        fpr, tpr = FP / (FP + TN), recall;
        fprX.append(fpr);
        tprY.append(tpr);
        print(f"alpha = {alpha}, precision = {precision}, recall = {recall}, fpr = {fpr}, tpr = {tpr}");
    fprX.append(1);
    tprY.append(1);

    plt.figure(figsize = (12, 14));
    plt.plot(fprX, tprY);
    plt.xlabel("FPR");
    plt.ylabel("TPR");
    plt.title(f"P = {len(X_test_anomaly)}, AUC = {auc(-X_test_anomaly_rp, -X_test_normal_rp)}");
    plt.show(block = True);

    print("exit.");


def testGMM_TagData():
    dataName = "主反原料带水";
    tagNames = ["FIC6421.PV", "FI6241.PV", "TIC6421.PV", "PI6410.PV", "FIC6419.PV", "TIC6201.PV", "TIC6414.PV"];
    # dataName = "副反原料带水";
    # tagNames = ["FIC6451.PV", "TIC6208.PV", "FIQ6243.PV", "FIC6201.PV", "PI6102B.PV", "TIC6101.PV"];
    # dataName = "一再尾燃";
    # tagNames = ["TI6113A.PV", "TI6116A.PV", "TIC6115.PV", "AI6103.PV"];
    # dataName = "增压机停机";
    # tagNames = ["TIC6102.PV", "LIC6101.PV", "FIC6121.PV", "FIC6104.PV", "TI6131A.PV", "PI6139.PV", "PI6133.PV", "TI6110.PV", "FT6702.PV", "TI6109.PV", "TI6192.PV", "TI6113A.PV", "PI6138.PV", "FIC6122.PV", "PT6707.PV", "LIC6102.PV", "TI6132.PV"];
    # dataName = "二再碳堆积";
    # tagNames = ["TIC6102.PV", "FIC6101D.PV", "FIC6418.PV", "FIQ6316.PV", "FI6241.PV", "LI6219.PV", "FIC6101C.PV", "FIC6328.PV", "FIC6211.PV", "LIC6270.PV", "TIC6201.PV", "FIQ6315.PV", "FIC6270.PV", "TI6105.PV", "TI6131A.PV", "FIQ6317.PV", "FIC6101B.PV", "FIC6309.PV", "FIC6101A.PV", "TIC6414.PV", "TI6414D.PV", "LIC6206.PV", "TIC6208.PV", "FI6313.PV", "TIC6101.PV", "FI6802.PV", "FIC6276.PV", "LI6212.PV", "FI6312.PV", "LIC6307.PV", "FIC6417.PV", "FIC6416.PV"];
    # dataName = "主原料中断";
    # tagNames = ["FI6180C.PV", "TI6417E.PV", "FI6180A.PV", "FIC6421.PV", "TIC6421.PV", "TI6410.PV", "TI6113A.PV", "FIC6419.PV", "TI6417F.PV", "TI6417C.PV", "TI6417B.PV", "FI6180D.PV", "FT6801.PV", "TI6255.PV", "TI6417A.PV", "TI6417D.PV", "PI6410.PV", "TI6426.PV", "TIC6414.PV", "TI6420.PV", "FI6180B.PV"];
    # dataName = "副原料中断";
    # tagNames = ["TI6103.PV", "FI6453D.PV", "FT6801.PV", "FI6453C.PV", "PI6102B.PV", "FIC6451.PV", "TI6113A.PV", "TI6454.PV", "TI6108.PV", "TIC6101.PV", "FI6453A.PV", "FI6453B.PV", "FIC6452.PV"];
    # dataName = "烟机异常振动";
    # tagNames = ["ZE6603B.PV", "TE6658.PV", "VE6606Y.PV", "ZE6603A.PV", "TE6648A.PV", "VE6606X.PV", "VE6605X.PV", "VE6605Y.PV", "TE6649A.PV", "PT6620.PV", "JI6601.PV", "PT6621.PV"];
    # dataName = "PK301停机";
    # tagNames = ["9PIC341.PV", "9PIC311.PV"];
    data, marks = loadRawTSData(tagNames);
    dataset = data[np.all(marks == 1, axis = -1)];

    filename = f"GMM_tagData_{dataName}.result";
    batchSize = 32;
    scaler = StandardScaler();
    model = GaussianMixture();

    Q = np.quantile(dataset, [0.25, 0.75], axis = 0);
    Q1, Q3 = Q[0], Q[1];
    IQR = Q3 - Q1;
    mask = np.all(np.logical_and(dataset >= Q1 - 5 * IQR, dataset <= Q3 + 5 * IQR), axis = -1);
    X = dataset[mask];
    np.random.shuffle(X);
    trainSize = int(len(X) * 0.8);
    X_train, X_test_normal, X_test_anomaly = X[: trainSize], X[trainSize:], dataset[~mask];

    scaler.fit(X_train);
    X_train, X_test_normal, X_test_anomaly = scaler.transform(X_train), scaler.transform(X_test_normal), scaler.transform(X_test_anomaly);

    print(f"train GMM: {dataName}");
    model = GaussianMixture.optimalK(X_train, 100, verbose = True);
    with open(filename, "wb") as file:
        pickle.dump((model.params, scaler.params, X_train, X_test_normal, X_test_anomaly), file);

    # if os.path.isfile(filename):
    #     with open(filename, "br") as file:
    #         modelParams, scalerParams, X_train, X_test_normal, X_test_anomaly = pickle.load(file);
    #     if isinstance(modelParams[0], cp.ndarray):
    #         modelParams = [cp.asnumpy(p) for p in modelParams];
    #     scaler.params = scalerParams;
    #     model.params = modelParams;
    #
    # ratio = 1 / 365.0;
    # X_test_normal_rp = model.logPdf(X_test_normal);
    # X_test_anomaly_rp = model.logPdf(X_test_anomaly);
    # X_train_rp = np.concatenate(tuple([model.logPdf(item[0]) for item in SequentialDataIterator([X_train], batchSize, False)]), axis = 0);
    # X_all_rp = np.concatenate((X_test_anomaly_rp, X_test_normal_rp, X_train_rp), axis = 0);
    # X_all = np.concatenate((X_test_anomaly, X_test_normal, X_train), axis = 0);
    # threshold = np.quantile(X_all_rp, ratio);
    # X_anomaly = X_all[np.argwhere(X_all_rp <= threshold)[:, 0]];
    #
    # filename = f"GMM_tagData_{dataName}.csv";
    # X_anomaly = scaler.inverse(X_anomaly);
    # content = [",".join(tagNames)];
    # for item in X_anomaly:
    #     content.append(",".join([str(v) for v in item]));
    # with open(filename, "wt", encoding = "utf-8") as file:
    #     file.write("\n".join(content));
    # print(f"anomaly ratio is {ratio}, the anomaly data saved to file: {filename}");
    #
    # plt.figure(figsize = (12, 14));
    # bins = plt.hist(X_train_rp, bins = 1000, color = "k");
    # plt.hist(X_test_normal_rp, bins = 1000, label = "normal", color = "b");
    # # plt.hist(X_test_anomaly_rp, bins = 1000, label = "normal", color = "r");
    # plt.show(block = True);
    #
    # fprX, tprY = [0], [0];
    # for alpha in np.arange(0.001, 1, 0.001):
    #     threshold = np.quantile(X_train_rp, alpha);
    #     X_test_normal_class = X_test_normal_rp < threshold;
    #     X_test_anomaly_class = X_test_anomaly_rp < threshold;
    #
    #     TP, FP = np.sum(X_test_anomaly_class == True), np.sum(X_test_normal_class == True);
    #     FN, TN = np.sum(X_test_anomaly_class == False), np.sum(X_test_normal_class == False);
    #     precision, recall = TP / (TP + FP), TP / (TP + FN);
    #     fpr, tpr = FP / (FP + TN), recall;
    #     fprX.append(fpr);
    #     tprY.append(tpr);
    #     print(f"alpha = {alpha}, precision = {precision}, recall = {recall}, fpr = {fpr}, tpr = {tpr}");
    # fprX.append(1);
    # tprY.append(1);
    #
    # plt.figure(figsize = (12, 14));
    # plt.plot(fprX, tprY);
    # plt.xlabel("FPR");
    # plt.ylabel("TPR");
    # plt.title(f"P = {len(X_test_anomaly)}, AUC = {auc(-X_test_anomaly_rp, -X_test_normal_rp)}");
    # plt.show(block = True);

    print("exit.");


def testGaussianVAE_TagData():
    dataName = "主反原料带水";
    tagNames = ["FIC6421.PV", "FI6241.PV", "TIC6421.PV", "PI6410.PV", "FIC6419.PV", "TIC6201.PV", "TIC6414.PV"];
    # dataName = "副反原料带水";
    # tagNames = ["FIC6451.PV", "TIC6208.PV", "FIQ6243.PV", "FIC6201.PV", "PI6102B.PV", "TIC6101.PV"];
    # dataName = "一再尾燃";
    # tagNames = ["TI6113A.PV", "TI6116A.PV", "TIC6115.PV", "AI6103.PV"];
    # dataName = "增压机停机";
    # tagNames = ["TIC6102.PV", "LIC6101.PV", "FIC6121.PV", "FIC6104.PV", "TI6131A.PV", "PI6139.PV", "PI6133.PV", "TI6110.PV", "FT6702.PV", "TI6109.PV", "TI6192.PV", "TI6113A.PV", "PI6138.PV", "FIC6122.PV", "PT6707.PV", "LIC6102.PV", "TI6132.PV"];
    # dataName = "二再碳堆积";
    # tagNames = ["TIC6102.PV", "FIC6101D.PV", "FIC6418.PV", "FIQ6316.PV", "FI6241.PV", "LI6219.PV", "FIC6101C.PV", "FIC6328.PV", "FIC6211.PV", "LIC6270.PV", "TIC6201.PV", "FIQ6315.PV", "FIC6270.PV", "TI6105.PV", "TI6131A.PV", "FIQ6317.PV", "FIC6101B.PV", "FIC6309.PV", "FIC6101A.PV", "TIC6414.PV", "TI6414D.PV", "LIC6206.PV", "TIC6208.PV", "FI6313.PV", "TIC6101.PV", "FI6802.PV", "FIC6276.PV", "LI6212.PV", "FI6312.PV", "LIC6307.PV", "FIC6417.PV", "FIC6416.PV"];
    # dataName = "主原料中断";
    # tagNames = ["FI6180C.PV", "TI6417E.PV", "FI6180A.PV", "FIC6421.PV", "TIC6421.PV", "TI6410.PV", "TI6113A.PV", "FIC6419.PV", "TI6417F.PV", "TI6417C.PV", "TI6417B.PV", "FI6180D.PV", "FT6801.PV", "TI6255.PV", "TI6417A.PV", "TI6417D.PV", "PI6410.PV", "TI6426.PV", "TIC6414.PV", "TI6420.PV", "FI6180B.PV"];
    # dataName = "副原料中断";
    # tagNames = ["TI6103.PV", "FI6453D.PV", "FT6801.PV", "FI6453C.PV", "PI6102B.PV", "FIC6451.PV", "TI6113A.PV", "TI6454.PV", "TI6108.PV", "TIC6101.PV", "FI6453A.PV", "FI6453B.PV", "FIC6452.PV"];
    # dataName = "烟机异常振动";
    # tagNames = ["ZE6603B.PV", "TE6658.PV", "VE6606Y.PV", "ZE6603A.PV", "TE6648A.PV", "VE6606X.PV", "VE6605X.PV", "VE6605Y.PV", "TE6649A.PV", "PT6620.PV", "JI6601.PV", "PT6621.PV"];
    # dataName = "PK301停机";
    # tagNames = ["9PIC341.PV", "9PIC311.PV"];
    data, marks = loadRawTSData(tagNames);
    dataset = data[np.all(marks == 1, axis = -1)];
    print(dataName);

    filename = f"GaussianVAE_tagData_{dataName}.result";
    D = dataset.shape[-1];
    inputSize, hiddenSize1, hiddenSize2, latentSize, batchSize, maxEpoch = D, 4 * D, 2 * D, D // 2, 32, 20;
    scaler = StandardScaler();
    model = GaussianVAE(AggregateNetModule(
        AffineLayer(inputSize, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, hiddenSize2),
        ReluLayer(),
        AffineLayer(hiddenSize2, 2 * latentSize),
    ), AggregateNetModule(
        AffineLayer(latentSize, hiddenSize2),
        ReluLayer(),
        AffineLayer(hiddenSize2, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, 2 * inputSize),
    ), latentSize);

    Q = np.quantile(dataset, [0.25, 0.75], axis = 0);
    Q1, Q3 = Q[0], Q[1];
    IQR = Q3 - Q1;
    mask = np.all(np.logical_and(dataset >= Q1 - 5 * IQR, dataset <= Q3 + 5 * IQR), axis = -1);
    X = dataset[mask];
    np.random.shuffle(X);
    trainSize = int(len(X) * 0.8);
    X_train, X_test_normal, X_test_anomaly = X[: trainSize], X[trainSize:], dataset[~mask];

    scaler.fit(X_train);
    X_train, X_test_normal, X_test_anomaly = scaler.transform(X_train), scaler.transform(X_test_normal), scaler.transform(X_test_anomaly);

    lossFunc = GaussianVAELoss();
    optimizer = Adam();
    trainingIterator = SequentialDataIterator([X_train], batchSize);
    model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, plot = True);

    # with open(filename, "wb") as file:
    #     pickle.dump((model.params, scaler.params, X_train, X_test_normal, X_test_anomaly), file);
    # with open("adam.lr", "wb") as file:
    #     pickle.dump(optimizer.learningRate, file);

    # if os.path.isfile(filename):
    #     with open(filename, "br") as file:
    #         modelParams, scalerParams, X_train, X_test_normal, X_test_anomaly = pickle.load(file);
    #     if isinstance(modelParams[0], cp.ndarray):
    #         modelParams = [cp.asnumpy(p) for p in modelParams];
    #     scaler.params = scalerParams;
    #     model.params = modelParams;
    #
    # M, V = model.encode(X_train);
    # plt.figure(figsize = (12, 14));
    # plt.scatter(M[:, 0], M[:, 1], c = "b");
    # plt.scatter(V[:, 0], V[:, 1], c = "r");
    # plt.show(block = True);
    #
    # ratio = 1 / 365.0;
    # X_test_normal_rp = model.reconstructionProbability(X_test_normal);
    # X_test_anomaly_rp = model.reconstructionProbability(X_test_anomaly);
    # X_train_rp = np.concatenate(tuple([model.reconstructionProbability(item[0]) for item in SequentialDataIterator([X_train], batchSize, False)]), axis = 0);
    # X_all_rp = np.concatenate((X_test_anomaly_rp, X_test_normal_rp, X_train_rp), axis = 0);
    # X_all = np.concatenate((X_test_anomaly, X_test_normal, X_train), axis = 0);
    # threshold = np.quantile(X_all_rp, ratio);
    # X_anomaly = X_all[np.argwhere(X_all_rp <= threshold)[:, 0]];
    #
    # # filename = f"GaussianVAE_tagData_{dataName}.csv";
    # # X_anomaly = scaler.inverse(X_anomaly);
    # # content = [",".join(tagNames)];
    # # for item in X_anomaly:
    # #     content.append(",".join([str(v) for v in item]));
    # # with open(filename, "wt", encoding = "utf-8") as file:
    # #     file.write("\n".join(content));
    # # print(f"anomaly ratio is {ratio}, the anomaly data saved to file: {filename}");
    #
    # plt.figure(figsize = (12, 14));
    # bins = plt.hist(X_train_rp, bins = 1000, color = "k");
    # plt.hist(X_test_normal_rp, bins = 1000, label = "normal", color = "b");
    # # plt.hist(X_test_anomaly_rp, bins = 1000, label = "normal", color = "r");
    # plt.show(block = True);
    #
    # fprX, tprY = [0], [0];
    # for alpha in np.arange(0.001, 1, 0.001):
    #     threshold = np.quantile(X_train_rp, alpha);
    #     X_test_normal_class = X_test_normal_rp < threshold;
    #     X_test_anomaly_class = X_test_anomaly_rp < threshold;
    #
    #     TP, FP = np.sum(X_test_anomaly_class == True), np.sum(X_test_normal_class == True);
    #     FN, TN = np.sum(X_test_anomaly_class == False), np.sum(X_test_normal_class == False);
    #     precision, recall = TP / (TP + FP), TP / (TP + FN);
    #     fpr, tpr = FP / (FP + TN), recall;
    #     fprX.append(fpr);
    #     tprY.append(tpr);
    #     print(f"alpha = {alpha}, precision = {precision}, recall = {recall}, fpr = {fpr}, tpr = {tpr}");
    # fprX.append(1);
    # tprY.append(1);
    #
    # plt.figure(figsize = (12, 14));
    # plt.plot(fprX, tprY);
    # plt.xlabel("FPR");
    # plt.ylabel("TPR");
    # plt.title(f"P = {len(X_test_anomaly)}, AUC = {auc(-X_test_anomaly_rp, -X_test_normal_rp)}");
    # plt.show(block = True);

    print("exit.");


def testBernoulliVAE_TagData():
    # dataName = "主反原料带水";
    # tagNames = ["FIC6421.PV", "FI6241.PV", "TIC6421.PV", "PI6410.PV", "FIC6419.PV", "TIC6201.PV", "TIC6414.PV"];
    # dataName = "副反原料带水";
    # tagNames = ["FIC6451.PV", "TIC6208.PV", "FIQ6243.PV", "FIC6201.PV", "PI6102B.PV", "TIC6101.PV"];
    # dataName = "一再尾燃";
    # tagNames = ["TI6113A.PV", "TI6116A.PV", "TIC6115.PV", "AI6103.PV"];
    # dataName = "增压机停机";
    # tagNames = ["TIC6102.PV", "LIC6101.PV", "FIC6121.PV", "FIC6104.PV", "TI6131A.PV", "PI6139.PV", "PI6133.PV", "TI6110.PV", "FT6702.PV", "TI6109.PV", "TI6192.PV", "TI6113A.PV", "PI6138.PV", "FIC6122.PV", "PT6707.PV", "LIC6102.PV", "TI6132.PV"];
    # dataName = "二再碳堆积";
    # tagNames = ["TIC6102.PV", "FIC6101D.PV", "FIC6418.PV", "FIQ6316.PV", "FI6241.PV", "LI6219.PV", "FIC6101C.PV", "FIC6328.PV", "FIC6211.PV", "LIC6270.PV", "TIC6201.PV", "FIQ6315.PV", "FIC6270.PV", "TI6105.PV", "TI6131A.PV", "FIQ6317.PV", "FIC6101B.PV", "FIC6309.PV", "FIC6101A.PV", "TIC6414.PV", "TI6414D.PV", "LIC6206.PV", "TIC6208.PV", "FI6313.PV", "TIC6101.PV", "FI6802.PV", "FIC6276.PV", "LI6212.PV", "FI6312.PV", "LIC6307.PV", "FIC6417.PV", "FIC6416.PV"];
    # dataName = "主原料中断";
    # tagNames = ["FI6180C.PV", "TI6417E.PV", "FI6180A.PV", "FIC6421.PV", "TIC6421.PV", "TI6410.PV", "TI6113A.PV", "FIC6419.PV", "TI6417F.PV", "TI6417C.PV", "TI6417B.PV", "FI6180D.PV", "FT6801.PV", "TI6255.PV", "TI6417A.PV", "TI6417D.PV", "PI6410.PV", "TI6426.PV", "TIC6414.PV", "TI6420.PV", "FI6180B.PV"];
    # dataName = "副原料中断";
    # tagNames = ["TI6103.PV", "FI6453D.PV", "FT6801.PV", "FI6453C.PV", "PI6102B.PV", "FIC6451.PV", "TI6113A.PV", "TI6454.PV", "TI6108.PV", "TIC6101.PV", "FI6453A.PV", "FI6453B.PV", "FIC6452.PV"];
    # dataName = "烟机异常振动";
    # tagNames = ["ZE6603B.PV", "TE6658.PV", "VE6606Y.PV", "ZE6603A.PV", "TE6648A.PV", "VE6606X.PV", "VE6605X.PV", "VE6605Y.PV", "TE6649A.PV", "PT6620.PV", "JI6601.PV", "PT6621.PV"];
    dataName = "PK301停机";
    tagNames = ["9PIC341.PV", "9PIC311.PV"];
    data, marks = loadRawTSData(tagNames);
    dataset = data[np.all(marks == 1, axis = -1)];

    filename = f"BernoulliVAE_tagData_{dataName}.result";
    D = dataset.shape[-1];
    inputSize, hiddenSize1, hiddenSize2, latentSize, batchSize, maxEpoch = D, 4 * D, 2 * D, D // 2, 32, 20;
    model = BernoulliVAE(AggregateNetModule(
        AffineLayer(inputSize, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, hiddenSize2),
        ReluLayer(),
        AffineLayer(hiddenSize2, 2 * latentSize),
    ), AggregateNetModule(
        AffineLayer(latentSize, hiddenSize2),
        ReluLayer(),
        AffineLayer(hiddenSize2, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, inputSize),
    ), latentSize);

    # Q = np.quantile(dataset, [0.25, 0.75], axis = 0);
    # Q1, Q3 = Q[0], Q[1];
    # IQR = Q3 - Q1;
    # mask = np.all(np.logical_and(dataset >= Q1 - 5 * IQR, dataset <= Q3 + 5 * IQR), axis = -1);
    # X = dataset[mask];
    # np.random.shuffle(X);
    # trainSize = int(len(X) * 0.8);
    # X_train, X_test_normal, X_test_anomaly = X[: trainSize], X[trainSize:], dataset[~mask];
    #
    # scaler = MinMaxScaler();
    # scaler.fit(X_train);
    # X_train, X_test_normal, X_test_anomaly = scaler.transform(X_train), scaler.transform(X_test_normal), scaler.transform(X_test_anomaly);
    #
    # lossFunc = BernoulliVAELoss();
    # optimizer = Adam();
    # trainingIterator = SequentialDataIterator([X_train], batchSize);
    # model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, plot = True);
    #
    # with open(filename, "wb") as file:
    #     pickle.dump((model.params, X_train, X_test_normal, X_test_anomaly), file);

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params, X_train, X_test_normal, X_test_anomaly = pickle.load(file);
        if isinstance(params[0], cp.ndarray):
            params = [cp.asnumpy(p) for p in params];
        model.params = params;

    M, V = model.encode(X_train);
    plt.figure(figsize = (12, 14));
    plt.scatter(M[:, 0], M[:, 1], c = "b");
    plt.scatter(V[:, 0], V[:, 1], c = "r");
    plt.show(block = True);

    X_test_normal_rp = model.reconstructionProbability(X_test_normal);
    X_test_anomaly_rp = model.reconstructionProbability(X_test_anomaly);
    X_train_rp = np.concatenate(tuple([model.reconstructionProbability(item[0]) for item in SequentialDataIterator([X_train], batchSize, False)]), axis = 0);

    plt.figure(figsize = (12, 14));
    bins = plt.hist(X_train_rp, bins = 1000, color = "k");
    plt.hist(X_test_normal_rp, bins = 1000, label = "normal", color = "b");
    # plt.hist(X_test_anomaly_rp, bins = 1000, label = "normal", color = "r");
    plt.show(block = True);

    fprX, tprY = [0], [0];
    for alpha in np.arange(0.001, 1, 0.001):
        threshold = np.quantile(X_train_rp, alpha);
        X_test_normal_class = X_test_normal_rp < threshold;
        X_test_anomaly_class = X_test_anomaly_rp < threshold;

        TP, FP = np.sum(X_test_anomaly_class == True), np.sum(X_test_normal_class == True);
        FN, TN = np.sum(X_test_anomaly_class == False), np.sum(X_test_normal_class == False);
        precision, recall = TP / (TP + FP), TP / (TP + FN);
        fpr, tpr = FP / (FP + TN), recall;
        fprX.append(fpr);
        tprY.append(tpr);
        print(f"alpha = {alpha}, precision = {precision}, recall = {recall}, fpr = {fpr}, tpr = {tpr}");
    fprX.append(1);
    tprY.append(1);

    plt.figure(figsize = (12, 14));
    plt.plot(fprX, tprY);
    plt.xlabel("FPR");
    plt.ylabel("TPR");
    plt.title(f"AUC = {auc(-X_test_anomaly_rp, -X_test_normal_rp)}");
    plt.show(block = True);

    print("exit.");


def loadSequence(path : str, index = 7) -> (Vocab, np.ndarray, np.ndarray):
    with open(path, "rt", encoding = "utf-8") as file:
        tokens = [list(line[:-1]) for line in file.readlines()];

    vocab = Vocab(tokens);
    X = np.array([vocab[line[:index]] for line in tokens]);
    Y = np.array([vocab[line[index:]] for line in tokens]);

    return vocab, X, Y;


def testAddition():
    filename = "seq2seq_addition.weights";
    vocab, X, Y = loadSequence("/media/WindowsD/WorkSpace/深度学习进阶：自然语言处理/dataset/addition.txt");

    trainSize = int(len(X) * 0.9);
    X_train, Y_train = X[:trainSize], Y[:trainSize];
    X_test, Y_test = X[trainSize:], Y[trainSize:];

    X_train, X_test = X_train[:, ::-1], X_test[:, ::-1];

    batchSize, vocabSize, vectorSize, hiddenSize, maxEpoch = 256, len(vocab), 16, 128, 25;
    model = Seq2SeqModel(SeqEncoderLayer(vocabSize, vectorSize, hiddenSize),
                         SeqPeekyDecoderLayer(vocabSize, vectorSize, hiddenSize));

    # if os.path.isfile(filename):
    #     with open(filename, "rb") as file:
    #         params = pickle.load(file);
    #     model.params = params;
    #
    # startIndex = vocab["_"];
    # questions = ["6+9    ", "3+22   ", "7+681  ", "135+87", "794+809"];
    # for i in range(len(questions)):
    #     X = np.array(vocab[list(questions[i])][::-1]).reshape(1, -1);
    #     Y = model.generate(X, startIndex, 4);
    #     answer = "".join(vocab.getTokens(Y));
    #     print(f"{questions[i]} = {answer}");

    lossFunc = SoftmaxWithCrossEntropy1DLoss();
    optimizer = GradientsClipping(5.0, Adam());
    trainingIterator = SequentialDataIterator([X_train, Y_train], batchSize);
    testIterator = SequentialDataIterator([X_test, Y_test], batchSize);
    evaluator = IdentityAccuracyEvaluator();
    model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = True);

    with open(filename, "wb") as file:
        pickle.dump(model.params, file);

    print("exit.");


_idx = 0
def visualize(attention_map, row_labels, column_labels):
    fig, ax = plt.subplots()
    ax.pcolor(attention_map, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)

    ax.patch.set_facecolor('black')
    ax.set_yticks(np.arange(attention_map.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(attention_map.shape[1])+0.5, minor=False)
    ax.invert_yaxis()
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)

    global _idx
    _idx += 1
    plt.show(block = True);


def testSeq2Seq():
    filename = "seq2seq_date.weights";
    vocab, X, Y = loadSequence("/media/WindowsD/WorkSpace/深度学习进阶：自然语言处理/dataset/date.txt", index = 29);

    trainSize = int(len(X) * 0.9);
    X_train, Y_train = X[:trainSize], Y[:trainSize];
    X_test, Y_test = X[trainSize:], Y[trainSize:];

    # X_train, X_test = X_train[:, ::-1], X_test[:, ::-1];

    batchSize, vocabSize, vectorSize, hiddenSize, maxEpoch = 128, len(vocab), 16, 256, 10;
    model = Seq2SeqModel(SeqAttentionEncoderLayer(vocabSize, vectorSize, hiddenSize),
                         SeqBahdanauAttentionDecoderLayer(vocabSize, vectorSize, 2 * hiddenSize));

    if os.path.isfile(filename):
        with open(filename, "rb") as file:
            params = pickle.load(file);
        if isinstance(params[0], cp.ndarray):
            params = [cp.asnumpy(p) for p in params];
        model.params = params;

    # np.random.seed(1984);
    # for _ in range(5):
    #     idx = [np.random.randint(0, len(X_test))]
    #     x = X_test[idx]
    #     t = Y_test[idx]
    #
    #     model.forward(x, t)
    #     d = model.decoder.attentionWeight
    #     attention_map = d.reshape(d.shape[1], d.shape[2])
    #
    #     # reverse for print
    #     # attention_map = attention_map[:, ::-1]
    #     # x = x[:, ::-1]
    #
    #     row_labels = vocab.getTokens(x[0].tolist())
    #     column_labels = vocab.getTokens(t[0].tolist())
    #     column_labels = column_labels[1:]
    #
    #     visualize(attention_map, row_labels, column_labels);


    # startIndex = vocab["_"];
    # # questions = ["6+9    ", "3+22   ", "7+681  ", "135+87", "794+809"];
    # questions = ["THURSDAY, NOVEMBER 20, 1980  ", "thursday, july 9, 1970       ", "MARCH 16, 1985               ", "Oct 7, 2004                  ", "12/6/83                      "];
    # for i in range(len(questions)):
    #     # X = np.array(vocab[list(questions[i])][::-1]).reshape(1, -1);
    #     X = np.array(vocab[list(questions[i])]).reshape(1, -1);
    #     Y = model.generate(X, startIndex, Y_test.shape[-1] - 1);
    #     answer = "".join(vocab.getTokens(Y));
    #     print(f"{questions[i]} = {answer}");

    lossFunc = SoftmaxWithCrossEntropy1DLoss();
    optimizer = GradientsClipping(5.0, Adam());
    trainingIterator = SequentialDataIterator([X_train, Y_train], batchSize);
    testIterator = SequentialDataIterator([X_test, Y_test], batchSize);
    evaluator = IdentityAccuracyEvaluator();
    model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, testIterator, evaluator, plot = True);

    with open(filename, "wb") as file:
        pickle.dump(model.params, file);

    print("exit.");


def unitTest():
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} start to unit test\n");

    # testSGD_Numba();
    # testSeq2col_Numba();

    # testPerformance();
    # testFunctionalNetModuleGradient1();
    # testFunctionalNetModuleGradient2();
    # testRelu_Numba();
    # testReluLayer_Numba();
    # testSigmoid1();
    # testSigmoid2();
    # testSigmoidGradient1();
    # testSigmoidGradient2();
    # testMinMaxScaler();

    # testDropout();

    # testAggregateNetLrScheduler();
    # testCyclicNetLrScheduler();

    # testGetLossMaskByValidLength1();
    # testSoftmax1();
    # testSoftmaxLayerGradient1();
    # testSoftmaxLayerGradient2();
    # testSoftmaxLayerGradient3();
    # testLogSoftmax1();
    # testLogSoftmaxGradient1();
    # testLogSoftmaxGradient2();
    # testGatedAdditiveLayerGradient1();
    # testGatedAdditiveLayerGradient2();
    # testIdentityWithMeanAbsoluteLossGradient1();
    # testIdentityWithMeanAbsoluteLossGradient2();
    # testIdentityWithMeanAbsoluteLossGradient3();
    # testIdentityWithMeanAbsoluteLossGradient4();
    # testIdentityWithMeanAbsolutePercentLossGradient1();
    # testIdentityWithMeanAbsolutePercentLossGradient2();
    # testIdentityWithMeanAbsolutePercentLossGradient3();
    # testIdentityWithMeanAbsolutePercentLossGradient4();
    # testCrossEntropyLossGradient1();
    # testCrossEntropyLossGradient2();
    # testCrossEntropyLossGradient3();
    # testCrossEntropyLossGradient4();
    # testCrossEntropyLossGradient5();
    # testCrossEntropyLossGradient6();
    # testSoftmaxWithCrossEntropyLossGradient1();
    # testSoftmaxWithCrossEntropyLossGradient2();
    # testSoftmaxWithCrossEntropyLossGradient3();
    # testSoftmaxWithCrossEntropyLossGradient4();
    # testSoftmaxWithCrossEntropyLossGradient5();
    # testSoftmaxWithCrossEntropyLossGradient6();
    # testSigmoidWithCrossEntropyLossGradient1();
    # testSigmoidWithCrossEntropyLossGradient2();
    # testSigmoidWithCrossEntropyLossGradient3();
    # testSigmoidWithCrossEntropyLossGradient4();
    # testSigmoidWithCrossEntropyLossGradient5();
    # testSigmoidWithCrossEntropyLossGradient6();
    # testSoftmaxWithCrossEntropy1DLossGradient1();
    # testSoftmaxWithCrossEntropy1DLossGradient2();
    # testSoftmaxWithCrossEntropy1DLossGradient3();
    # testSoftmaxWithCrossEntropy1DLossGradient4();
    # testSoftmaxWithCrossEntropy1DLossGradient5();
    # testSoftmaxWithCrossEntropy1DLossGradient6();
    # testSequenceSoftmaxWithCrossEntropy1DLossGradient1();
    # testIdentityWithMeanSquareLossGradient1();
    # testIdentityWithMeanSquareLossGradient2();
    # testIdentityWithMeanSquareLossGradient3();
    # testIdentityWithMeanSquareLossGradient4();
    # testSumWithMeanSquareLossGradient1();
    # testSumWithMeanSquareLossGradient2();
    # testPReluLayerGradient1();
    # testPReluLayerGradient2();
    # testPReluLayerGradient3();
    # testSoftplus1();
    # testSoftplusLayerGradient1();
    # testSoftplusLayerGradient2();
    # testSoftplusLayerGradient3();
    # testSwishLayerGradient1();
    # testSwishLayerGradient2();
    # testSwishLayerGradient3();
    # testSiluLayerGradient1();
    # testSiluLayerGradient2();
    # testGeluLayerGradient1();
    # testGeluLayerGradient2();
    # testMaxoutLayer1();
    # testMaxoutLayer2();
    # testMaxoutLayerGradient1();
    # testMaxoutLayerGradient2();
    # testMaxoutLayerGradient3();
    # testIdentityWithHuberLossGradient1();
    # testIdentityWithHuberLossGradient2();
    # testIdentityWithHuberLossGradient3();
    # testIdentityWithHuberLossGradient4();
    # testAffineLayerGradient1();
    # testAffineLayerGradient2();
    # testAffineLayerGradient3();
    # testConvolution1DLayer1();
    # testConvolution1DLayer2();
    # testConvolution1DLayer3();
    # testConvolution1DLayerGradient1();
    # testConvolution1DLayerGradient2();
    # testConvolution1DLayerGradient3();
    # testConvolution1DLayerGradient4();
    # testMaxPooling1DLayer1();
    # testMaxPooling1DLayerGradient1();
    # testMaxPooling1DLayerGradient2();
    # testMaxPooling1DLayerGradient3();
    # testMaxPooling1DLayerGradient4();
    # testAvgPooling1DLayer1();
    # testAvgPooling1DLayerGradient1();
    # testAvgPooling1DLayerGradient2();
    # testAvgPooling1DLayerGradient3();
    # testAvgPooling1DLayerGradient4();
    # testConvolution2DLayer1();
    # testConvolution2DLayer2();
    # testConvolution2DLayer3();
    # testConvolution2DLayerGradient1();
    # testConvolution2DLayerGradient2();
    # testConvolution2DLayerGradient3();
    # testConvolution2DLayerGradient4();
    # testMaxPooling2DLayer1();
    # testMaxPooling2DLayerGradient1();
    # testMaxPooling2DLayerGradient2();
    # testMaxPooling2DLayerGradient3();
    # testMaxPooling2DLayerGradient4();
    # testAvgPooling2DLayer1();
    # testAvgPooling2DLayerGradient1();
    # testAvgPooling2DLayerGradient2();
    # testAvgPooling2DLayerGradient3();
    # testAvgPooling2DLayerGradient4();
    # testBatchNormalization1DLayer1();
    # testBatchNormalization1DLayer2();
    # testBatchNormalization1DLayerGradient1();
    # testBatchNormalizationLayer1DGradient2();
    # testBatchNormalization2DLayer1();
    # testBatchNormalizationLayer2DGradient1();
    # testLayerNormalizationLayer1();
    # testLayerNormalizationLayerGradient1();
    # testLayerNormalizationLayerGradient2();
    # testLayerNormalizationLayerGradient3();
    # testMinMaxLayerGradient1();
    # testMinMaxLayerGradient2();
    # testMinMaxLayerGradient3();
    # testMinMaxLayerGradient4();
    # testMinMaxLayerGradient5();
    # testMinMaxLayerGradient6();
    # testEmbeddingLayerGradient1();
    # testEmbeddingLayerGradient2();
    # testEmbeddingWithDotLayerGradient1();
    # testEmbeddingWithDotLayerGradient2();
    # testAdditiveResidualBlockGradient1();
    # testAdditiveResidualBlockGradient2();
    # testAdditiveResidualBlockGradient3();
    # testRepeatedWrapperOfAffineLayerGradient();
    # testRnnCell1();
    # testRnnCellGradient1();
    # testRnnLayer1();
    # testRnnLayerGradient1_Sequence();
    # testRnnLayerGradient2_Sequence();
    # testRnnLayerGradient3_State();
    # testRnnLayerGradient4_Sequence_State();
    # testRnnLayerGradient5_Foreign_Sequence_State();
    # testGruCell1();
    # testGruCellGradient1();
    # testGruLayer1();
    # testGruLayerGradient1_Sequence();
    # testGruLayerGradient2_State();
    # testGruLayerGradient3_Sequence_State();
    # testGruLayerGradient4_Foreign_Sequence_State();
    # testLstmCell1();
    # testLstmCellGradient1();
    # testLstmCellGradient2();
    # testLstmCellGradient_Dropout();
    # testLstmLayer1();
    # testLstmLayerGradient1_Sequence();
    # testLstmLayerGradient2_State();
    # testLstmLayerGradient3_Sequence_State();
    # testLstmLayerGradient4_Foreign_Sequence_State();
    # testLstmLayerGradient_State_Dropout(False);
    # testLstmLayerGradient_Stepwise(False);
    # testLstmLayerGradient_Stepwise_State(False);
    # testLstmLayerGradient_Stepwise_State_Dropout(False);
    # testBahdanauAttentionLstmLayerGradient(True);
    # testBahdanauAttentionLstmLayerGradient_Stepwise(True);
    # testBahdanauAttentionLstmLayerGradient_Stepwise_State(True);
    # testBahdanauAttentionLstmLayerGradient_Stepwise_State_Dropout(True);
    # testBiRnnLayerGradient1_Gru_InnerState_Sequence();
    # testBiRnnLayerGradient2_Gru_InnerState_State();
    # testBiRnnLayerGradient3_Gru_InnerState_Sequence_State();
    # testBiRnnLayerGradient4_Gru_ForeignState_Sequence();
    # testBiRnnLayerGradient5_Gru_ForeignState_State();
    # testBiRnnLayerGradient6_Gru_ForeignState_Sequence_State();
    # testBiRnnLayerGradient7_LstmLayer_InnerState_Sequence();
    # testBiRnnLayerGradient8_LstmLayer_InnerState_State();
    # testBiRnnLayerGradient9_LstmLayer_InnerState_Sequence_State();
    # testBiRnnLayerGradient10_LstmLayer_ForeignState_Sequence();
    # testBiRnnLayerGradient11_LstmLayer_ForeignState_State();
    # testBiRnnLayerGradient12_LstmLayer_ForeignState_Sequence_State();
    # testStackRnnLayerGradient1_Gru_InnerState_Sequence();
    # testStackRnnLayerGradient2_Gru_InnerState_State();
    # testStackRnnLayerGradient3_Gru_InnerState_Sequence_State();
    # testStackRnnLayerGradient4_Gru_ForeignState_Sequence();
    # testStackRnnLayerGradient5_Gru_ForeignState_State();
    # testStackRnnLayerGradient6_Gru_ForeignState_Sequence_State();
    # testStackRnnLayerGradient7_Lstm_InnerState_Sequence();
    # testStackRnnLayerGradient8_Lstm_InnerState_State();
    # testStackRnnLayerGradient9_Lstm_InnerState_Sequence_State();
    # testStackRnnLayerGradient10_Lstm_ForeignState_Sequence();
    # testStackRnnLayerGradient11_Lstm_ForeignState_State();
    # testStackRnnLayerGradient12_Lstm_ForeignState_Sequence_State();
    # testStackRnnLayerGradient13_BiGru_ForeignState_Sequence_State();
    # testStackRnnLayerGradient14_BiLstm_ForeignState_Sequence_State();
    # testStackRnnLayerGradient15_Lstm_ForeignState_Sequence_State_Normal();
    # testStackRnnLayerGradient16_BiLstm_ForeignState_Sequence_State_Normal();
    # testStackLstmLayerGradient_State(False);
    # testStackLstmLayerGradient_State_Dropout(False);

    # testAdditiveAttentionModule1();
    # testAdditiveAttentionModuleGradient1();
    # testAdditiveAttentionModuleGradient2();
    # testAdditiveAttentionModuleGradient3();
    # testAdditiveAttentionModuleGradient4();
    # testDotProductAttentionModuleGradient1();
    # testDotProductAttentionModuleGradient2();
    # testDotProductAttentionModuleGradient3();
    # testDotProductAttentionModuleGradient4();
    # testMultiHeadAttentionModule1();
    # testMultiHeadAttentionModule2();
    # testMultiHeadAttentionModule3();
    # testMultiHeadAttentionModule4();
    # testMultiHeadAttentionModuleGradient1();
    # testMultiHeadAttentionModuleGradient2();
    # testMultiHeadAttentionModuleGradient3();
    # testMultiHeadAttentionModuleGradient4();
    # testMultiHeadAttentionModuleGradient5();
    # testSelfAttentionModuleGradient1();
    # testSelfAttentionModuleGradient2();
    # testSelfAttentionModuleGradient3();
    # testSelfAttentionModuleGradient4();
    # testSelfAttentionModuleGradient5();
    # testSinePositionalEncodingModuleGradient1();
    # testSinePositionalEncodingModuleGradient2();
    # testSinePositionalEncodingModuleGradient3();
    # testSinePositionalEncodingModuleGradient4();
    # testTransformerAddNormalizationModuleGradient1();
    # testTransformerAddNormalizationModuleGradient2();
    # testTransformerPositionwiseFFNModuleGradient1();
    # testTransformerEncoderBlockGradient1();
    # testTransformerEncoderBlockGradient2();
    # testTransformerEncoderBlockGradient3();
    # testTransformerEncoderBlockGradient4();
    # testTransformerEncoderGradient1();
    # testTransformerEncoderGradient2();
    # testTransformerEncoderGradient3();
    # testTransformerEncoderGradient4();
    # testTransformerDecoderBlockGradient1();
    # testTransformerDecoderBlockGradient2();
    # testTransformerDecoderBlockGradient3();
    # testTransformerDecoderBlockGradient4();
    # testTransformerDecoder1();
    # testTransformerDecoder2();
    # testTransformerDecoderGradient1();
    # testTransformerDecoderGradient2();
    # testTransformerDecoderGradient3();
    # testTransformerEmbeddingEncoderGradient1();
    # testTransformerEmbeddingEncoderGradient2();
    # testTransformerEmbeddingEncoderGradient3();
    # testTransformerEmbeddingDecoder1();
    # testTransformerEmbeddingDecoder2();
    # testTransformerEmbeddingDecoderGradient1();
    # testTransformerEmbeddingDecoderGradient2();
    # testTransformerEmbeddingDecoderGradient3();
    # testAttentionPoolingLayerGradient1();
    # testAttentionPoolingLayerGradient2();
    # testAttentionPoolingLayerGradient3();
    # testAttentionPoolingLayerGradient4();

    testRnnWithCupy();
    testRnnWithTorch();

    # testSelectByWeightModuleGradient();
    # testAdditiveAttentionWeight1TModuleGradient();
    # testAdditiveAttentionWeightNTModuleGradient();
    # testDotProductAttentionWeightModuleGradient();
    # testQKVAttentionLayerGradient();

    # testGaussianVAELossGradient();
    # testGaussianVAEGradient1();
    # testGaussianVAEGradient2();
    # testGaussianVAE_Anomaly_HTRU2();
    # testBernoulliVAELossGradient();
    # testBernoulliVAEGradient1();
    # testBernoulliVAEGradient2();
    # testBernoulliVAE_MNIST();
    # testBernoulliVAE_Anomaly_MNIST();

    # testSeqAttentionDecoderGradient();
    # testSeqBahdanauAttentionDecoderGradient();
    # testSeqTSEncoderLayerGradient();
    # testSeqTSDecoderLayerGradient();
    # testSeq2SeqTSModel1();
    # testSeq2SeqTSModel2();
    # testSeq2SeqTSModel_Dropout();

    print(f"\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} end to unit test\n");


def sumAll(*X : np.ndarray) -> float:
    return sum([float(np.sum(x)) for x in X]);


def getErrorText(title : str, x1 : np.ndarray, x2 : np.ndarray) -> str:
    return f", {title}: {np.sum(np.fabs(x1 - x2))}({np.linalg.norm(x1 - x2) / (np.linalg.norm(x1) + np.linalg.norm(x2))})";


def testSGD_Numba():
    N = 10000;
    sdg = SGD();
    context = NetContext();
    params = [NetParamDefinition(f"weight{i}", np.random.randn(100, 100), grad = np.random.randn(100, 100)) for i in range(100)];

    sdg.updateStep(params, context);

    st = time.time();
    for _ in range(N):
        sdg.updateStep(params, context);
    et = time.time();

    print(f"time: {et - st} s.");


def testSeq2col_Numba():
    N = 1000;
    FW, stride, padding = 5, 1, 0;
    X = np.random.randn(1000, 30, 10);

    Y, OT = seq2col(X, FW, stride, padding);

    st = time.time();
    for _ in range(N):
        Y, OT = seq2col(X, FW, stride, padding);
    et = time.time();

    print(f"time: {et - st} s.");


def testPerformance():
    value = 999.99;
    X = np.random.randn(32, 1024, 512);
    Y, Z = X.copy(), X.copy();
    M = np.random.randint(0, 2, X.shape);

    times = 100;

    now = time.time();
    for _ in range(times):
        putArrayMask(X, M, value);
    t1 = time.time() - now;
    print(f"t1: {t1}");

    now = time.time();
    for _ in range(times):
        np.place(Y, M, value);
    t2 = time.time() - now;
    print(f"t2: {t2}, {np.all(X == Y)}");

    now = time.time();
    for _ in range(times):
        np.putmask(Z, M, value);
    t3 = time.time() - now;
    print(f"t3: {t3}, {np.all(X == Z)}");

    print([t1, t2, t3]);

    print("exit.");


def testFunctionalNetModuleGradient1():
    N, T, D = 32, 24, 16;
    X = np.fabs(np.random.randn(N, T, D)) + 1.0;
    m = FunctionalNetModule("Log", lambda x: np.log(x), lambda x, y, dy: dy * (1 / x));
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"FunctionalNetModule, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testFunctionalNetModuleGradient2():
    N, T, D = 32, 24, 16;
    X1 = np.fabs(np.random.randn(N, T, D)) + 1.0;
    X2 = np.fabs(np.random.randn(N, T, D)) + 2.0;
    m = FunctionalNetModule("Log", lambda x: np.log(x), lambda x, y, dy: dy * (1 / x));
    Y1, Y2 = m.forward(X1, X2);
    dX1, dX2 = m.backward(np.ones_like(Y1), np.ones_like(Y1));
    dX1N = numericGradient(lambda x: np.sum(np.add(*m.forward(x, X2))), X1);
    dX2N = numericGradient(lambda x: np.sum(np.add(*m.forward(X1, x))), X2);
    print(f"FunctionalNetModule, numericGradient2, dX1 error: {np.sum(np.abs(dX1 - dX1N))}, dX2 error: {np.sum(np.abs(dX2 - dX2N))}");
    print("\n");


def testRelu_Numba():
    N = 1000;
    X = np.random.randn(N, N);

    relu(X);

    st = time.time();
    for _ in range(N):
        relu(X);
    et = time.time();

    print(f"time: {et - st} s.");


# conclusion: not applicable
def testReluLayer_Numba():
    N = 1000;
    X = np.random.randn(N, N);
    dY = np.ones_like(X);

    m = ReluLayer();
    Y, = m.forward(X);
    dX, = m.backward(dY);

    st = time.time();
    for _ in range(N):
        Y, = m.forward(X);
        dX, = m.backward(dY);
    et = time.time();

    print(f"time: {et - st} s.");


def testSigmoid1():
    X = np.array([-25.0, -21, 21, 25]);
    Y = sigmoid(X);

    X1, X2 = X[:2], X[2:];
    Y1 = np.exp(X1);
    Y1 = Y1 / (1 + Y1);
    Y2 = 1 / (1 + np.exp(-X2));

    print(f"Sigmoid, value1, error: {np.sum(np.abs(Y - np.concatenate((Y1, Y2))))}");
    print("\n");


def testSigmoid2():
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D) * 10;
    Y1 = sigmoid(X);
    Y2 = 1 / (1 + np.exp(-X));

    print(f"Sigmoid, value2, error: {np.sum(np.abs(Y1 - Y2))}");
    print("\n");


def testSigmoidGradient1():
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D) * 20;
    m = SigmoidLayer();
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"SigmoidLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSigmoidGradient2():
    N, D = 256, 256;
    X = np.random.randn(N, D) * 20;
    m = SigmoidLayer();
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"SigmoidLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testLabelSmoothing():
    X = np.array([-25.0, -21, 21, 25]);
    Y = sigmoid(X);

    X1, X2 = X[:2], X[2:];
    Y1 = np.exp(X1);
    Y1 = Y1 / (1 + Y1);
    Y2 = 1 / (1 + np.exp(-X2));

    print(Y - np.concatenate((Y1, Y2)));


def testModuleGradient(m : INetModule, title: str, *data : np.ndarray):
    numGradients = [];

    for p in m.params:
        v = p.value;
        numGradients.append(numericGradient(lambda x : sumAll(*m.copy(True).forward(*data)), v));

    message = '\n'.join([f'param {m.params[i].name}{i}{m.params[i].value.shape} error value: {np.sum(np.fabs(m.params[i].grad - numGradients[i]))}, error ratio: {np.linalg.norm(m.params[i].grad - numGradients[i]) / (np.linalg.norm(m.params[i].grad) + np.linalg.norm(numGradients[i]))}' for i in range(len(m.params))]);
    print(f"{title}\n{message}");


def testMinMaxScaler():
    N, T, D = 32000, 40, 26;
    X = np.random.randn(N, T, D);

    trainSize, stepSize = int(N * 0.8), 30;
    X_train_1, Y_train_1 = X[:trainSize, :stepSize], X[:trainSize, stepSize:];
    X_test_1, Y_test_1 = X[trainSize:, :stepSize], X[trainSize:, stepSize:];

    scaler = MinMaxScaler();
    scaler.fit(X);
    X_train_2 , Y_train_2 = scaler.inverse(scaler.transform(X_train_1)), scaler.inverse(scaler.transform(Y_train_1));
    X_test_2, Y_test_2 = scaler.inverse(scaler.transform(X_test_1)), scaler.inverse(scaler.transform(Y_test_1));

    print(f"MinMaxScaler, transform-inverse, {np.sum(np.abs(X_train_1 - X_train_2))}, {np.sum(np.abs(Y_train_1 - Y_train_2))}, {np.sum(np.abs(X_test_1 - X_test_2))}, {np.sum(np.abs(Y_test_1 - Y_test_2))}");
    print("\n");


def testDropout():
    batchSize, channelNum, imageHeight, imageWidth, inputSize, timeStep = 32, 16, 24, 24, 48, 10;

    X1 = np.random.randn(batchSize, inputSize);
    m1 = DropoutLayer();
    m1.context.isTrainingMode = True;
    Y1, = m1.forward(X1);
    print(f"DropoutLayer, mask shape: {m1.mask.shape}");

    X2 = np.random.randn(timeStep, batchSize, inputSize);
    m2 = VariationalDropoutLayer();
    m2.context.isTrainingMode = True;
    Y2, = m2.forward(X2);
    print(f"DropoutVariationalDropoutLayerLayer, mask shape: {m2.mask.shape}");

    X3 = np.random.randn(batchSize, channelNum, inputSize);
    m3 = Dropout1DLayer();
    m3.context.isTrainingMode = True;
    Y3, = m3.forward(X3);
    print(f"Dropout1DLayer, mask shape: {m3.mask.shape}");

    X4 = np.random.randn(batchSize, channelNum, imageHeight, imageWidth);
    m4 = Dropout2DLayer();
    m4.context.isTrainingMode = True;
    Y4, = m4.forward(X4);
    print(f"Dropout2DLayer, mask shape: {m4.mask.shape}");

    print("\n");


def testAggregateNetLrScheduler():
    lrs = [];
    # scheduler = LinearNetLrScheduler(1.0, minEpoch = 2, maxEpoch = 18);
    # scheduler = MultiStepNetLrScheduler(1.0, [5, 10, 15], minEpoch = 2, maxEpoch = 18);
    # scheduler = CosineNetLrScheduler(1.0, minLr = 0.2, minEpoch = 5, maxEpoch = 15);
    scheduler = AggregateNetLrScheduler([
        LinearNetLrScheduler(1.0, minEpoch = 0, maxEpoch = 5),
        MultiStepNetLrScheduler(1.0, [8, 12], gamma = math.sqrt(0.5), minEpoch = 5, maxEpoch = 15),
        CosineNetLrScheduler(0.5, minLr = 0.05, minEpoch = 15, maxEpoch = 25),
        ConstantNetLrScheduler(0.05, minEpoch = 25),
    ]);

    for i in range(0, 30):
        scheduler.epochStep(i);
        lrs.append(scheduler.learningRate);

    plt.figure();
    plt.plot(lrs);
    plt.show(block = True);


def testCyclicNetLrScheduler():
    lrs = [];
    scheduler = CyclicNetLrScheduler(CosineNetLrScheduler(1.0, minLr = 0.0, minEpoch = 0, maxEpoch = 9), cycleSize = 10, minEpoch = 5, maxEpoch = 34);

    for i in range(0, 40):
        scheduler.epochStep(i);
        lrs.append(scheduler.learningRate);

    plt.figure();
    plt.plot(lrs);
    plt.show(block = True);


def testGetLossMaskByValidLength1():
    maxLength = 5;

    validLen1 = np.array(2);
    M11 = getLossMaskByValidLength(maxLength, validLen1);
    M12 = np.array([True] * 2 + [False] * 3);
    assert np.all(M11 == M12);

    validLen2 = np.array([2, 3]);
    M21 = getLossMaskByValidLength(maxLength, validLen2);
    M22 = np.array([
        [True] * 2 + [False] * 3,
        [True] * 3 + [False] * 2,
    ]);
    assert np.all(M21 == M22);

    validLen3 = np.array([[1, 2], [3, 4]]);
    M31 = getLossMaskByValidLength(maxLength, validLen3);
    M32 = np.array([
        [
            [True] * 1 + [False] * 4,
            [True] * 2 + [False] * 3,
        ],
        [
            [True] * 3 + [False] * 2,
            [True] * 4 + [False] * 1,
        ]
    ]);
    assert np.all(M31 == M32);


def testSoftmax1():
    N, C = 32, 256;
    X = np.random.randn(N, C);
    M = np.random.randint(0, 2, (N, C));

    X1 = X;
    Y1 = softmax(X1, M);

    X2 = X * M;
    X2 += ~M.astype(bool) * -1e8;
    Y2 = np.exp(X2) / np.sum(np.exp(X2), axis = -1, keepdims = True);

    print(f"Softmax, value1, Y error: {np.sum(np.abs(Y1 - Y2))}");
    print("\n");


def testSoftmaxLayerGradient1():
    N, C = 64, 24;
    X = np.random.randn(N, C);
    dY = np.random.randn(N, C);
    m = SoftmaxLayer();
    Y, = m.forward(X);
    dX1, = m.backward(dY);
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0] * dY), X);
    print(f"SoftmaxLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxLayerGradient2():
    N, C = 64, 24;
    X = np.random.randn(N, C);
    M = np.random.randint(0, 2, (N, C));
    dY = np.random.randn(N, C);
    m = SoftmaxLayer();
    Y, = m.forward(X, M);
    dX1, = m.backward(dY);
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0] * dY), X);
    print(f"SoftmaxLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxLayerGradient3():
    N, Q, C = 32, 64, 24;
    X = np.random.randn(N, Q, C);
    M = np.random.randint(0, 2, (N, Q, C));
    dY = np.random.randn(N, Q, C);
    m = SoftmaxLayer();
    Y, = m.forward(X, M);
    dX1, = m.backward(dY);
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0] * dY), X);
    print(f"SoftmaxLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testLogSoftmax1():
    N, C = 32, 256;
    X = np.random.randn(N, C);

    Y1 = logSoftmax(X);

    Y2 = np.log(softmax(X));

    print(f"LogSoftmax, value1, Y error: {np.sum(np.abs(Y1 - Y2))}");
    print("\n");


def testLogSoftmaxGradient1():
    N, C = 64, 24;
    X = np.random.randn(N, C);
    dY = np.random.randn(N, C);
    Y, Z = logSoftmax(X, returnSoftmax = True);
    dX1 = logSoftmaxGradient(Z, dY);
    dXN = numericGradient(lambda x: np.sum(logSoftmax(x) * dY), X);
    print(f"LogSoftmax, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testLogSoftmaxGradient2():
    N, T, C = 32, 24, 12;
    X = np.random.randn(N, T, C);
    dY = np.random.randn(N, T, C);
    Y, Z = logSoftmax(X, returnSoftmax = True);
    dX1 = logSoftmaxGradient(Z, dY);
    dXN = numericGradient(lambda x: np.sum(logSoftmax(x) * dY), X);
    print(f"LogSoftmax, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testGatedAdditiveLayerGradient1():
    N, D = 32, 48;
    X1, X2 = np.random.randn(N, D), np.random.randn(N, D);
    weight, bias = np.random.randn(2 * D, D), np.random.randn(D);
    m = GatedAdditiveLayer(D, W = weight, b = bias);
    Y, = m.forward(X1, X2);
    dX1, dX2 = m.backward(np.ones_like(Y));
    dW1, db1 = m.params[0].grad, m.params[1].grad;
    dX1N = numericGradient(lambda x: np.sum(m.forward(x, X2)[0]), X1);
    dX2N = numericGradient(lambda x: np.sum(m.forward(X1, x)[0]), X2);
    dWN = numericGradient(lambda x: np.sum(GatedAdditiveLayer(D, W = x, b = bias).forward(X1, X2)[0]), weight);
    dbN = numericGradient(lambda x: np.sum(GatedAdditiveLayer(D, W = weight, b = x).forward(X1, X2)[0]), bias);
    print(f"GatedAdditiveLayer, numericGradient1 {getErrorText('dX1 error', dX1, dX1N)} {getErrorText('dX2 error', dX2, dX2N)} {getErrorText('dW error', dW1, dWN)} {getErrorText('db error', db1, dbN)}");
    print("\n");


def testGatedAdditiveLayerGradient2():
    N, M, T, D = 32, 2, 6, 24;
    X1, X2 = np.random.randn(N, M, T, D), np.random.randn(N, M, T, D);
    weight, bias = np.random.randn(2 * D, D), np.random.randn(D);
    m = GatedAdditiveLayer(D, W = weight, b = bias);
    Y, = m.forward(X1, X2);
    dX1, dX2 = m.backward(np.ones_like(Y));
    dW1, db1 = m.params[0].grad, m.params[1].grad;
    dX1N = numericGradient(lambda x: np.sum(m.forward(x, X2)[0]), X1);
    dX2N = numericGradient(lambda x: np.sum(m.forward(X1, x)[0]), X2);
    dWN = numericGradient(lambda x: np.sum(GatedAdditiveLayer(D, W = x, b = bias).forward(X1, X2)[0]), weight);
    dbN = numericGradient(lambda x: np.sum(GatedAdditiveLayer(D, W = weight, b = x).forward(X1, X2)[0]), bias);
    print(f"GatedAdditiveLayer, numericGradient1 {getErrorText('dX1 error', dX1, dX1N)} {getErrorText('dX2 error', dX2, dX2N)} {getErrorText('dW error', dW1, dWN)} {getErrorText('db error', db1, dbN)}");
    print("\n");


def testIdentityWithMeanAbsoluteLossGradient1():
    N = 32;
    X, T = np.random.randn(N), np.random.randn(N);
    m = IdentityWithMeanAbsoluteLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithMeanAbsoluteLoss, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanAbsoluteLossGradient2():
    N, D = 32, 24;
    X, T = np.random.randn(N, D), np.random.randn(N, D);
    m = IdentityWithMeanAbsoluteLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithMeanAbsoluteLoss, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanAbsoluteLossGradient3():
    N, C, D = 32, 24, 10;
    X, T = np.random.randn(N, C, D), np.random.randn(N, C, D);
    m = IdentityWithMeanAbsoluteLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithMeanAbsoluteLoss, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanAbsoluteLossGradient4():
    N, C, D = 32, 24, 10;
    X, W, T = np.random.randn(N, C, D), np.fabs(np.random.randn(N, C, D)), np.random.randn(N, C, D);
    m = IdentityWithMeanAbsoluteLoss();
    loss = m.forward(X, W, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, W, T), X);
    print(f"IdentityWithMeanAbsoluteLoss, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanAbsolutePercentLossGradient1():
    N = 32;
    X, T = np.random.randn(N), np.fabs(np.random.randn(N));
    m = IdentityWithMeanAbsolutePercentLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithMeanAbsolutePercentLoss, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanAbsolutePercentLossGradient2():
    N, D = 32, 24;
    X, T = np.random.randn(N, D), np.fabs(np.random.randn(N, D));
    m = IdentityWithMeanAbsolutePercentLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithMeanAbsolutePercentLoss, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanAbsolutePercentLossGradient3():
    N, C, D = 32, 24, 10;
    X, T = np.random.randn(N, C, D), np.fabs(np.random.randn(N, C, D));
    m = IdentityWithMeanAbsolutePercentLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithMeanAbsolutePercentLoss, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanAbsolutePercentLossGradient4():
    N, C, D = 32, 24, 10;
    X, W, T = np.random.randn(N, C, D), np.fabs(np.random.randn(N, C, D)), np.fabs(np.random.randn(N, C, D));
    m = IdentityWithMeanAbsolutePercentLoss();
    loss = m.forward(X, W, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, W, T), X);
    print(f"IdentityWithMeanAbsolutePercentLoss, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testCrossEntropyLossGradient1():
    N, C = 32, 24;
    X, T = np.random.randn(N, C), np.zeros((N, C));
    Y = softmax(X);
    T[np.arange(N), np.argmax(X, axis = -1)] = 1;
    m = CrossEntropyLoss();
    loss = m.forward(Y, T);
    dY1 = m.backward()[0];
    dYN = numericGradient(lambda x: m.forward(x, T), Y);
    print(f"CrossEntropyLoss, numericGradient1 {getErrorText('dY error', dY1, dYN)}");
    print("\n");


def testCrossEntropyLossGradient2():
    N, D, C = 32, 10, 24;
    X, T = np.random.randn(N, D, C), np.zeros((N, D, C));
    index = np.argmax(X.reshape(-1, C), axis = -1);
    Y = softmax(X);
    T.reshape(-1, C)[np.arange(N * D), index] = 1;
    m = CrossEntropyLoss();
    loss = m.forward(Y, T);
    dY1 = m.backward()[0];
    dYN = numericGradient(lambda x: m.forward(x, T), Y);
    print(f"CrossEntropyLoss, numericGradient2 {getErrorText('dY error', dY1, dYN)}");
    print("\n");


def testCrossEntropyLossGradient3():
    N, C = 32, 24;
    X, T = np.random.randn(N, C), np.zeros((N, C));
    Y = softmax(X);
    T[np.arange(N), np.argmax(X, axis = -1)] = 1;
    M = np.random.randn(N) > 0;
    m = CrossEntropyLoss();
    loss = m.forward(Y, M, T);
    dY1 = m.backward()[0];
    dYN = numericGradient(lambda x: m.forward(x, M, T), Y);
    print(f"CrossEntropyLoss, numericGradient3 {getErrorText('dY error', dY1, dYN)}");
    print("\n");


def testCrossEntropyLossGradient4():
    N, D, C = 32, 10, 24;
    X, T = np.random.randn(N, D, C), np.zeros((N, D, C));
    index = np.argmax(X.reshape(-1, C), axis = -1);
    Y = softmax(X);
    T.reshape(-1, C)[np.arange(N * D), index] = 1;
    M = np.random.randn(N, D) > 0;
    m = CrossEntropyLoss();
    loss = m.forward(Y, M, T);
    dY1 = m.backward()[0];
    dYN = numericGradient(lambda x: m.forward(x, M, T), Y);
    print(f"CrossEntropyLoss, numericGradient4 {getErrorText('dY error', dY1, dYN)}");
    print("\n");


def testCrossEntropyLossGradient5():
    N, D, C = 32, 10, 24;
    X, T = np.random.randn(N, D, C), np.zeros((N, D, C));
    index = np.argmax(X.reshape(-1, C), axis = -1);
    Y = softmax(X);
    T.reshape(-1, C)[np.arange(N * D), index] = 1;
    M = np.random.randn(N, D) > 0;
    m = CrossEntropyLoss(reductionType = LossReductionType.Sum);
    loss = m.forward(Y, M, T);
    dY1 = m.backward()[0];
    dYN = numericGradient(lambda x: m.forward(x, M, T), Y);
    print(f"CrossEntropyLoss, numericGradient5 {getErrorText('dY error', dY1, dYN)}");
    print("\n");


def testCrossEntropyLossGradient6():
    N, D, C = 32, 10, 24;
    X, T = np.random.randn(N, D, C), np.zeros((N, D, C));
    index = np.argmax(X.reshape(-1, C), axis = -1);
    Y = softmax(X);
    T.reshape(-1, C)[np.arange(N * D), index] = 1;
    M = np.random.randn(N, D) > 0;
    m = CrossEntropyLoss(reductionType = LossReductionType.No);
    loss = np.sum(m.forward(Y, M, T));
    dY1 = m.backward()[0];
    dYN = numericGradient(lambda x: np.sum(m.forward(x, M, T)), Y);
    print(f"CrossEntropyLoss, numericGradient6 {getErrorText('dY error', dY1, dYN)}");
    print("\n");


def testSoftmaxWithCrossEntropyLossGradient1():
    N, C = 32, 24;
    X, T = np.random.randn(N, C), np.zeros((N, C));
    T[np.arange(N), np.argmax(X, axis = -1)] = 1;
    m = SoftmaxWithCrossEntropyLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"SoftmaxWithCrossEntropyLoss, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropyLossGradient2():
    N, D, C = 32, 24, 10;
    X, T = np.random.randn(N, D, C), np.zeros((N, D, C));
    index = np.argmax(X.reshape(-1, C), axis = -1);
    T.reshape(-1, C)[np.arange(N * D), index] = 1;
    m = SoftmaxWithCrossEntropyLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"SoftmaxWithCrossEntropyLoss, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropyLossGradient3():
    N, C = 32, 24;
    X, T = np.random.randn(N, C), np.zeros((N, C));
    T[np.arange(N), np.argmax(X, axis = -1)] = 1;
    M = np.random.randn(N) > 0;
    m = SoftmaxWithCrossEntropyLoss();
    loss = m.forward(X, M, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, M, T), X);
    print(f"SoftmaxWithCrossEntropyLoss, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropyLossGradient4():
    N, D, C = 32, 24, 10;
    X, T = np.random.randn(N, D, C), np.zeros((N, D, C));
    index = np.argmax(X.reshape(-1, C), axis = -1);
    T.reshape(-1, C)[np.arange(N * D), index] = 1;
    M = np.random.randn(N, D) > 0;
    m = SoftmaxWithCrossEntropyLoss();
    loss = m.forward(X, M, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, M, T), X);
    print(f"SoftmaxWithCrossEntropyLoss, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropyLossGradient5():
    N, C = 32, 24;
    X, T = np.random.randn(N, C), np.zeros((N, C));
    T[np.arange(N), np.argmax(X, axis = -1)] = 1;
    M = np.random.randn(N) > 0;
    m = SoftmaxWithCrossEntropyLoss(reductionType = LossReductionType.Sum);
    loss = m.forward(X, M, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, M, T), X);
    print(f"SoftmaxWithCrossEntropyLoss, numericGradient5 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropyLossGradient6():
    N, D, C = 32, 24, 10;
    X, T = np.random.randn(N, D, C), np.zeros((N, D, C));
    index = np.argmax(X.reshape(-1, C), axis = -1);
    T.reshape(-1, C)[np.arange(N * D), index] = 1;
    M = np.random.randn(N, D) > 0;
    m = SoftmaxWithCrossEntropyLoss(reductionType = LossReductionType.No);
    loss = np.sum(m.forward(X, M, T));
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M, T)), X);
    print(f"SoftmaxWithCrossEntropyLoss, numericGradient6 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSigmoidWithCrossEntropyLossGradient1():
    N = 32;
    X, T = np.random.randn(N), np.random.choice(np.arange(2), size = N, replace = True);
    m = SigmoidWithCrossEntropyLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"SigmoidWithCrossEntropyLoss, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSigmoidWithCrossEntropyLossGradient2():
    N, D = 32, 24;
    X, T = np.random.randn(N, D), np.random.choice(np.arange(2), size = (N, D), replace = True);
    m = SigmoidWithCrossEntropyLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"SigmoidWithCrossEntropyLoss, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSigmoidWithCrossEntropyLossGradient3():
    N = 32;
    X, T = np.random.randn(N), np.random.choice(np.arange(2), size = N, replace = True);
    M = np.random.randn(N) > 0;
    m = SigmoidWithCrossEntropyLoss();
    loss = m.forward(X, M, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, M, T), X);
    print(f"SigmoidWithCrossEntropyLoss, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSigmoidWithCrossEntropyLossGradient4():
    N, D = 32, 24;
    X, T = np.random.randn(N, D), np.random.choice(np.arange(2), size = (N, D), replace = True);
    M = np.random.randn(N, D) > 0;
    m = SigmoidWithCrossEntropyLoss();
    loss = m.forward(X, M, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, M, T), X);
    print(f"SigmoidWithCrossEntropyLoss, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSigmoidWithCrossEntropyLossGradient5():
    N = 32;
    X, T = np.random.randn(N), np.random.choice(np.arange(2), size = N, replace = True);
    M = np.random.randn(N) > 0;
    m = SigmoidWithCrossEntropyLoss(reductionType = LossReductionType.Sum);
    loss = m.forward(X, M, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, M, T), X);
    print(f"SigmoidWithCrossEntropyLoss, numericGradient5 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSigmoidWithCrossEntropyLossGradient6():
    N, D = 32, 24;
    X, T = np.random.randn(N, D), np.random.choice(np.arange(2), size = (N, D), replace = True);
    M = np.random.randn(N, D) > 0;
    m = SigmoidWithCrossEntropyLoss(reductionType = LossReductionType.No);
    loss = np.sum(m.forward(X, M, T));
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M, T)), X);
    print(f"SigmoidWithCrossEntropyLoss, numericGradient6 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropy1DLossGradient1():
    N, C = 32, 24;
    X, T = np.random.randn(N, C), np.random.choice(np.arange(C), N, replace = True);
    m = SoftmaxWithCrossEntropy1DLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"SoftmaxWithCrossEntropy1DLoss, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropy1DLossGradient2():
    N, D, C = 32, 24, 10;
    X, T = np.random.randn(N, D, C), np.random.choice(np.arange(C), size = (N, D), replace = True);
    m = SoftmaxWithCrossEntropy1DLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"SoftmaxWithCrossEntropy1DLoss, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropy1DLossGradient3():
    N, C = 32, 24;
    X, T = np.random.randn(N, C), np.random.choice(np.arange(C), N, replace = True);
    M = np.random.randn(N) > 0;
    m = SoftmaxWithCrossEntropy1DLoss();
    loss = m.forward(X, M, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, M, T), X);
    print(f"SoftmaxWithCrossEntropy1DLoss, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropy1DLossGradient4():
    N, D, C = 32, 24, 10;
    X, T = np.random.randn(N, D, C), np.random.choice(np.arange(C), size = (N, D), replace = True);
    M = np.random.randn(N, D) > 0;
    m = SoftmaxWithCrossEntropy1DLoss();
    loss = m.forward(X, M, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, M, T), X);
    print(f"SoftmaxWithCrossEntropy1DLoss, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropy1DLossGradient5():
    N, C = 32, 24;
    X, T = np.random.randn(N, C), np.random.choice(np.arange(C), N, replace = True);
    M = np.random.randn(N) > 0;
    m = SoftmaxWithCrossEntropy1DLoss(reductionType = LossReductionType.Sum);
    loss = m.forward(X, M, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, M, T), X);
    print(f"SoftmaxWithCrossEntropy1DLoss, numericGradient5 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftmaxWithCrossEntropy1DLossGradient6():
    N, D, C = 32, 24, 10;
    X, T = np.random.randn(N, D, C), np.random.choice(np.arange(C), size = (N, D), replace = True);
    M = np.random.randn(N, D) > 0;
    m = SoftmaxWithCrossEntropy1DLoss(reductionType = LossReductionType.No);
    loss = np.sum(m.forward(X, M, T));
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M, T)), X);
    print(f"SoftmaxWithCrossEntropy1DLoss, numericGradient6 {getErrorText('dX error', dX1, dXN)}");
    print("\n");

def testSequenceSoftmaxWithCrossEntropy1DLossGradient1():
    N, D, C = 32, 24, 10;
    X, T = np.random.randn(N, D, C), np.random.choice(np.arange(C), size = (N, D), replace = True);
    validLen = np.random.choice(np.arange(D), size = N, replace = True);
    m = SequenceSoftmaxWithCrossEntropy1DLoss();
    loss = m.forward(X, validLen, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, validLen, T), X);
    print(f"SequenceSoftmaxWithCrossEntropy1DLoss, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanSquareLossGradient1():
    N = 32;
    X, T = np.random.randn(N), np.random.randn(N);
    m = IdentityWithMeanSquareLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithMeanSquareLoss, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanSquareLossGradient2():
    N, D = 32, 24;
    X, T = np.random.randn(N, D), np.random.randn(N, D);
    m = IdentityWithMeanSquareLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithMeanSquareLoss, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanSquareLossGradient3():
    N, C, D = 32, 24, 10;
    X, T = np.random.randn(N, C, D), np.random.randn(N, C, D);
    m = IdentityWithMeanSquareLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithMeanSquareLoss, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithMeanSquareLossGradient4():
    N, C, D = 32, 24, 10;
    X, W, T = np.random.randn(N, C, D), np.random.randn(N, C, D), np.random.randn(N, C, D);
    m = IdentityWithMeanSquareLoss();
    loss = m.forward(X, W, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, W, T), X);
    print(f"IdentityWithMeanSquareLoss, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSumWithMeanSquareLossGradient1():
    N, D = 32, 24;
    X, T = np.random.randn(N, D), np.random.randn(N, 1);
    m = SumWithMeanSquareLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"SumWithMeanSquareLoss, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSumWithMeanSquareLossGradient2():
    N, C, D = 32, 24, 10;
    X, T = np.random.randn(N, C, D), np.random.randn(N, C, 1);
    m = SumWithMeanSquareLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"SumWithMeanSquareLoss, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testPReluLayerGradient1():
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D);
    m = PReluLayer();
    beta = m.beta;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dBeta1 = m.params[0].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dBetaN = numericGradient(lambda x: np.sum(PReluLayer(beta = x).forward(X)[0]), beta);
    print(f"PReluLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}{getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testPReluLayerGradient2():
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D);
    m = PReluLayer(outputSize = D);
    beta = m.beta;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dBeta1 = m.params[0].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dBetaN = numericGradient(lambda x: np.sum(PReluLayer(beta = x).forward(X)[0]), beta);
    print(f"PReluLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}{getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testPReluLayerGradient3():
    N, D = 256, 256;
    X = np.random.randn(N, D);
    m = PReluLayer(outputSize = D);
    beta = m.beta;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dBeta1 = m.params[0].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dBetaN = numericGradient(lambda x: np.sum(PReluLayer(beta = x).forward(X)[0]), beta);
    print(f"PReluLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}{getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testSoftplus1():
    X = np.array([-25.0, -21, 21, 25]);
    Y = softplus(X);

    X1, X2 = X[:2], X[2:];
    Y1 = np.log(1 + np.exp(X1));
    Y2 = X2;

    print(f"Softplus, value1, error: {np.sum(np.abs(Y - np.concatenate((Y1, Y2))))}");
    print("\n");


def testSoftplusLayerGradient1():
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D) * 0.1;
    m = SoftplusLayer();
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"SoftplusLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftplusLayerGradient2():
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D) * 20;
    m = SoftplusLayer();
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"SoftplusLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSoftplusLayerGradient3():
    N, D = 256, 256;
    X = np.random.randn(N, D) * 20;
    m = SoftplusLayer();
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"SoftplusLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSwishLayerGradient1():
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D);
    m = SwishLayer();
    beta = m.beta;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dBeta1 = m.params[0].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dBetaN = numericGradient(lambda x: np.sum(SwishLayer(beta = x).forward(X)[0]), beta);
    print(f"SwishLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}{getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testSwishLayerGradient2():
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D);
    m = SwishLayer(outputSize = D);
    beta = m.beta;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dBeta1 = m.params[0].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dBetaN = numericGradient(lambda x: np.sum(SwishLayer(beta = x).forward(X)[0]), beta);
    print(f"SwishLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}{getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testSwishLayerGradient3():
    N, D = 256, 256;
    X = np.random.randn(N, D);
    m = SwishLayer(outputSize = D);
    beta = m.beta;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dBeta1 = m.params[0].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dBetaN = numericGradient(lambda x: np.sum(SwishLayer(beta = x).forward(X)[0]), beta);
    print(f"SwishLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}{getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testSiluLayerGradient1():
    N, D = 256, 256;
    X = np.random.randn(N, D);
    m = SiluLayer();
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"SiluLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSiluLayerGradient2():
    N, D = 256, 256;
    X = np.random.randn(N, D);
    C = np.random.randn(N, D);
    m = SequentialContainer(
        SiluLayer(),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"SiluLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testGeluLayerGradient1():
    N, D = 256, 256;
    X = np.random.randn(N, D);
    m = GeluLayer();
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"GeluLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testGeluLayerGradient2():
    N, D = 256, 256;
    X = np.random.randn(N, D);
    C = np.random.randn(N, D);
    m = SequentialContainer(
        GeluLayer(),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"GeluLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMaxoutLayer1():
    N, T, D = 2, 3, 6;
    X = np.random.randn(N, T, D);
    m = MaxoutLayer();
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX = m.backward(np.ones_like(Y))[0];
    assert(Y.shape == (N, T, D // m.K));
    assert(dX.shape == X.shape);
    print("\n");


def testMaxoutLayer2():
    x = np.linspace(0, 40, 100).reshape(-1, 1);
    y = 2.2 * x + 3.3 + np.random.randn(*x.shape) * 5;

    plt.figure();
    plt.scatter(x.flatten(), y.flatten());
    plt.show(block = True);

    batchSize, maxEcho = 10, 100;
    inputSize, outputSize = 1, 1;
    trainIterator = SequentialDataIterator([x, y], batchSize = batchSize, shuffle = True);
    testIterator = SequentialDataIterator([x, y], batchSize = batchSize, shuffle = False);
    lossFunc = IdentityWithMeanSquareLoss();
    optimizer = SGD(0.001);
    # optimizer = NN.Adam(0.01);
    evaluator = MaeAccuracyEvaluator();

    model = SequentialContainer(
        AffineLayer(inputSize, outputSize),
        # MaxoutLayer(2),
        # AffineLayer(outputSize, outputSize),
    );
    model.fit(trainIterator, lossFunc, optimizer, maxEcho, testIterator, evaluator, plot = True);

    plt.figure();
    plt.scatter(x.flatten(), y.flatten());
    plt.plot()
    plt.show(block = True);

    print("\n");


def testMaxoutLayerGradient1():
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D);
    m = MaxoutLayer();
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxoutLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMaxoutLayerGradient2():
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D);
    m = MaxoutLayer(4);
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxoutLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMaxoutLayerGradient3():
    N, D = 256, 256;
    X = np.random.randn(N, D);
    m = MaxoutLayer(8);
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxoutLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithHuberLossGradient1():
    N = 32;
    X = np.random.randn(N);
    T = np.random.randn(N);
    m = IdentityWithHuberLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithHuberLoss, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithHuberLossGradient2():
    N, D = 32, 24;
    X = np.random.randn(N, D);
    T = np.random.randn(N, D);
    m = IdentityWithHuberLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithHuberLoss, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithHuberLossGradient3():
    N, C, D = 32, 24, 10;
    X = np.random.randn(N, C, D);
    T = np.random.randn(N, C, D);
    m = IdentityWithHuberLoss();
    loss = m.forward(X, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, T), X);
    print(f"IdentityWithHuberLoss, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testIdentityWithHuberLossGradient4():
    N, C, D = 32, 24, 10;
    X = np.random.randn(N, C, D);
    W = np.fabs(np.random.randn(N, C, D));
    T = np.random.randn(N, C, D);
    m = IdentityWithHuberLoss();
    loss = m.forward(X, W, T);
    dX1 = m.backward()[0];
    dXN = numericGradient(lambda x: m.forward(x, W, T), X);
    print(f"IdentityWithHuberLoss, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testAffineLayerGradient1():
    N, C, T, inputSize, outputSize = 32, 3, 24, 12, 16;
    X, W, b = np.random.randn(N, C, T, inputSize), np.random.randn(inputSize, outputSize), np.random.randn(outputSize);
    m = AffineLayer(inputSize, outputSize, W = W, b = b);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1, db1 = tuple([p.grad for p in m.params]);
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWN = numericGradient(lambda x: np.sum(AffineLayer(inputSize, outputSize, W = x, b = b).forward(X)[0]), W);
    dbN = numericGradient(lambda x: np.sum(AffineLayer(inputSize, outputSize, W = W, b = x).forward(X)[0]), b);
    print(f"AffineLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}{getErrorText('dW error', dW1, dWN)}{getErrorText('db error', db1, dbN)}");
    print("\n");


def testAffineLayerGradient2():
    N1, N2, T, inputSize, outputSize = 64, 32, 11, 16, 24;
    X = np.random.randn(N1, N2, T, inputSize);
    m1 = AffineLayer(inputSize, outputSize, includeBias = False);
    W = m1.params[0].value;
    m2 = AffineLayer(inputSize, outputSize, includeBias = False, W = W);

    Y1 = np.zeros(X.shape[:-1] + (outputSize, ));
    dX1, dW1 = np.zeros_like(X), np.zeros_like(W);
    for t in range(X.shape[-2]):
        Y1[..., t, :] = m1.forward(X[..., t, :])[0];
        dX = m1.backward(np.ones_like(Y1[..., t, :]))[0];
        dX1[..., t, :] = dX;
        dW1 += m1.params[0].grad;

    Y2 = m2.forward(X)[0];
    dX2 = m2.backward(np.ones_like(Y2))[0];
    dW2 = m2.params[0].grad;
    print(f"AffineLayer, numericGradient2 {getErrorText('Y error', Y1, Y2)}{getErrorText('dX error', dX1, dX2)}{getErrorText('dW error', dW1, dW2)}");
    print("\n");


def testAffineLayerGradient3():
    N, D, outputSize = 1024, 200, 1;
    X = np.random.randn(N, D);
    X[:, 3:] = 0;
    m = AffineLayer(D, outputSize);
    W, b = m.params[0].value, m.params[1].value;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1, db1 = tuple([p.grad for p in m.params]);
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWN = numericGradient(lambda x: np.sum(AffineLayer(D, outputSize, W = x, b = b).forward(X)[0]), W);
    dbN = numericGradient(lambda x: np.sum(AffineLayer(D, outputSize, W = W, b = x).forward(X)[0]), b);
    print(f"AffineLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}{getErrorText('dW error', dW1, dWN)}{getErrorText('db error', db1, dbN)}");
    print("\n");


def testAffineLayerGradient4():
    N, C, T, inputSize, hiddenSize1, hiddenSize2, outputSize = 2, 3, 24, 12, 16, 20, 24;
    X = np.random.randn(N, C, T, inputSize);
    m = AggregateNetModule(
        AffineLayer(inputSize, hiddenSize1, W = np.random.randn(inputSize, hiddenSize1), b = np.random.randn(hiddenSize1)),
        SoftplusLayer(),
        AffineLayer(hiddenSize1, hiddenSize2, W = np.random.randn(hiddenSize1, hiddenSize2), b = np.random.randn(hiddenSize2)),
        SoftplusLayer(),
        AffineLayer(hiddenSize2, outputSize, W = np.random.randn(hiddenSize2, outputSize), b = np.random.randn(outputSize)),
    );
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AffineLayer, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "AffineLayer, numericGradient4", X);
    print("\n");


def testConvolution1DLayer1():
    N, T, D = 32, 48, 8;
    FN, FW, S, P = 16, 3, 1, 0;
    X = np.random.randn(N, D, T);
    W = np.random.randn(FN, D, FW);
    b = np.random.randn(FN);

    m = Convolution1DLayer(D, FN, FW, S, P, W = W, b = b);
    Y1 = m.forward(X)[0];

    OT = convOutputSize(T, FW, S, P);
    Y2 = np.zeros((N, FN, OT));
    for i in range(N):
        for j in range(FN):
            for k in range(OT):
                t = k * S;
                x = X[i, :, t: t + FW];

                Y2[i, j, k] = np.sum(x * W[j, :, :]) + b[j];

    print(f"Convolution1DLayer, value1 {getErrorText('Y error', Y1, Y2)}");
    print("\n");


def testConvolution1DLayer2():
    def innerTest(x : np.ndarray, p : int, s : int, d : int):
        m1 = torch.nn.Conv1d(inputChannel, outputChannel, kernelSize, stride = s, padding = p, dilation = d);
        X1 = torch.tensor(X, dtype = torch.float32, requires_grad = True);
        Y1 = m1(X1);
        torch.sum(Y1).backward();
        Y1 = Y1.detach().numpy();
        dX1 = X1.grad.detach().numpy();
        dW1 = m1.weight.grad.detach().numpy();
        db1 = m1.bias.grad.detach().numpy();

        m2 = Convolution1DLayer(inputChannel, outputChannel, kernelSize, stride = s, padding = p, dilation = d, W = m1.weight.detach().numpy(), b = m1.bias.detach().numpy());
        X2 = X;
        Y2, = m2.forward(X2);
        dX2, = m2.backward(np.ones_like(Y2));
        dW2, db2 = m2.params[0].grad, m2.params[1].grad;
        print(f"Convolution1DLayer, value2, padding = {p}, stride = {s}, dilation = {d} {getErrorText('Y error', Y1, Y2)} {getErrorText('dX error', dX1, dX2)} {getErrorText('dW error', dW1, dW2)} {getErrorText('db error', db1, db2)}");


    batchSize, inputChannel, outputChannel, inputSize, kernelSize = 32, 16, 24, 49, 3;
    X = np.random.randn(batchSize, inputChannel, inputSize);

    paddings, strides, dilations = (0, 2), (1, 2, 3), (1, 3);

    for padding in paddings:
        for stride in strides:
            for dilation in dilations:
                innerTest(X, padding, stride, dilation);

    print("\n");


def testConvolution1DLayer3():
    N, T, D = 32, 49, 8;
    FN, FW, S = 16, 4, 2;
    X = np.random.randn(N, D, T);

    P, R = ("same", "causal"), (1, 3);

    for p in P:
        for r in R:
            m = Convolution1DLayer(D, FN, FW, stride = S, padding = p, dilation = r);
            Y, = m.forward(X);
            Y, = m.forward(X);
            assert Y.shape[-1] == X.shape[-1], "the length is not same";
    
    print(f"Convolution1DLayer, value3, padding test pass.");
    print("\n");


def testConvolution1DLayerGradient1():
    N, T, D = 32, 48, 8;
    FN, FW, S, P = 16, 3, 1, 0;
    X = np.random.randn(N, D, T);
    W = np.random.randn(FN, D, FW);
    b = np.random.randn(FN);
    m = Convolution1DLayer(D, FN, FW, S, P, W = W, b = b);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1, db1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWN = numericGradient(lambda x: np.sum(Convolution1DLayer(D, FN, FW, S, P, W = x, b = b).forward(X)[0]), W);
    dbN = numericGradient(lambda x: np.sum(Convolution1DLayer(D, FN, FW, S, P, W = W, b = x).forward(X)[0]), b);
    print(f"Convolution1DLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)} {getErrorText('dW error', dW1, dWN)} {getErrorText('db error', db1, dbN)}");
    print("\n");


def testConvolution1DLayerGradient2():
    N, T, D = 32, 48, 8;
    FN, FW, S, P = 16, 3, 2, (12, 13);
    X = np.random.randn(N, D, T);
    W = np.random.randn(FN, D, FW);
    b = np.random.randn(FN);
    m = Convolution1DLayer(D, FN, FW, S, P, W = W, b = b);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1, db1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWN = numericGradient(lambda x: np.sum(Convolution1DLayer(D, FN, FW, S, P, W = x, b = b).forward(X)[0]), W);
    dbN = numericGradient(lambda x: np.sum(Convolution1DLayer(D, FN, FW, S, P, W = W, b = x).forward(X)[0]), b);
    print(f"Convolution1DLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)} {getErrorText('dW error', dW1, dWN)} {getErrorText('db error', db1, dbN)}");
    print("\n");


def testConvolution1DLayerGradient3():
    N, T, D = 32, 49, 8;
    FN, FW, S, P, R = 16, 3, 2, (2, 4), 3;
    X = np.random.randn(N, D, T);
    W = np.random.randn(FN, D, FW);
    b = np.random.randn(FN);
    m = Convolution1DLayer(D, FN, FW, S, P, dilation = R, W = W, b = b);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1, db1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWN = numericGradient(lambda x: np.sum(Convolution1DLayer(D, FN, FW, S, P, dilation = R, W = x, b = b).forward(X)[0]), W);
    dbN = numericGradient(lambda x: np.sum(Convolution1DLayer(D, FN, FW, S, P, dilation = R, W = W, b = x).forward(X)[0]), b);
    print(f"Convolution1DLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)} {getErrorText('dW error', dW1, dWN)} {getErrorText('db error', db1, dbN)}");
    print("\n");


def testConvolution1DLayerGradient4():
    N, T, D = 32, 49, 8;
    FN, FW, S, P, R = 16, 3, 5, (2, 4), 3;
    X = np.random.randn(N, D, T);
    W = np.random.randn(FN, D, FW);
    b = np.random.randn(FN);
    m = Convolution1DLayer(D, FN, FW, S, P, dilation = R, W = W, b = b);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1, db1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWN = numericGradient(lambda x: np.sum(Convolution1DLayer(D, FN, FW, S, P, dilation = R, W = x, b = b).forward(X)[0]), W);
    dbN = numericGradient(lambda x: np.sum(Convolution1DLayer(D, FN, FW, S, P, dilation = R, W = W, b = x).forward(X)[0]), b);
    print(f"Convolution1DLayer, numericGradient4 {getErrorText('dX error', dX1, dXN)} {getErrorText('dW error', dW1, dWN)} {getErrorText('db error', db1, dbN)}");
    print("\n");


def testMaxPooling1DLayer1():
    N, C, T = 32, 3, 48;
    PW, S, P = 13, 2, (1, 2);
    X = np.random.randn(N, C, T);
    m = MaxPooling1DLayer(PW, S, P);
    m.context.isTrainingMode = True;
    Y1 = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y1))[0];

    OT = convOutputSize(T, PW, S, P[0] + P[1]);
    Y2 = np.zeros((N, C, OT));
    PX = np.pad(X, [(0, 0), (0, 0), (P[0], P[1])], "constant");
    dPX = np.zeros_like(PX);

    for n in range(N):
        for c in range(C):
            for i in range(OT):
                x = PX[n, c, i * S: i * S + PW];
                Y2[n, c, i] = np.amax(x, axis = None);
                idx = np.argmax(x, axis = None);
                dPX[n, c, i * S + idx] += 1;
    dX2 = dPX[:, :, P[0]: T + P[0]];

    print(f"MaxPooling1DLayer, value1 {getErrorText('Y error', Y1, Y2)} {getErrorText('dX error', dX1, dX2)}");
    print("\n");


def testMaxPooling1DLayerGradient1():
    N, C, T = 32, 3, 28;
    PW, S, P = 3, 1, 0;
    X = np.random.randn(N, C, T);
    m = MaxPooling1DLayer(PW, S, P);
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    m.context.isTrainingMode = False;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxPooling1DLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMaxPooling1DLayerGradient2():
    N, C, T = 32, 3, 24;
    PW, S, P = 4, 4, 2;
    X = np.random.randn(N, C, T);
    m = MaxPooling1DLayer(PW, S, P);
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    m.context.isTrainingMode = False;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxPooling1DLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMaxPooling1DLayerGradient3():
    N, C, T = 32, 3, 24;
    PW, S, P = 3, 5, (3, 4);
    X = np.random.randn(N, C, T);
    m = MaxPooling1DLayer(PW, S, P);
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    m.context.isTrainingMode = False;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxPooling1DLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMaxPooling1DLayerGradient4():
    N, C, T = 32, 3, 24;
    PW, S, P = 4, 4, 2;
    OT = convOutputSize(T, PW, S, 2 * P);
    X = np.random.randn(N, C, T);
    m = SequentialContainer(
        MaxPooling1DLayer(PW, S, P),
        AffineLayer(OT, 2 * OT),
    );
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    m.context.isTrainingMode = False;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxPooling1DLayer, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testAvgPooling1DLayer1():
    N, C, T = 32, 3, 50;
    PW, S, P = 13, 4, (3, 4);
    X = np.random.randn(N, C, T);
    m = AvgPooling1DLayer(PW, S, P);
    Y1 = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y1))[0];

    dm = 1.0 / PW;
    OT = convOutputSize(T, PW, S, P[0] + P[1]);
    Y2 = np.zeros((N, C, OT));
    PX = np.pad(X, [(0, 0), (0, 0), (P[0], P[1])], "constant");
    dPX = np.zeros_like(PX);

    for n in range(N):
        for c in range(C):
            for i in range(OT):
                x = PX[n, c, i * S: i * S + PW];
                Y2[n, c, i] = np.mean(x);
                dPX[n, c, i * S: i * S + PW] += dm;
    dX2 = dPX[:, :, P[0]: T + P[0]];

    print(f"AvgPooling1DLayer, value1 {getErrorText('Y error', Y1, Y2)} {getErrorText('dX error', dX1, dX2)}");
    print("\n");


def testAvgPooling1DLayerGradient1():
    N, C, T = 32, 3, 28;
    PW, S, P = 3, 1, 0;
    X = np.random.randn(N, C, T);
    m = AvgPooling1DLayer(PW, S, P);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AvgPooling1DLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testAvgPooling1DLayerGradient2():
    N, C, T = 32, 3, 24;
    PW, S, P = 4, 4, 2;
    X = np.random.randn(N, C, T);
    m = AvgPooling1DLayer(PW, S, P);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AvgPooling1DLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testAvgPooling1DLayerGradient3():
    N, C, T = 32, 3, 24;
    PW, S, P = 3, 5, (3, 4);
    X = np.random.randn(N, C, T);
    m = AvgPooling1DLayer(PW, S, P);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AvgPooling1DLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testAvgPooling1DLayerGradient4():
    N, C, T = 32, 3, 24;
    PW, S, P = 4, 4, 2;
    OT = convOutputSize(T, PW, S, 2 * P);
    X = np.random.randn(N, C, T);
    m = SequentialContainer(
        AvgPooling1DLayer(PW, S, P),
        AffineLayer(OT, 2 * OT),
    );
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AvgPooling1DLayer, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testConvolution2DLayer1():
    N, C, H, W = 32, 4, 28, 28;
    FN, FH, FW, S, P = 6, 5, 3, (2, 4), (1, 2, 3, 4);
    X = np.random.randn(N, C, H, W);
    Weight = np.random.randn(FN, C, FH, FW);
    bias = np.random.randn(FN);

    m = Convolution2DLayer(C, FN, (FH, FW), S, P, W = Weight, b = bias);
    Y1 = m.forward(X)[0];

    OH = convOutputSize(H, FH, S[0], P[0] + P[1]);
    OW = convOutputSize(W, FW, S[1], P[2] + P[3]);
    Y2 = np.zeros((N, FN, OH, OW));
    PX = np.pad(X, [(0, 0), (0, 0), (P[0], P[1]), (P[2], P[3])], "constant");
    for n in range(N):
        for j in range(OH):
            for i in range(OW):
                x = PX[n, :, j * S[0]: j * S[0] + FH, i * S[1]: i * S[1] + FW];

                for k in range(FN):
                    Y2[n, k, j, i] = np.sum(x * Weight[k]) + bias[k];

    print(f"Convolution2DLayer, value1 {getErrorText('Y error', Y1, Y2)}");
    print("\n");


def testConvolution2DLayer2():
    def innerTest(x : np.ndarray, p : int, s : int, d : int):
        m1 = torch.nn.Conv2d(inputChannel, outputChannel, kernelSize, stride = s, padding = p, dilation = d);
        X1 = torch.tensor(X, dtype = torch.float32, requires_grad = True);
        Y1 = m1(X1);
        torch.sum(Y1).backward();
        Y1 = Y1.detach().numpy();
        dX1 = X1.grad.detach().numpy();
        dW1 = m1.weight.grad.detach().numpy();
        db1 = m1.bias.grad.detach().numpy();

        m2 = Convolution2DLayer(inputChannel, outputChannel, kernelSize, stride = s, padding = p, dilation = d, W = m1.weight.detach().numpy(), b = m1.bias.detach().numpy());
        X2 = X;
        Y2, = m2.forward(X2);
        dX2, = m2.backward(np.ones_like(Y2));
        dW2, db2 = m2.params[0].grad, m2.params[1].grad;
        print(f"Convolution2DLayer, value2, padding = {p}, stride = {s}, dilation = {d} {getErrorText('Y error', Y1, Y2)} {getErrorText('dX error', dX1, dX2)} {getErrorText('dW error', dW1, dW2)} {getErrorText('db error', db1, db2)}");


    batchSize, inputChannel, outputChannel, imageHeight, imageWidth, kernelSize = 32, 16, 24, 51, 49, 3;
    X = np.random.randn(batchSize, inputChannel, imageHeight, imageWidth);

    paddings, strides, dilations = (0, 2), (1, 2, 5), (1, 3);

    for padding in paddings:
        for stride in strides:
            for dilation in dilations:
                innerTest(X, padding, stride, dilation);

    print("\n");


def testConvolution2DLayer3():
    N, C, H, W = 32, 8, 49, 51;
    FN, FH, FW, S = 16, 4, 6, 2;
    X = np.random.randn(N, C, H, W);

    P, R = ("same", "causal"), (1, 3);

    for p in P:
        for r in R:
            m = Convolution2DLayer(C, FN, (FH, FW), stride = S, padding = p, dilation = r);
            Y, = m.forward(X);
            Y, = m.forward(X);
            assert Y.shape[-2: ] == X.shape[-2: ], "the length is not same";
    
    print(f"Convolution2DLayer, value3, padding test pass.");
    print("\n");


def testConvolution2DLayerGradient1():
    N, C, H, W = 32, 3, 28, 28;
    FN, FH, FW, S, P = 4, 3, 3, 1, 0;
    X = np.random.randn(N, C, H, W);
    Weight = np.random.randn(FN, C, FH, FW);
    bias = np.random.randn(FN);
    m = Convolution2DLayer(C, FN, (FH, FW), S, P, W = Weight, b = bias);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1, db1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWN = numericGradient(lambda x: np.sum(Convolution2DLayer(C, FN, (FH, FW), S, P, W = x, b = bias).forward(X)[0]), Weight);
    dbN = numericGradient(lambda x: np.sum(Convolution2DLayer(C, FN, (FH, FW), S, P, W = Weight, b = x).forward(X)[0]), bias);
    print(f"Convolution2DLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)} {getErrorText('dW error', dW1, dWN)} {getErrorText('db error', db1, dbN)}");
    print("\n");


def testConvolution2DLayerGradient2():
    N, C, H, W = 32, 3, 28, 28;
    FN, FH, FW, S, P = 6, 4, 4, 2, 2;
    X = np.random.randn(N, C, H, W);
    Weight = np.random.randn(FN, C, FH, FW);
    bias = np.random.randn(FN);
    m = Convolution2DLayer(C, FN, (FH, FW), S, P, W = Weight, b = bias);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1, db1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWN = numericGradient(lambda x: np.sum(Convolution2DLayer(C, FN, (FH, FW), S, P, W = x, b = bias).forward(X)[0]), Weight);
    dbN = numericGradient(lambda x: np.sum(Convolution2DLayer(C, FN, (FH, FW), S, P, W = Weight, b = x).forward(X)[0]), bias);
    print(f"Convolution2DLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)} {getErrorText('dW error', dW1, dWN)} {getErrorText('db error', db1, dbN)}");
    print("\n");


def testConvolution2DLayerGradient3():
    N, C, H, W = 32, 4, 28, 28;
    FN, FH, FW, S, P = 6, 5, 3, (3, 5), (1, 2, 3, 4);
    X = np.random.randn(N, C, H, W);
    Weight = np.random.randn(FN, C, FH, FW);
    bias = np.random.randn(FN);
    m = Convolution2DLayer(C, FN, (FH, FW), S, P, W = Weight, b = bias);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1, db1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWN = numericGradient(lambda x: np.sum(Convolution2DLayer(C, FN, (FH, FW), S, P, W = x, b = bias).forward(X)[0]), Weight);
    dbN = numericGradient(lambda x: np.sum(Convolution2DLayer(C, FN, (FH, FW), S, P, W = Weight, b = x).forward(X)[0]), bias);
    print(f"Convolution2DLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)} {getErrorText('dW error', dW1, dWN)} {getErrorText('db error', db1, dbN)}");
    print("\n");


def testConvolution2DLayerGradient4():
    N, C, H, W = 32, 4, 29, 27;
    FN, FH, FW, S, P, R = 6, 5, 3, (2, 4), (1, 3, 5, 7), 3;
    X = np.random.randn(N, C, H, W);
    Weight = np.random.randn(FN, C, FH, FW);
    bias = np.random.randn(FN);
    m = Convolution2DLayer(C, FN, (FH, FW), S, P, dilation = R, W = Weight, b = bias);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1, db1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWN = numericGradient(lambda x: np.sum(Convolution2DLayer(C, FN, (FH, FW), S, P, dilation = R, W = x, b = bias).forward(X)[0]), Weight);
    dbN = numericGradient(lambda x: np.sum(Convolution2DLayer(C, FN, (FH, FW), S, P, dilation = R, W = Weight, b = x).forward(X)[0]), bias);
    print(f"Convolution2DLayer, numericGradient4 {getErrorText('dX error', dX1, dXN)} {getErrorText('dW error', dW1, dWN)} {getErrorText('db error', db1, dbN)}");
    print("\n");


def testMaxPooling2DLayer1():
    N, C, H, W = 32, 3, 48, 50;
    PH, PW, S, P = 9, 13, (2, 4), (1, 2, 3, 4);
    X = np.random.randn(N, C, H, W);
    m = MaxPooling2DLayer((PH, PW), S, P);
    m.context.isTrainingMode = True;
    Y1 = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y1))[0];

    OH = convOutputSize(H, PH, S[0], P[0] + P[1]);
    OW = convOutputSize(W, PW, S[1], P[2] + P[3]);
    Y2 = np.zeros((N, C, OH, OW));
    PX = np.pad(X, [(0, 0), (0, 0), (P[0], P[1]), (P[2], P[3])], "constant");
    dPX = np.zeros_like(PX);

    for n in range(N):
        for c in range(C):
            for j in range(OH):
                for i in range(OW):
                    x = PX[n, c, j * S[0]: j * S[0] + PH, i * S[1]: i * S[1] + PW];
                    Y2[n, c, j, i] = np.amax(x, axis = None);
                    idx = np.argmax(x, axis = None);
                    dPX[n, c, j * S[0] + idx // PW, i * S[1] + idx % PW] += 1;
    dX2 = dPX[:, :, P[0]: H + P[0], P[2]: W + P[2]];

    print(f"MaxPooling2DLayer, value1 {getErrorText('Y error', Y1, Y2)} {getErrorText('dX error', dX1, dX2)}");
    print("\n");


def testMaxPooling2DLayerGradient1():
    N, C, H, W = 32, 3, 28, 28;
    PH, PW, S, P = 3, 3, 1, 0;
    X = np.random.randn(N, C, H, W);
    m = MaxPooling2DLayer((PH, PW), S, P);
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    m.context.isTrainingMode = False;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxPooling2DLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMaxPooling2DLayerGradient2():
    N, C, H, W = 32, 3, 24, 24;
    PH, PW, S, P = 4, 4, 4, 2;
    X = np.random.randn(N, C, H, W);
    m = MaxPooling2DLayer((PH, PW), S, P);
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    m.context.isTrainingMode = False;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxPooling2DLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMaxPooling2DLayerGradient3():
    N, C, H, W = 32, 3, 24, 24;
    PH, PW, S, P = 5, 3, (3, 5), (1, 2, 3, 4);
    X = np.random.randn(N, C, H, W);
    m = MaxPooling2DLayer((PH, PW), S, P);
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    m.context.isTrainingMode = False;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxPooling2DLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMaxPooling2DLayerGradient4():
    N, C, H, W = 32, 3, 24, 24;
    PH, PW, S, P = 4, 4, 4, 2;
    OH = convOutputSize(H, PH, S, 2 * P);
    OW = convOutputSize(W, PW, S, 2 * P);
    X = np.random.randn(N, C, H, W);
    m = SequentialContainer(
        MaxPooling2DLayer((PH, PW), S, P),
        AffineLayer(OW, 2 * OW),
    );
    m.context.isTrainingMode = True;
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    m.context.isTrainingMode = False;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MaxPooling2DLayer, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testAvgPooling2DLayer1():
    N, C, H, W = 32, 3, 48, 50;
    PH, PW, S, P = 9, 13, (2, 4), (1, 2, 3, 4);
    X = np.random.randn(N, C, H, W);
    m = AvgPooling2DLayer((PH, PW), S, P);
    Y1 = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y1))[0];

    dm = 1.0 / (PH * PW);
    OH = convOutputSize(H, PH, S[0], P[0] + P[1]);
    OW = convOutputSize(W, PW, S[1], P[2] + P[3]);
    Y2 = np.zeros((N, C, OH, OW));
    PX = np.pad(X, [(0, 0), (0, 0), (P[0], P[1]), (P[2], P[3])], "constant");
    dPX = np.zeros_like(PX);

    for n in range(N):
        for c in range(C):
            for j in range(OH):
                for i in range(OW):
                    x = PX[n, c, j * S[0]: j * S[0] + PH, i * S[1]: i * S[1] + PW];
                    Y2[n, c, j, i] = np.mean(x);
                    dPX[n, c, j * S[0]: j * S[0] + PH, i * S[1]: i * S[1] + PW] += dm;
    dX2 = dPX[:, :, P[0]: H + P[0], P[2]: W + P[2]];

    print(f"AvgPooling2DLayer, value1 {getErrorText('Y error', Y1, Y2)} {getErrorText('dX error', dX1, dX2)}");
    print("\n");


def testAvgPooling2DLayerGradient1():
    N, C, H, W = 32, 3, 28, 28;
    PH, PW, S, P = 3, 3, 1, 0;
    X = np.random.randn(N, C, H, W);
    m = AvgPooling2DLayer((PH, PW), S, P);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AvgPooling2DLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testAvgPooling2DLayerGradient2():
    N, C, H, W = 32, 3, 24, 24;
    PH, PW, S, P = 4, 4, 4, 2;
    X = np.random.randn(N, C, H, W);
    m = AvgPooling2DLayer((PH, PW), S, P);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AvgPooling2DLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testAvgPooling2DLayerGradient3():
    N, C, H, W = 32, 3, 24, 24;
    PH, PW, S, P = 5, 3, (3, 5), (1, 2, 3, 4);
    X = np.random.randn(N, C, H, W);
    m = AvgPooling2DLayer((PH, PW), S, P);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AvgPooling2DLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testAvgPooling2DLayerGradient4():
    N, C, H, W = 32, 3, 24, 24;
    PH, PW, S, P = 4, 4, 4, 2;
    OH = convOutputSize(H, PH, S, 2 * P);
    OW = convOutputSize(W, PW, S, 2 * P);
    X = np.random.randn(N, C, H, W);
    m = SequentialContainer(
        AvgPooling2DLayer((PH, PW), S, P),
        AffineLayer(OW, 2 * OW),
    );
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AvgPooling2DLayer, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testBatchNormalization1DLayer1():
    N, D = 1024, 1024;
    X01 = torch.randn(N, D);
    X02 = X01.numpy();
    X11 = torch.randn(N, D);
    X12 = X11.numpy();
    X21 = torch.randn(N, D);
    X22 = X21.numpy();
    X31 = torch.randn(N, D);
    X32 = X31.numpy();

    m1 = torch.nn.BatchNorm1d(D, eps = 1e-8, momentum = 0.1);
    m1.train();

    m2 = BatchNormalization1DLayer(D, momentum = 0.1);
    m2.context.isTrainingMode = True;

    Y1 = m1.forward(X01).detach().numpy();
    Y2, = m2.forward(X02);

    print(f"BatchNormalization1DLayer, value1 {getErrorText('Y error', Y1, Y2)}");
    print(f"BatchNormalization1DLayer, value1 {getErrorText('eval mean error', m1.running_mean.detach().numpy(), m2.evalMean)}");
    print(f"BatchNormalization1DLayer, value1 {getErrorText('eval var error', m1.running_var.detach().numpy(), m2.evalVar)}");

    Y1 = m1.forward(X11).detach().numpy();
    Y2, = m2.forward(X12);

    print(f"BatchNormalization1DLayer, value1 {getErrorText('Y error', Y1, Y2)}");
    print(f"BatchNormalization1DLayer, value1 {getErrorText('eval mean error', m1.running_mean.detach().numpy(), m2.evalMean)}");
    print(f"BatchNormalization1DLayer, value1 {getErrorText('eval var error', m1.running_var.detach().numpy(), m2.evalVar)}");

    Y1 = m1.forward(X21).detach().numpy();
    Y2, = m2.forward(X22);

    print(f"BatchNormalization1DLayer, value1 {getErrorText('Y error', Y1, Y2)}");
    print(f"BatchNormalization1DLayer, value1 {getErrorText('eval mean error', m1.running_mean.detach().numpy(), m2.evalMean)}");
    print(f"BatchNormalization1DLayer, value1 {getErrorText('eval var error', m1.running_var.detach().numpy(), m2.evalVar)}");

    m1.eval();
    m2.context.isTrainingMode = False;
    Y1 = m1.forward(X31).detach().numpy();
    Y2, = m2.forward(X32);

    print(f"BatchNormalization1DLayer, value1 {getErrorText('Y error', Y1, Y2)}");
    print("\n");


def testBatchNormalization1DLayer2():
    N, D = 1024, 1024;
    X01 = torch.randn(N, D);
    X02 = X01.numpy();
    X11 = torch.randn(N, D);
    X12 = X11.numpy();
    X21 = torch.randn(N, D);
    X22 = X21.numpy();
    X31 = torch.randn(N, D);
    X32 = X31.numpy();

    m1 = torch.nn.BatchNorm1d(D, eps = 1e-8, momentum = None);
    m1.train();

    m2 = BatchNormalization1DLayer(D, momentum = None);
    m2.context.isTrainingMode = True;

    Y1 = m1.forward(X01).detach().numpy();
    m2.context.trainingIterations += 1;
    Y2, = m2.forward(X02);

    print(f"BatchNormalization1DLayer, value2 {getErrorText('Y error', Y1, Y2)}");
    print(f"BatchNormalization1DLayer, value2 {getErrorText('eval mean error', m1.running_mean.detach().numpy(), m2.evalMean)}");
    print(f"BatchNormalization1DLayer, value2 {getErrorText('eval var error', m1.running_var.detach().numpy(), m2.evalVar)}");

    Y1 = m1.forward(X11).detach().numpy();
    m2.context.trainingIterations += 1;
    Y2, = m2.forward(X12);

    print(f"BatchNormalization1DLayer, value2 {getErrorText('Y error', Y1, Y2)}");
    print(f"BatchNormalization1DLayer, value2 {getErrorText('eval mean error', m1.running_mean.detach().numpy(), m2.evalMean)}");
    print(f"BatchNormalization1DLayer, value2 {getErrorText('eval var error', m1.running_var.detach().numpy(), m2.evalVar)}");

    Y1 = m1.forward(X21).detach().numpy();
    m2.context.trainingIterations += 1;
    Y2, = m2.forward(X22);

    print(f"BatchNormalization1DLayer, value2 {getErrorText('Y error', Y1, Y2)}");
    print(f"BatchNormalization1DLayer, value2 {getErrorText('eval mean error', m1.running_mean.detach().numpy(), m2.evalMean)}");
    print(f"BatchNormalization1DLayer, value2 {getErrorText('eval var error', m1.running_var.detach().numpy(), m2.evalVar)}");

    m1.eval();
    m2.context.isTrainingMode = False;
    Y1 = m1.forward(X31).detach().numpy();
    Y2, = m2.forward(X32);

    print(f"BatchNormalization1DLayer, value2 {getErrorText('Y error', Y1, Y2)}");
    print("\n");


def testBatchNormalization1DLayerGradient1():
    def createModel(inputSize, g, b, c) -> BatchNormalization1DLayer:
        layer = SequentialContainer(
            BatchNormalization1DLayer(inputSize, gamma = g, beta = b),
            FunctionalNetModule("*C", lambda x: x * c, lambda x, y, dy: dy * c),
        )
        layer.context.isTrainingMode = True;
        return layer;


    N, T, D = 32, 24, 8;
    X = np.random.randn(N, D, T);
    C = np.random.randn(N, D, T);
    gamma = np.random.randn(D);
    beta = np.random.randn(D);
    m = createModel(D, gamma, beta, C);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dGamma1, dBeta1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dGammaN = numericGradient(lambda x: np.sum(createModel(D, x, beta, C).forward(X)[0]), gamma);
    dBetaN = numericGradient(lambda x: np.sum(createModel(D, gamma, x, C).forward(X)[0]), beta);
    print(f"BatchNormalization1DLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)} {getErrorText('dGamma error', dGamma1, dGammaN)} {getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testBatchNormalizationLayer1DGradient2():
    def createModel(inputSize, g, b, c) -> BatchNormalization1DLayer:
        layer = SequentialContainer(
            BatchNormalization1DLayer(inputSize, gamma = g, beta = b),
            FunctionalNetModule("*C", lambda x: x * c, lambda x, y, dy: dy * c)
        );
        layer.context.isTrainingMode = True;
        return layer;

    N, D = 32, 8;
    X, C = np.random.randn(N, D), np.random.randn(N, D);
    gamma = np.random.randn(D);
    beta = np.random.randn(D);
    m = createModel(D, gamma, beta, C);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dGamma1, dBeta1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dGammaN = numericGradient(lambda x: np.sum(createModel(D, x, beta, C).forward(X)[0]), gamma);
    dBetaN = numericGradient(lambda x: np.sum(createModel(D, gamma, x, C).forward(X)[0]), beta);
    print(f"BatchNormalization1DLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)} {getErrorText('dGamma error', dGamma1, dGammaN)} {getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testBatchNormalization2DLayer1():
    batchSize, channelNum, imageHeight, imageWidth = 32, 8, 12, 16;

    X = np.random.randn(batchSize, channelNum, imageHeight, imageWidth);

    X1 = torch.tensor(X, dtype = torch.float32);
    m1 = torch.nn.BatchNorm2d(channelNum, eps = 1e-8);
    m1.train();
    Y1 = m1(X1).detach().numpy();

    X2 = X;
    m2 = BatchNormalization2DLayer(channelNum);
    m2.context.isTrainingMode = True;
    m2.context.trainingIterations += 1;
    Y2, = m2.forward(X2);

    print(f"BatchNormalization2DLayer, value1 {getErrorText('Y error', Y1, Y2)}");
    print(f"BatchNormalization2DLayer, value1 {getErrorText('eval mean error', m1.running_mean.detach().numpy(), m2.evalMean)}");
    print(f"BatchNormalization2DLayer, value1 {getErrorText('eval var error', m1.running_var.detach().numpy(), m2.evalVar)}");

    X = np.random.randn(batchSize, channelNum, imageHeight, imageWidth);

    X1 = torch.tensor(X, dtype = torch.float32);
    Y1 = m1(X1).detach().numpy();

    X2 = X;
    m2.context.trainingIterations += 1;
    Y2, = m2.forward(X2);

    print(f"BatchNormalization2DLayer, value1 {getErrorText('Y error', Y1, Y2)}");
    print(f"BatchNormalization2DLayer, value1 {getErrorText('eval mean error', m1.running_mean.detach().numpy(), m2.evalMean)}");
    print(f"BatchNormalization2DLayer, value1 {getErrorText('eval var error', m1.running_var.detach().numpy(), m2.evalVar)}");

    X = np.random.randn(batchSize, channelNum, imageHeight, imageWidth);

    X1 = torch.tensor(X, dtype = torch.float32);
    m1.eval();
    Y1 = m1(X1).detach().numpy();

    X2 = X;
    m2.context.isTrainingMode = False;
    m2.context.trainingIterations += 1;
    Y2, = m2.forward(X2);

    print(f"BatchNormalization2DLayer, value1 {getErrorText('Y error', Y1, Y2)}");
    print("\n");


def testBatchNormalizationLayer2DGradient1():
    def createModel(inputSize, g, b, c) -> BatchNormalization2DLayer:
        layer = SequentialContainer(
            BatchNormalization2DLayer(inputSize, gamma = g, beta = b),
            FunctionalNetModule("*C", lambda x: x * c, lambda x, y, dy: dy * c)
        );
        layer.context.isTrainingMode = True;
        return layer;

    batchSize, channelNum, imageHeight, imageWidth = 32, 8, 12, 16;
    X = np.random.randn(batchSize, channelNum, imageHeight, imageWidth);
    C = np.random.randn(batchSize, channelNum, imageHeight, imageWidth);
    gamma = np.random.randn(channelNum);
    beta = np.random.randn(channelNum);
    m = createModel(channelNum, gamma, beta, C);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dGamma1, dBeta1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dGammaN = numericGradient(lambda x: np.sum(createModel(channelNum, x, beta, C).forward(X)[0]), gamma);
    dBetaN = numericGradient(lambda x: np.sum(createModel(channelNum, gamma, x, C).forward(X)[0]), beta);
    print(f"BatchNormalization2DLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)} {getErrorText('dGamma error', dGamma1, dGammaN)} {getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testLayerNormalizationLayer1():
    N, L, C, H, W = 32, 24, 8, 64, 128;

    X = np.random.randn(N, C);
    m1 = LayerNormalizationLayer(C);
    Y1, = m1.forward(X);
    m2 = torch.nn.LayerNorm(C, eps = 1e-8, dtype = torch.float64);
    Y2 = m2.forward(torch.tensor(X));
    Y2 = Y2.detach().numpy();
    print(f"LayerNormalizationLayer, value1, Y error: {np.sum(np.abs(Y1 - Y2))}");

    X = np.random.randn(N, L, C);
    m1 = LayerNormalizationLayer(C);
    Y1, = m1.forward(X);
    m2 = torch.nn.LayerNorm(C, eps = 1e-8, dtype = torch.float64);
    Y2 = m2.forward(torch.tensor(X));
    Y2 = Y2.detach().numpy();
    print(f"LayerNormalizationLayer, value1, Y error: {np.sum(np.abs(Y1 - Y2))}");

    X = np.random.randn(N, C, H, W);
    m1 = LayerNormalizationLayer((C, H, W));
    Y1, = m1.forward(X);
    m2 = torch.nn.LayerNorm((C, H, W), eps = 1e-8, dtype = torch.float64);
    Y2 = m2.forward(torch.tensor(X));
    Y2 = Y2.detach().numpy();
    print(f"LayerNormalizationLayer, value1, Y error: {np.sum(np.abs(Y1 - Y2))}");

    print("\n");


def testLayerNormalizationLayerGradient1():
    N, C = 32, 16;
    X = np.random.randn(N, C);
    gamma, beta = np.random.randn(C), np.random.randn(C);
    m = LayerNormalizationLayer(C, gamma = gamma, beta = beta);
    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y))[0];
    dGamma1, dBeta1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dGammaN = numericGradient(lambda x: np.sum(LayerNormalizationLayer(C, gamma = x, beta = beta).forward(X)[0]), gamma);
    dBetaN = numericGradient(lambda x: np.sum(LayerNormalizationLayer(C, gamma = gamma, beta = x).forward(X)[0]), beta);
    print(f"LayerNormalizationLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)} {getErrorText('dGamma error', dGamma1, dGammaN)} {getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testLayerNormalizationLayerGradient2():
    N, L, C = 32, 16, 24;
    X = np.random.randn(N, L, C);
    gamma, beta = np.random.randn(C), np.random.randn(C);
    m = LayerNormalizationLayer(C, gamma = gamma, beta = beta);
    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y))[0];
    dGamma1, dBeta1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dGammaN = numericGradient(lambda x: np.sum(LayerNormalizationLayer(C, gamma = x, beta = beta).forward(X)[0]), gamma);
    dBetaN = numericGradient(lambda x: np.sum(LayerNormalizationLayer(C, gamma = gamma, beta = x).forward(X)[0]), beta);
    print(f"LayerNormalizationLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)} {getErrorText('dGamma error', dGamma1, dGammaN)} {getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testLayerNormalizationLayerGradient3():
    N, C, H, W = 32, 16, 24, 48;
    X = np.random.randn(N, C, H, W);
    gamma, beta = np.random.randn(C, H, W), np.random.randn(C, H, W);
    m = LayerNormalizationLayer((C, H, W), gamma = gamma, beta = beta);
    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y))[0];
    dGamma1, dBeta1 = m.params[0].grad, m.params[1].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dGammaN = numericGradient(lambda x: np.sum(LayerNormalizationLayer((C, H, W), gamma = x, beta = beta).forward(X)[0]), gamma);
    dBetaN = numericGradient(lambda x: np.sum(LayerNormalizationLayer((C, H, W), gamma = gamma, beta = x).forward(X)[0]), beta);
    print(f"LayerNormalizationLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)} {getErrorText('dGamma error', dGamma1, dGammaN)} {getErrorText('dBeta error', dBeta1, dBetaN)}");
    print("\n");


def testMinMaxLayerGradient1():
    minValue, maxValue = 1, None;
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D);
    m = MinMaxLayer(minValue, maxValue);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MinMaxLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMinMaxLayerGradient2():
    minValue, maxValue = None, 1;
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D);
    m = MinMaxLayer(minValue, maxValue);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MinMaxLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMinMaxLayerGradient3():
    minValue, maxValue = 1, 2;
    N, T, D = 32, 24, 16;
    X = np.random.randn(N, T, D);
    m = MinMaxLayer(minValue, maxValue);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"MinMaxLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testMinMaxLayerGradient4():
    minValue, maxValue = 1, None;
    N, T, D = 32, 24, 16;
    X1 = np.random.randn(N, T, D);
    X2 = np.random.randn(N, T, D);
    m = MinMaxLayer(minValue, maxValue);
    Y1, Y2 = m.forward(X1, X2);
    dX1, dX2 = m.backward(np.ones_like(Y1), np.ones_like(Y2));
    dX1N = numericGradient(lambda x: np.sum(np.add(*m.forward(x, X2))), X1);
    dX2N = numericGradient(lambda x: np.sum(np.add(*m.forward(X1, x))), X2);
    print(f"MinMaxLayer, numericGradient4 {getErrorText('dX1 error', dX1, dX1N)} {getErrorText('dX2 error', dX2, dX2N)}");
    print("\n");


def testMinMaxLayerGradient5():
    minValue, maxValue = None, 1;
    N, T, D = 32, 24, 16;
    X1 = np.random.randn(N, T, D);
    X2 = np.random.randn(N, T, D);
    m = MinMaxLayer(minValue, maxValue);
    Y1, Y2 = m.forward(X1, X2);
    dX1, dX2 = m.backward(np.ones_like(Y1), np.ones_like(Y2));
    dX1N = numericGradient(lambda x: np.sum(np.add(*m.forward(x, X2))), X1);
    dX2N = numericGradient(lambda x: np.sum(np.add(*m.forward(X1, x))), X2);
    print(f"MinMaxLayer, numericGradient5 {getErrorText('dX1 error', dX1, dX1N)} {getErrorText('dX2 error', dX2, dX2N)}");
    print("\n");


def testMinMaxLayerGradient6():
    minValue, maxValue = 1, 2;
    N, T, D = 32, 24, 16;
    X1 = np.random.randn(N, T, D);
    X2 = np.random.randn(N, T, D);
    m = MinMaxLayer(minValue, maxValue);
    Y1, Y2 = m.forward(X1, X2);
    dX1, dX2 = m.backward(np.ones_like(Y1), np.ones_like(Y2));
    dX1N = numericGradient(lambda x: np.sum(np.add(*m.forward(x, X2))), X1);
    dX2N = numericGradient(lambda x: np.sum(np.add(*m.forward(X1, x))), X2);
    print(f"MinMaxLayer, numericGradient6 {getErrorText('dX1 error', dX1, dX1N)} {getErrorText('dX2 error', dX2, dX2N)}");
    print("\n");


def testEmbeddingLayerGradient1():
    N, T, H, V = 320, 24, 10, 1000;
    Weight = np.random.randn(V, H);
    X = np.random.choice(np.arange(V), (N, T), True);
    m = EmbeddingLayer(V, H, W = Weight);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1 = m.params[0].grad;
    dWN = numericGradient(lambda x: np.sum(*EmbeddingLayer(V, H, W = x).forward(X)), Weight);
    print(f"EmbeddingLayer, numericGradient1, dW error: {np.sum(np.abs(dW1 - dWN))}");
    print("\n");


def testEmbeddingLayerGradient2():
    def createModel(W1 : np.ndarray, W2 : np.ndarray) -> SequentialContainer:
        return SequentialContainer(
            EmbeddingLayer(V, H, W = W1),
            AffineLayer(H, 2 * H, includeBias = False, W = W2),
        );


    N, T, H, V = 320, 24, 10, 1000;
    Weight1 = np.random.randn(V, H);
    Weight2 = np.random.randn(H, 2 * H);
    X = np.random.choice(np.arange(V), (N, T), True);
    m = createModel(Weight1, Weight2);
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dW1 = m.modules[0].params[0].grad;
    dWN = numericGradient(lambda x: np.sum(*createModel(x, Weight2).forward(X)), Weight1);
    print(f"EmbeddingLayer, numericGradient2, dW error: {np.sum(np.abs(dW1 - dWN))}");
    print("\n");


def testEmbeddingWithDotLayerGradient1():
    N, C1, C2, H, V = 32, 24, 12, 10, 1000;
    Weight = np.random.randn(V, H);
    X = np.random.randn(N, C1, C2, H);
    T = np.random.choice(np.arange(V), (N, C1, C2), True);
    m = EmbeddingWithDotLayer(V, H, W = Weight);
    Y = m.forward(X, T)[0];
    dX = m.backward(np.ones_like(Y))[0];
    dW = m.params[0].grad;
    dXN = numericGradient(lambda x: np.sum(*EmbeddingWithDotLayer(V, H, W = Weight).forward(x, T)), X);
    dWN = numericGradient(lambda x: np.sum(*EmbeddingWithDotLayer(V, H, W = x).forward(X, T)), Weight);
    print(f"EmbeddingWithDotLayer, numericGradient1, dX error: {np.sum(np.abs(dX - dXN))}, dW error: {np.sum(np.abs(dW - dWN))}");
    print("\n");


def testEmbeddingWithDotLayerGradient2():
    def createModel(W1 : np.ndarray, W2 : np.ndarray) -> SequentialContainer:
        return SequentialContainer(
            EmbeddingWithDotLayer(V, H, W = W1),
            AffineLayer(C2, 2 * C2, includeBias = False, W = W2),
        );


    N, C1, C2, H, V = 32, 24, 12, 10, 1000;
    Weight1 = np.random.randn(V, H);
    Weight2 = np.random.randn(C2, 2 * C2);
    X = np.random.randn(N, C1, C2, H);
    T = np.random.choice(np.arange(V), (N, C1, C2), True);
    m = createModel(Weight1, Weight2);
    Y = m.forward(X, T)[0];
    dX = m.backward(np.ones_like(Y))[0];
    dW = m.modules[0].params[0].grad;
    dXN = numericGradient(lambda x: np.sum(*createModel(Weight1, Weight2).forward(x, T)), X);
    dWN = numericGradient(lambda x: np.sum(*createModel(x, Weight2).forward(X, T)), Weight1);
    print(f"EmbeddingWithDotLayer, numericGradient2, dX error: {np.sum(np.abs(dX - dXN))}, dW error: {np.sum(np.abs(dW - dWN))}");
    print("\n");


def testAdditiveResidualBlockGradient1():
    N, D = 32, 16;
    X = np.random.randn(N, D);
    m = AdditiveResidualBlock(
        AffineLayer(D, D)
    );
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AdditiveResidualBlock, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "AdditiveResidualBlock, numericGradient1", X);
    print("\n");


def testAdditiveResidualBlockGradient2():
    N, D1, D2 = 32, 16, 24;
    X = np.random.randn(N, D1);
    m = AdditiveResidualBlock(
        AffineLayer(D1, D2),
        AffineLayer(D1, D2),
    );
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AdditiveResidualBlock, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "AdditiveResidualBlock, numericGradient2", X);
    print("\n");


def testAdditiveResidualBlockGradient3():
    N, D1, D2 = 32, 16, 24;
    X = np.random.randn(N, D1);
    m = AdditiveResidualBlock(
        AffineLayer(D1, D2),
        AffineLayer(D1, D2),
        ReluLayer(),
    );
    Y = m.forward(X)[0];
    dX1 = m.backward(np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AdditiveResidualBlock, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "AdditiveResidualBlock, numericGradient3", X);
    print("\n");


def testRepeatedWrapperOfAffineLayerGradient():
    def forward(model : INetModule, X2 : np.ndarray):
        Y2 = np.zeros(X2.shape[:-1] + (outputSize, ));

        model.reset();
        for t2 in range(X.shape[-2]):
            Y2[..., t2, :], = model.forward(X2[..., t2, :]);

        return Y2;


    N1, N2, T, inputSize, outputSize = 64, 32, 11, 16, 24;
    X, W, b = np.random.randn(N1, N2, T, inputSize), np.random.randn(inputSize, outputSize), np.random.randn(outputSize);
    m1 = AffineLayer(inputSize, outputSize, W = W, b = b);
    m2 = RepeatedWrapper(AffineLayer(inputSize, outputSize, W = W, b = b));

    Y1 = m1.forward(X)[0];
    dX1 = m1.backward(np.ones_like(Y1))[0];
    dW1, db1 = m1.grads;

    Y2 = forward(m2, X);
    dX2 = np.zeros_like(X);
    for t in reversed(range(X.shape[-2])):
        dX = m2.backward(np.ones_like(Y2[..., t, :]))[0];
        dX2[..., t, :] = dX;
    dW2, db2 = m2.grads;

    dWN = numericGradient(lambda x: np.sum(forward(RepeatedWrapper(AffineLayer(inputSize, outputSize, W = x, b = b)), X)), W);
    dbN = numericGradient(lambda x: np.sum(forward(RepeatedWrapper(AffineLayer(inputSize, outputSize, W = W, b = x)), X)), b);

    print(f"RepeatedWrapper, batch process, Y error: {np.sum(np.abs(Y1 - Y2))}, dX error: {np.sum(np.abs(dX1 - dX2))}, dW error: {np.sum(np.abs(dW1 - dW2))}, db error: {np.sum(np.abs(db1 - db2))}, dWN error: {np.sum(np.abs(dW2 - dWN))}, dbN error: {np.sum(np.abs(db2 - dbN))}");
    print("\n");


def testRnnCell1():
    N, inputSize, hiddenSize = 32, 100, 64;
    X, H = np.random.randn(N, inputSize), np.random.randn(N, hiddenSize);
    Wx, Wh = np.random.randn(inputSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize);
    bx, bh = np.random.randn(hiddenSize), np.random.randn(hiddenSize);

    Y1 = tanh(X @ Wx + H @ Wh + bx + bh);

    m = RnnCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    Y2, = m.forward(X, H);

    print(f"RnnCell1, value1 {getErrorText('Y error', Y1, Y2)}");
    print("\n");


def testRnnCellGradient1():
    N, inputSize, hiddenSize = 32, 24, 48;
    X, H = np.random.randn(N, inputSize), np.random.randn(N, hiddenSize);
    Wx, Wh = np.random.randn(inputSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize);
    bx, bh = np.random.randn(hiddenSize), np.random.randn(hiddenSize);
    m = RnnCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    Y, = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(Y));
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x, H)[0]), X);
    dHN = numericGradient(lambda x: np.sum(m.forward(X, x)[0]), H);
    dWxN = numericGradient(lambda x: np.sum(RnnCell(inputSize, hiddenSize, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X, H)[0]), Wx);
    dWhN = numericGradient(lambda x: np.sum(RnnCell(inputSize, hiddenSize, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X, H)[0]), Wh);
    dbxN = numericGradient(lambda x: np.sum(RnnCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X, H)[0]), bx);
    dbhN = numericGradient(lambda x: np.sum(RnnCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X, H)[0]), bh);
    print(f"RnnCell, numericGradient1 {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testRnnLayer1():
    T, N, inputSize, hiddenSize = 50, 32, 24, 48;
    Xs, H = np.random.randn(T, N, inputSize), np.zeros((N, hiddenSize));
    Wx, Wh = np.random.randn(inputSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize);
    bx, bh = np.random.randn(hiddenSize), np.random.randn(hiddenSize);

    Y1 = [];
    for t in range(T):
        H = tanh(Xs[t] @ Wx + H @ Wh + bx + bh);
        Y1.append(H);
    Y1 = np.array(Y1);

    m = RnnLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    Y2, = m.forward(Xs);

    print(f"RnnLayer, value1 {getErrorText('Y error', Y1, Y2)}");
    print("\n");


def testRnnLayerGradient1_Sequence():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    Wx, Wh = np.random.randn(inputSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize);
    bx, bh = np.random.randn(hiddenSize), np.random.randn(hiddenSize);
    m = RnnLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y))[0];
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWxN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, stateful = False, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X)[0]), Wx);
    dWhN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X)[0]), Wh);
    dbxN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X)[0]), bx);
    dbhN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X)[0]), bh);
    print(f"RnnLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testRnnLayerGradient2_Sequence():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    Wx, Wh = np.random.randn(inputSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize);
    bx, bh = np.random.randn(hiddenSize), np.random.randn(hiddenSize);
    m = RnnLayer(inputSize, hiddenSize, activationFuncSelector = lambda size: ReluLayer(), stateful = False, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y))[0];
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWxN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, activationFuncSelector = lambda size: ReluLayer(), stateful = False, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X)[0]), Wx);
    dWhN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, activationFuncSelector = lambda size: ReluLayer(), stateful = False, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X)[0]), Wh);
    dbxN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, activationFuncSelector = lambda size: ReluLayer(), stateful = False, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X)[0]), bx);
    dbhN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, activationFuncSelector = lambda size: ReluLayer(), stateful = False, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X)[0]), bh);
    print(f"RnnLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testRnnLayerGradient3_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    Wx, Wh = np.random.randn(inputSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize);
    bx, bh = np.random.randn(hiddenSize), np.random.randn(hiddenSize);
    m = RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y))[0];
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWxN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X)[0]), Wx);
    dWhN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X)[0]), Wh);
    dbxN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X)[0]), bx);
    dbhN = numericGradient(lambda x: np.sum(RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X)[0]), bh);
    print(f"RnnLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testRnnLayerGradient4_Sequence_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    Wx, Wh = np.random.randn(inputSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize);
    bx, bh = np.random.randn(hiddenSize), np.random.randn(hiddenSize);
    m = RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, S = m.forward(X);
    dX1 = m.backward(np.ones_like(Y), np.ones_like(S))[0];
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    dWxN = numericGradient(lambda x: sumAll(*RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X)), Wh);
    dbxN = numericGradient(lambda x: sumAll(*RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X)), bx);
    dbhN = numericGradient(lambda x: sumAll(*RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X)), bh);
    print(f"RnnLayer, numericGradient4, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testRnnLayerGradient5_Foreign_Sequence_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X, H = np.random.randn(T, N, inputSize), np.random.randn(N, hiddenSize);
    beta = np.random.randn(hiddenSize);
    Wx, Wh = np.random.randn(inputSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize);
    bx, bh = np.random.randn(hiddenSize), np.random.randn(hiddenSize);
    m = RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = bh, activationFuncSelector = lambda size: SwishLayer(beta = beta, outputSize = size));
    Y, S = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(Y), np.ones_like(S));
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x)), H);
    dWxN = numericGradient(lambda x: sumAll(*RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = x, Wh = Wh, bx = bx, bh = bh, activationFuncSelector = lambda size: SwishLayer(beta = beta, outputSize = size)).forward(X, H)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = x, bx = bx, bh = bh, activationFuncSelector = lambda size: SwishLayer(beta = beta, outputSize = size)).forward(X, H)), Wh);
    dbxN = numericGradient(lambda x: sumAll(*RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = x, bh = bh, activationFuncSelector = lambda size: SwishLayer(beta = beta, outputSize = size)).forward(X, H)), bx);
    dbhN = numericGradient(lambda x: sumAll(*RnnLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = x, activationFuncSelector = lambda size: SwishLayer(beta = beta, outputSize = size)).forward(X, H)), bh);
    print(f"RnnLayer, numericGradient5 {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testGruCell1():
    N, inputSize, hiddenSize = 32, 100, 64;
    X, H = np.random.randn(N, inputSize), np.random.randn(N, hiddenSize);
    Wxr, Wxz, Wxh = np.random.randn(inputSize, hiddenSize), np.random.randn(inputSize, hiddenSize), np.random.randn(inputSize, hiddenSize);
    Whr, Whz, Whh = np.random.randn(hiddenSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize);
    bxr, bxz, bxh = np.random.randn(hiddenSize), np.random.randn(hiddenSize), np.random.randn(hiddenSize);
    bhr, bhz, bhh = np.random.randn(hiddenSize), np.random.randn(hiddenSize), np.random.randn(hiddenSize);

    R = sigmoid(X @ Wxr + H @ Whr + bxr + bhr);
    Z = sigmoid(X @ Wxz + H @ Whz + bxz + bhz);
    A = tanh(X @ Wxh + (R * H) @ Whh + bxh + bhh);
    Y1 = Z * H + (1 - Z) * A;

    m = GruCell(inputSize, hiddenSize,
                Wx = np.concatenate((Wxr, Wxz, Wxh), axis = -1), Wh = np.concatenate((Whr, Whz, Whh), axis = -1),
                bx = np.concatenate((bxr, bxz, bxh), axis = -1), bh = np.concatenate((bhr, bhz, bhh), axis = -1));
    Y2, = m.forward(X, H);

    print(f"GruCell1, value1 {getErrorText('Y error', Y1, Y2)}");
    print("\n");


def testGruCellGradient1():
    N, inputSize, hiddenSize = 32, 100, 64;
    X, H = np.random.randn(N, inputSize), np.random.randn(N, hiddenSize);
    Wx, Wh = np.random.randn(inputSize, 3 * hiddenSize), np.random.randn(hiddenSize, 3 * hiddenSize);
    bx, bh = np.random.randn(3 * hiddenSize), np.random.randn(3 * hiddenSize);
    m = GruCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    Y, = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(Y));
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x, H)[0]), X);
    dHN = numericGradient(lambda x: np.sum(m.forward(X, x)[0]), H);
    dWxN = numericGradient(lambda x: np.sum(GruCell(inputSize, hiddenSize, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X, H)[0]), Wx);
    dWhN = numericGradient(lambda x: np.sum(GruCell(inputSize, hiddenSize, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X, H)[0]), Wh);
    dbxN = numericGradient(lambda x: np.sum(GruCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X, H)[0]), bx);
    dbhN = numericGradient(lambda x: np.sum(GruCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X, H)[0]), bh);
    print(f"GruCell, numericGradient1 {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testGruLayer1():
    N, stepSize, inputSize, hiddenSize = 32, 100, 24, 48;
    Xs, H = np.random.randn(stepSize, N, inputSize), np.zeros((N, hiddenSize));
    Wx, Wh = np.random.randn(inputSize, 3 * hiddenSize), np.random.randn(hiddenSize, 3 * hiddenSize);
    bx, bh = np.random.randn(3 * hiddenSize), np.random.randn(3 * hiddenSize);
    dYs = np.random.randn(stepSize, N, hiddenSize);

    cells = [GruCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = bx, bh = bh) for _ in range(stepSize)];
    layer = GruLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = bx, bh = bh);

    Y1 = [];
    for X, cell in zip(Xs, cells):
        H,  = cell.forward(X, H);
        Y1.append(H);
    Y1 = np.array(Y1);

    dXs1, dH1 = [], np.zeros_like(H);
    dWx1, dWh1, dbx1, dbh1 = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(bx), np.zeros_like(bh);
    for t in reversed(range(stepSize)):
        dY, cell = dYs[t], cells[t];
        dX, dH1 = cell.backward(dY + dH1);

        dXs1.append(dX);
        dWx1 += cell.params[0].grad;
        dWh1 += cell.params[1].grad;
        dbx1 += cell.params[2].grad;
        dbh1 += cell.params[3].grad;
    dXs1.reverse();
    dXs1 = np.array(dXs1);

    Y2, = layer.forward(Xs);
    dXs2, = layer.backward(dYs);
    dH2 = layer.dH;
    dWx2 = layer.params[0].grad;
    dWh2 = layer.params[1].grad;
    dbx2 = layer.params[2].grad;
    dbh2 = layer.params[3].grad;

    print(f"GruLayer, value1 {getErrorText('Y error', Y1, Y2)} {getErrorText('dX error', dXs1, dXs2)} {getErrorText('dH error', dH1, dH2)} {getErrorText('dWx error', dWx1, dWx2)} {getErrorText('dWh error', dWh1, dWh2)} {getErrorText('dbx error', dbx1, dbx2)} {getErrorText('dbh error', dbh1, dbh2)}");

    print("\n");


def testGruLayerGradient1_Sequence():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    Wx, Wh = np.random.randn(inputSize, 3 * hiddenSize), np.random.randn(hiddenSize, 3 * hiddenSize);
    bx, bh = np.random.randn(3 * hiddenSize), np.random.randn(3 * hiddenSize);
    m = GruLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y))[0];
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWxN = numericGradient(lambda x: np.sum(GruLayer(inputSize, hiddenSize, stateful = False, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X)[0]), Wx);
    dWhN = numericGradient(lambda x: np.sum(GruLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X)[0]), Wh);
    dbxN = numericGradient(lambda x: np.sum(GruLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X)[0]), bx);
    dbhN = numericGradient(lambda x: np.sum(GruLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X)[0]), bh);
    print(f"GruLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testGruLayerGradient2_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    Wx, Wh = np.random.randn(inputSize, 3 * hiddenSize), np.random.randn(hiddenSize, 3 * hiddenSize);
    bx, bh = np.random.randn(3 * hiddenSize), np.random.randn(3 * hiddenSize);
    m = GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y))[0];
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWxN = numericGradient(lambda x: np.sum(GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X)[0]), Wx);
    dWhN = numericGradient(lambda x: np.sum(GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X)[0]), Wh);
    dbxN = numericGradient(lambda x: np.sum(GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X)[0]), bx);
    dbhN = numericGradient(lambda x: np.sum(GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X)[0]), bh);
    print(f"GruLayer, numericGradient2, State {getErrorText('dX error', dX1, dXN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testGruLayerGradient3_Sequence_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    Wx, Wh = np.random.randn(inputSize, 3 * hiddenSize), np.random.randn(hiddenSize, 3 * hiddenSize);
    bx, bh = np.random.randn(3 * hiddenSize), np.random.randn(3 * hiddenSize);
    m = GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, S = m.forward(X);
    dX1 = m.backward(np.ones_like(Y), np.ones_like(S))[0];
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    dWxN = numericGradient(lambda x: sumAll(*GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X)), Wh);
    dbxN = numericGradient(lambda x: sumAll(*GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X)), bx);
    dbhN = numericGradient(lambda x: sumAll(*GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X)), bh);
    print(f"GruLayer, numericGradient3, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testGruLayerGradient4_Foreign_Sequence_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X, H = np.random.randn(T, N, inputSize), np.random.randn(N, hiddenSize);
    Wx, Wh = np.random.randn(inputSize, 3 * hiddenSize), np.random.randn(hiddenSize, 3 * hiddenSize);
    bx, bh = np.random.randn(3 * hiddenSize), np.random.randn(3 * hiddenSize);
    m = GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, S = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(Y), np.ones_like(S));
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x)), H);
    dWxN = numericGradient(lambda x: sumAll(*GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X, H)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X, H)), Wh);
    dbxN = numericGradient(lambda x: sumAll(*GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X, H)), bx);
    dbhN = numericGradient(lambda x: sumAll(*GruLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X, H)), bh);
    print(f"GruLayer, numericGradient4, Foreign, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testLstmCell1():
    N, inputSize, hiddenSize = 32, 100, 64;
    X, H, C = np.random.randn(N, inputSize), np.random.randn(N, hiddenSize), np.random.randn(N, hiddenSize);
    Wxf, Wxi, Wxo, Wxh = np.random.randn(inputSize, hiddenSize), np.random.randn(inputSize, hiddenSize), np.random.randn(inputSize, hiddenSize), np.random.randn(inputSize, hiddenSize);
    Whf, Whi, Who, Whh = np.random.randn(hiddenSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize), np.random.randn(hiddenSize, hiddenSize);
    bxf, bxi, bxo, bxh = np.random.randn(hiddenSize), np.random.randn(hiddenSize), np.random.randn(hiddenSize), np.random.randn(hiddenSize);
    bhf, bhi, bho, bhh = np.random.randn(hiddenSize), np.random.randn(hiddenSize), np.random.randn(hiddenSize), np.random.randn(hiddenSize);

    F = sigmoid(X @ Wxf + H @ Whf + bxf + bhf);
    I = sigmoid(X @ Wxi + H @ Whi + bxi + bhi);
    O = sigmoid(X @ Wxo + H @ Who + bxo + bho);
    S = tanh(X @ Wxh + H @ Whh + bxh + bhh);
    YC1 = F * C + I * S;
    YH1 = O * tanh(YC1);

    m = LstmCell(inputSize, hiddenSize,
                 Wx = np.concatenate((Wxf, Wxi, Wxo, Wxh), axis = -1), Wh = np.concatenate((Whf, Whi, Who, Whh), axis = -1),
                 bx = np.concatenate((bxf, bxi, bxo, bxh), axis = -1), bh = np.concatenate((bhf, bhi, bho, bhh), axis = -1));
    YH2, YC2 = m.forward(X, H, C);

    print(f"LstmCell, value1 {getErrorText('H error', YH1, YH2)} {getErrorText('C error', YC1, YC2)}");
    print("\n");


def testLstmCellGradient1():
    N, inputSize, hiddenSize = 32, 24, 48;
    X, H, C = np.random.randn(N, inputSize), np.random.randn(N, hiddenSize), np.random.randn(N, hiddenSize);
    Wx, Wh, bx, bh = np.random.randn(inputSize, 4 * hiddenSize), np.random.randn(hiddenSize, 4 * hiddenSize), np.random.randn(4 * hiddenSize), np.random.randn(4 * hiddenSize);
    m = LstmCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    YH, YC = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(YH), np.ones_like(YC));
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    dWxN = numericGradient(lambda x: sumAll(*LstmCell(inputSize, hiddenSize, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X, H, C)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*LstmCell(inputSize, hiddenSize, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X, H, C)), Wh);
    dbxN = numericGradient(lambda x: sumAll(*LstmCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X, H, C)), bx);
    dbhN = numericGradient(lambda x: sumAll(*LstmCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X, H, C)), bh);
    print(f"LstmCell, numericGradient1 {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


# def testLstmCellGradient2():
#     N, inputSize, outputSize = 32, 12, 16;
#     X, H, C = np.random.randn(N, inputSize), np.random.randn(N, outputSize), np.random.randn(N, outputSize);
#     Wx, Wh, b = np.random.randn(inputSize, 4 * outputSize), np.random.randn(outputSize, 4 * outputSize), np.random.randn(4 * outputSize);
#     m = LstmCell(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b);
#     YH, YC = m.forward(X, H, C);
#     dX1, dH1, dC1 = m.backward(np.ones_like(YH), np.zeros_like(YC));
#     dWx1, dWh1, db1 = tuple(m.grads);
#     dXN = numericGradient(lambda x: np.sum(m.forward(x, H, C)[0]), X);
#     dHN = numericGradient(lambda x: np.sum(m.forward(X, x, C)[0]), H);
#     dCN = numericGradient(lambda x: np.sum(m.forward(X, H, x)[0]), C);
#     dWxN = numericGradient(lambda x: np.sum(LstmCell(inputSize, outputSize, Wx = x, Wh = Wh, b = b).forward(X, H, C)[0]), Wx);
#     dWhN = numericGradient(lambda x: np.sum(LstmCell(inputSize, outputSize, Wx = Wx, Wh = x, b = b).forward(X, H, C)[0]), Wh);
#     dbN = numericGradient(lambda x: np.sum(LstmCell(inputSize, outputSize, Wx = Wx, Wh = Wh, b = x).forward(X, H, C)[0]), b);
#     print(f"LstmCell, numericGradient, dX error: {np.sum(np.abs(dX1 - dXN))}, dH error: {np.sum(np.abs(dH1 - dHN))}, dC error: {np.sum(np.abs(dC1 - dCN))}, dWx error: {np.sum(np.abs(dWx1 - dWxN))}, dWh error: {np.sum(np.abs(dWh1 - dWhN))}, dbN error: {np.sum(np.abs(db1 - dbN))}");
#     print("\n");
# 
# 
# def testLstmCellGradient_Dropout():
#     def newCell(Wx2, Wh2, b2):
#         cell = LstmCell(inputSize, outputSize, Wx = Wx2, Wh = Wh2, b = b2, inputDropout = inputDropout, recurrentDropout = recurrentDropout);
#         cell.context.isTrainingMode = True;
#         cell.setInputDropoutMask(inputMask);
#         cell.setRecurrentDropoutMask(recurrentMask);
#         return cell;
# 
# 
#     N, inputSize, outputSize = 32, 12, 16;
#     inputDropout, recurrentDropout = 0.5, 0.5;
#     X, H, C = np.random.randn(N, inputSize), np.random.randn(N, outputSize), np.random.randn(N, outputSize);
#     Wx, Wh, b = np.random.randn(inputSize, 4 * outputSize), np.random.randn(outputSize, 4 * outputSize), np.random.randn(4 * outputSize);
#     inputMask = getDropoutMask(H, inputDropout);
#     recurrentMask = getDropoutMask(C, recurrentDropout);
#     m = LstmCell(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, inputDropout = inputDropout, recurrentDropout = recurrentDropout);
#     m.context.isTrainingMode = True;
#     m.setInputDropoutMask(inputMask);
#     m.setRecurrentDropoutMask(recurrentMask);
#     YH, YC = m.forward(X, H, C);
#     dX1, dH1, dC1 = m.backward(np.ones_like(YH), np.ones_like(YC));
#     dWx1, dWh1, db1 = tuple(m.grads);
#     dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
#     dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
#     dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
#     dWxN = numericGradient(lambda x: sumAll(*newCell(x, Wh, b).forward(X, H, C)), Wx);
#     dWhN = numericGradient(lambda x: sumAll(*newCell(Wx, x, b).forward(X, H, C)), Wh);
#     dbN = numericGradient(lambda x: sumAll(*newCell(Wx, Wh, x).forward(X, H, C)), b);
#     print(f"LstmCell, numericGradient dropout, dX error: {np.sum(np.abs(dX1 - dXN))}, dH error: {np.sum(np.abs(dH1 - dHN))}, dC error: {np.sum(np.abs(dC1 - dCN))}, dWx error: {np.sum(np.abs(dWx1 - dWxN))}, dWh error: {np.sum(np.abs(dWh1 - dWhN))}, dbN error: {np.sum(np.abs(db1 - dbN))}");
#     print("\n");


def testLstmLayer1():
    N, stepSize, inputSize, hiddenSize = 32, 100, 24, 48;
    Xs, H, C = np.random.randn(stepSize, N, inputSize), np.zeros((N, hiddenSize)), np.zeros((N, hiddenSize));
    Wx, Wh = np.random.randn(inputSize, 4 * hiddenSize), np.random.randn(hiddenSize, 4 * hiddenSize);
    bx, bh = np.random.randn(4 * hiddenSize), np.random.randn(4 * hiddenSize);
    dYs, dH, dC = np.random.randn(stepSize, N, hiddenSize), np.random.randn(N, hiddenSize), np.random.randn(N, hiddenSize);

    cells = [LstmCell(inputSize, hiddenSize, Wx = Wx, Wh = Wh, bx = bx, bh = bh) for _ in range(stepSize)];
    layer = LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = bh);

    Y1 = [];
    H1 = H;
    C1 = C;
    for X, cell in zip(Xs, cells):
        H1, C1  = cell.forward(X, H1, C1);
        Y1.append(H1);
    Y1 = np.array(Y1);

    dXs1, dH1, dC1 = [], dH, dC;
    dWx1, dWh1, dbx1, dbh1 = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(bx), np.zeros_like(bh);
    for t in reversed(range(stepSize)):
        dY, cell = dYs[t], cells[t];
        dX, dH1, dC1 = cell.backward(dY + dH1, dC1);

        dXs1.append(dX);
        dWx1 += cell.params[0].grad;
        dWh1 += cell.params[1].grad;
        dbx1 += cell.params[2].grad;
        dbh1 += cell.params[3].grad;
    dXs1.reverse();
    dXs1 = np.array(dXs1);

    Y2, H2, C2 = layer.forward(Xs);
    dXs2, = layer.backward(dYs, dH, dC);
    dH2 = layer.dH;
    dC2 = layer.dC;
    dWx2 = layer.params[0].grad;
    dWh2 = layer.params[1].grad;
    dbx2 = layer.params[2].grad;
    dbh2 = layer.params[3].grad;

    print(f"LstmLayer, value1 {getErrorText('Y error', Y1, Y2)} {getErrorText('H error', H1, H2)} {getErrorText('C error', C1, C2)} {getErrorText('dX error', dXs1, dXs2)} {getErrorText('dH error', dH1, dH2)} {getErrorText('dC error', dC1, dC2)} {getErrorText('dWx error', dWx1, dWx2)} {getErrorText('dWh error', dWh1, dWh2)} {getErrorText('dbx error', dbx1, dbx2)} {getErrorText('dbh error', dbh1, dbh2)}");

    print("\n");


def testLstmLayerGradient1_Sequence():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    Wx, Wh = np.random.randn(inputSize, 4 * hiddenSize), np.random.randn(hiddenSize, 4 * hiddenSize);
    bx, bh = np.random.randn(4 * hiddenSize), np.random.randn(4 * hiddenSize);
    m = LstmLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y))[0];
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    dWxN = numericGradient(lambda x: np.sum(LstmLayer(inputSize, hiddenSize, stateful = False, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X)[0]), Wx);
    dWhN = numericGradient(lambda x: np.sum(LstmLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X)[0]), Wh);
    dbxN = numericGradient(lambda x: np.sum(LstmLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X)[0]), bx);
    dbhN = numericGradient(lambda x: np.sum(LstmLayer(inputSize, hiddenSize, stateful = False, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X)[0]), bh);
    print(f"LstmLayer, numericGradient1, Sequence {getErrorText('dX error', dX1, dXN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testLstmLayerGradient2_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    Wx, Wh = np.random.randn(inputSize, 4 * hiddenSize), np.random.randn(hiddenSize, 4 * hiddenSize);
    bx, bh = np.random.randn(4 * hiddenSize), np.random.randn(4 * hiddenSize);
    m = LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    H, C = m.forward(X);
    dX1 = m.backward(np.ones_like(H), np.ones_like(C))[0];
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    dWxN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X)), Wh);
    dbxN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X)), bx);
    dbhN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = False, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X)), bh);
    print(f"LstmLayer, numericGradient2, State {getErrorText('dX error', dX1, dXN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testLstmLayerGradient3_Sequence_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    Wx, Wh = np.random.randn(inputSize, 4 * hiddenSize), np.random.randn(hiddenSize, 4 * hiddenSize);
    bx, bh = np.random.randn(4 * hiddenSize), np.random.randn(4 * hiddenSize);
    m = LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, H, C = m.forward(X);
    dX1 = m.backward(np.ones_like(Y), np.ones_like(H), np.ones_like(C))[0];
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    dWxN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X)), Wh);
    dbxN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X)), bx);
    dbhN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X)), bh);
    print(f"LstmLayer, numericGradient3, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testLstmLayerGradient4_Foreign_Sequence_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(N, hiddenSize), np.random.randn(N, hiddenSize);
    Wx, Wh = np.random.randn(inputSize, 4 * hiddenSize), np.random.randn(hiddenSize, 4 * hiddenSize);
    bx, bh = np.random.randn(4 * hiddenSize), np.random.randn(4 * hiddenSize);
    m = LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = bh);
    m.context.isTrainingMode = True;
    Y, SH, SC = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(Y), np.ones_like(SH), np.ones_like(SC));
    dWx1, dWh1 = m.params[0].grad, m.params[1].grad;
    dbx1, dbh1 = m.params[2].grad, m.params[3].grad;
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    dWxN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = x, Wh = Wh, bx = bx, bh = bh).forward(X, H, C)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = x, bx = bx, bh = bh).forward(X, H, C)), Wh);
    dbxN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = x, bh = bh).forward(X, H, C)), bx);
    dbhN = numericGradient(lambda x: sumAll(*LstmLayer(inputSize, hiddenSize, stateful = False, returnSequence = True, returnState = True, Wx = Wx, Wh = Wh, bx = bx, bh = x).forward(X, H, C)), bh);
    print(f"LstmLayer, numericGradient4, Foreign, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)} {getErrorText('dWx error', dWx1, dWxN)} {getErrorText('dWh error', dWh1, dWhN)} {getErrorText('dbx error', dbx1, dbxN)} {getErrorText('dbh error', dbh1, dbhN)}");
    print("\n");


def testLstmLayerGradient_State_Dropout(returnSequences = False):
    def newLayer(Wx2 : np.ndarray, Wh2 : np.ndarray, b2 : np.ndarray):
        layer = LstmLayer(inputSize, outputSize, Wx = Wx2, Wh = Wh2, b = b2, returnSequences = returnSequences, returnState = True, stateful = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout);
        layer.context.isTrainingMode = True;
        layer.setInputDropoutMask(inputMask);
        layer.setRecurrentDropoutMask(recurrentMask);
        return layer;


    N, T, inputSize, outputSize = 32, 10, 12, 16;
    X, H, C = np.random.randn(N, T, inputSize), np.random.randn(N, outputSize), np.random.randn(N, outputSize);
    inputDropout, recurrentDropout = 0.5, 0.5;
    inputMask = getDropoutMask(H, inputDropout);
    recurrentMask = getDropoutMask(C, recurrentDropout);
    Wx, Wh, b = np.random.randn(inputSize, 4 * outputSize), np.random.randn(outputSize, 4 * outputSize), np.random.randn(4 * outputSize);

    m = LstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, returnSequences = returnSequences, returnState = True, stateful = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout);
    m.context.isTrainingMode = True;
    m.setInputDropoutMask(inputMask);
    m.setRecurrentDropoutMask(recurrentMask);
    Y, OH, OC = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(Y), np.ones_like(OH), np.ones_like(OC));
    dWx1, dWh1, db1 = tuple(m.grads);
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    dWxN = numericGradient(lambda x: sumAll(*newLayer(x, Wh, b).forward(X, H, C)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*newLayer(Wx, x, b).forward(X, H, C)), Wh);
    dbN = numericGradient(lambda x: sumAll(*newLayer(Wx, Wh, x).forward(X, H, C)), b);
    print(f"LstmLayer, numericGradient state dropout, dX error: {np.sum(np.abs(dX1 - dXN))}, dH error: {np.sum(np.abs(dH1 - dHN))}, dC error: {np.sum(np.abs(dC1 - dCN))}, dWx error: {np.sum(np.abs(dWx1 - dWxN))}, dWh error: {np.sum(np.abs(dWh1 - dWhN))}, db error: {np.sum(np.abs(db1 - dbN))}");
    print("\n");


def testLstmLayerGradient_Stepwise(returnSequences = False):
    def forward(model : INetModule, X2 : np.ndarray, H2 : np.ndarray, C2 : np.ndarray):
        Y2 = np.zeros((N, T, outputSize));

        model.context.isTrainingMode = True;
        model.resetStepState();
        model.setState(H2, C2);
        for t2 in range(T):
            Y2[:, t2], = model.forward(X2[:, t2]);

        return (Y2, ) if returnSequences else (Y2[:, -1], );


    N, T, inputSize, outputSize = 32, 10, 12, 16;
    X, H, C = np.random.randn(N, T, inputSize), np.random.randn(N, outputSize), np.random.randn(N, outputSize);
    Wx, Wh, b = np.random.randn(inputSize, 4 * outputSize), np.random.randn(outputSize, 4 * outputSize), np.random.randn(4 * outputSize);
    m1 = LstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, returnSequences = returnSequences, stateful = True);
    m1.context.isTrainingMode = True;
    Y1, = m1.forward(X, H, C);
    dX1, dH1, dC1 = m1.backward(np.ones_like(Y1));
    m2 = LstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, returnSequences = returnSequences, stateful = True, stepwise = True);
    m2.context.isTrainingMode = True;
    Y2, = forward(m2, X, H, C);
    dX2 = np.zeros_like(X);
    for t in reversed(range(T)):
        if returnSequences:
            dX2[:, t], = m2.backward(np.ones_like(Y2[:, t]));
        else:
            dX2[:, t], = m2.backward(np.ones_like(Y2) if t == T - 1 else np.zeros_like(Y2));
    dH2, dC2 = m2.dH, m2.dC;
    dWx1, dWh1, db1 = tuple(m2.grads);
    dXN = numericGradient(lambda x: sumAll(*forward(m2, x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*forward(m2, X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*forward(m2, X, H, x)), C);
    dWxN = numericGradient(lambda x: sumAll(*forward(LstmLayer(inputSize, outputSize, Wx = x, Wh = Wh, b = b, returnSequences = returnSequences, stateful = True, stepwise = True), X, H, C)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*forward(LstmLayer(inputSize, outputSize, Wx = Wx, Wh = x, b = b, returnSequences = returnSequences, stateful = True, stepwise = True), X, H, C)), Wh);
    dbN = numericGradient(lambda x: sumAll(*forward(LstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = x, returnSequences = returnSequences, stateful = True, stepwise = True), X, H, C)), b);
    print(f"LstmLayer, stepwise, numericGradient, Y error: {np.sum(np.abs(Y1 - Y2))}, dX error: {np.sum(np.abs(dX1 - dX2))}, dH error: {np.sum(np.abs(dH1 - dH2))}, dC error: {np.sum(np.abs(dC1 - dC2))}, dXN error: {np.sum(np.abs(dX2 - dXN))}, dHN error: {np.sum(np.abs(dH2 - dHN))}, dCN error: {np.sum(np.abs(dC2 - dCN))}, dWx error: {np.sum(np.abs(dWx1 - dWxN))}, dWh error: {np.sum(np.abs(dWh1 - dWhN))}, db error: {np.sum(np.abs(db1 - dbN))}");
    print("\n");


def testLstmLayerGradient_Stepwise_State(returnSequences = False):
    def forward(model : INetModule, X2 : np.ndarray, H2 : np.ndarray, C2 : np.ndarray):
        Y2 = np.zeros((N, T, outputSize));

        model.context.isTrainingMode = True;
        model.resetStepState();
        for t2 in range(T):
            Y2[:, t2], H2, C2 = model.forward(X2[:, t2], H2, C2);

        return (Y2, H2, C2) if returnSequences else (Y2[:, -1], H2, C2);


    N, T, inputSize, outputSize = 32, 10, 12, 16;
    X, H, C = np.random.randn(N, T, inputSize), np.random.randn(N, outputSize), np.random.randn(N, outputSize);
    Wx, Wh, b = np.random.randn(inputSize, 4 * outputSize), np.random.randn(outputSize, 4 * outputSize), np.random.randn(4 * outputSize);
    m1 = LstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, returnSequences = returnSequences, returnState = True, stateful = True);
    m1.context.isTrainingMode = True;
    Y1, H1, C1 = m1.forward(X, H, C);
    dX1, dH1, dC1 = m1.backward(np.ones_like(Y1), np.ones_like(H1), np.ones_like(C1));
    m2 = LstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True);
    m2.context.isTrainingMode = True;
    Y2, H2, C2 = forward(m2, X, H, C);
    dX2, dH2, dC2 = np.zeros_like(X), np.ones_like(H2), np.ones_like(C2);
    for t in reversed(range(T)):
        if returnSequences:
            dX2[:, t], dH2, dC2 = m2.backward(np.ones_like(Y2[:, t]), dH2, dC2);
        else:
            dX2[:, t], dH2, dC2 = m2.backward(np.ones_like(Y2) if t == T - 1 else np.zeros_like(Y2), dH2, dC2);
    dWx1, dWh1, db1 = tuple(m2.grads);
    dXN = numericGradient(lambda x: sumAll(*forward(m2, x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*forward(m2, X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*forward(m2, X, H, x)), C);
    dWxN = numericGradient(lambda x: sumAll(*forward(LstmLayer(inputSize, outputSize, Wx = x, Wh = Wh, b = b, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True), X, H, C)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*forward(LstmLayer(inputSize, outputSize, Wx = Wx, Wh = x, b = b, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True), X, H, C)), Wh);
    dbN = numericGradient(lambda x: sumAll(*forward(LstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = x, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True), X, H, C)), b);
    print(f"LstmLayer, stepwise state, numericGradient, Y error: {np.sum(np.abs(Y1 - Y2))}, H error: {np.sum(np.abs(H1 - H2))}, C error: {np.sum(np.abs(C1 - C2))}, dX error: {np.sum(np.abs(dX1 - dX2))}, dH error: {np.sum(np.abs(dH1 - dH2))}, dC error: {np.sum(np.abs(dC1 - dC2))}, dXN error: {np.sum(np.abs(dX2 - dXN))}, dHN error: {np.sum(np.abs(dH2 - dHN))}, dCN error: {np.sum(np.abs(dC2 - dCN))}, dWx error: {np.sum(np.abs(dWx1 - dWxN))}, dWh error: {np.sum(np.abs(dWh1 - dWhN))}, db error: {np.sum(np.abs(db1 - dbN))}");
    print("\n");


def testLstmLayerGradient_Stepwise_State_Dropout(returnSequences = False):
    def forward(model : INetModule, X2 : np.ndarray, H2 : np.ndarray, C2 : np.ndarray):
        model.context.isTrainingMode = True;
        model.setInputDropoutMask(inputMask);
        model.setRecurrentDropoutMask(recurrentMask);
        model.resetStepState();

        Y2 = np.zeros((N, T, outputSize));
        for t2 in range(T):
            Y2[:, t2], H2, C2 = model.forward(X2[:, t2], H2, C2);

        return (Y2, H2, C2) if returnSequences else (Y2[:, -1], H2, C2);


    N, T, inputSize, outputSize = 32, 10, 12, 16;
    X, H, C = np.random.randn(N, T, inputSize), np.random.randn(N, outputSize), np.random.randn(N, outputSize);
    Wx, Wh, b = np.random.randn(inputSize, 4 * outputSize), np.random.randn(outputSize, 4 * outputSize), np.random.randn(4 * outputSize);
    inputDropout, recurrentDropout = 0.5, 0.5;
    inputMask = getDropoutMask(H, inputDropout);
    recurrentMask = getDropoutMask(C, recurrentDropout);

    m1 = LstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, returnSequences = returnSequences, returnState = True, stateful = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout);
    m1.context.isTrainingMode = True;
    m1.setInputDropoutMask(inputMask);
    m1.setRecurrentDropoutMask(recurrentMask);
    Y1, H1, C1 = m1.forward(X, H, C);
    dX1, dH1, dC1 = m1.backward(np.ones_like(Y1), np.ones_like(H1), np.ones_like(C1));
    m2 = LstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout);
    m2.context.isTrainingMode = True;
    m2.setInputDropoutMask(inputMask);
    m2.setRecurrentDropoutMask(recurrentMask);
    Y2, H2, C2 = forward(m2, X, H, C);
    dX2, dH2, dC2 = np.zeros_like(X), np.ones_like(H2), np.ones_like(C2);
    for t in reversed(range(T)):
        if returnSequences:
            dX2[:, t], dH2, dC2 = m2.backward(np.ones_like(Y2[:, t]), dH2, dC2);
        else:
            dX2[:, t], dH2, dC2 = m2.backward(np.ones_like(Y2) if t == T - 1 else np.zeros_like(Y2), dH2, dC2);
    dWx1, dWh1, db1 = tuple(m2.grads);
    dXN = numericGradient(lambda x: sumAll(*forward(m2, x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*forward(m2, X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*forward(m2, X, H, x)), C);
    dWxN = numericGradient(lambda x: sumAll(*forward(LstmLayer(inputSize, outputSize, Wx = x, Wh = Wh, b = b, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout), X, H, C)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*forward(LstmLayer(inputSize, outputSize, Wx = Wx, Wh = x, b = b, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout), X, H, C)), Wh);
    dbN = numericGradient(lambda x: sumAll(*forward(LstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = x, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout), X, H, C)), b);
    print(f"LstmLayer, stepwise state dropout, numericGradient, Y error: {np.sum(np.abs(Y1 - Y2))}, H error: {np.sum(np.abs(H1 - H2))}, C error: {np.sum(np.abs(C1 - C2))}, dX error: {np.sum(np.abs(dX1 - dX2))}, dH error: {np.sum(np.abs(dH1 - dH2))}, dC error: {np.sum(np.abs(dC1 - dC2))}, dXN error: {np.sum(np.abs(dX2 - dXN))}, dHN error: {np.sum(np.abs(dH2 - dHN))}, dCN error: {np.sum(np.abs(dC2 - dCN))}, dWx error: {np.sum(np.abs(dWx1 - dWxN))}, dWh error: {np.sum(np.abs(dWh1 - dWhN))}, db error: {np.sum(np.abs(db1 - dbN))}");
    print("\n");


def testBahdanauAttentionLstmLayerGradient(returnSequences = False):
    def forward(model : INetModule, X2 : np.ndarray, K2 : np.ndarray):
        return model.forward(X2, K2, K2[:, -1], None);


    N, T1, T2, inputSize, outputSize = 32, 11, 12, 22, 24;
    X, K = np.random.randn(N, T2, inputSize), np.random.randn(N, T1, outputSize);
    m = BahdanauAttentionLstmLayer(inputSize, outputSize, returnSequences = returnSequences, stateful = True);
    Wx, Wh, b, Wq, Wk, wv = tuple(m.params);

    Y, = forward(m, X, K);
    dX1, dK1, dH, dC = m.backward(np.ones_like(Y));
    dK1[:, -1] += dH;
    dWx1, dWh1, db1, dWq1, dWk1, dwv1 = tuple(m.grads);
    dXN = numericGradient(lambda x: sumAll(*forward(m, x, K)), X);
    dKN = numericGradient(lambda x: sumAll(*forward(m, X, x)), K);
    dWxN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = x, Wh = Wh, b = b, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, stateful = True), X, K)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = x, b = b, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, stateful = True), X, K)), Wh);
    dbN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = x, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, stateful = True), X, K)), b);
    dWqN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = x, Wk = Wk, wv = wv, returnSequences = returnSequences, stateful = True), X, K)), Wq);
    dWkN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = Wq, Wk = x, wv = wv, returnSequences = returnSequences, stateful = True), X, K)), Wk);
    dwvN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = Wq, Wk = Wk, wv = x, returnSequences = returnSequences, stateful = True), X, K)), wv);
    print(f"BahdanauAttentionLstmLayer, numericGradient, dX error: {np.sum(np.abs(dX1 - dXN))}, dK error: {np.sum(np.abs(dK1 - dKN))}, dWx error: {np.sum(np.abs(dWx1 - dWxN))}, dWh error: {np.sum(np.abs(dWh1 - dWhN))}, db error: {np.sum(np.abs(db1 - dbN))}, dWq error: {np.sum(np.abs(dWq1 - dWqN))}, dWk error: {np.sum(np.abs(dWk1 - dWkN))}, dwv error: {np.sum(np.abs(dwv1 - dwvN))}");
    print("\n");


def testBahdanauAttentionLstmLayerGradient_Stepwise(returnSequences = False):
    def forward(model : INetModule, X2 : np.ndarray, K2 : np.ndarray):
        Y2 = np.zeros((N, T2, outputSize));

        model.resetStepState();
        model.setState(K2[:, -1]);
        for t2 in range(T2):
            Y2[:, t2], = model.forward(X2[:, t2], K2);

        return (Y2, ) if returnSequences else (Y2[:, -1], );


    N, T1, T2, inputSize, outputSize = 32, 11, 12, 22, 24;
    X, K = np.random.randn(N, T2, inputSize), np.random.randn(N, T1, outputSize);
    m1 = BahdanauAttentionLstmLayer(inputSize, outputSize, returnSequences = returnSequences, stateful = True);
    Y1, = m1.forward(X, K, K[:, -1], None);
    dX1, dK1, dH, dC = m1.backward(np.ones_like(Y1));
    dK1[:, -1] += dH;
    Wx, Wh, b, Wq, Wk, wv = tuple(m1.params);
    m2 = BahdanauAttentionLstmLayer(inputSize, outputSize, returnSequences = returnSequences, stateful = True, stepwise = True);
    m2.params = m1.params;
    Y2, = forward(m2, X, K);
    dX2, dK2 = np.zeros_like(X), np.zeros_like(K);
    for t in reversed(range(T2)):
        if returnSequences:
            dX2[:, t], dK = m2.backward(np.ones_like(Y2[:, t]));
        else:
            dX2[:, t], dK = m2.backward(np.ones_like(Y2) if t == T2 - 1 else np.zeros_like(Y2));
        dK2 += dK;
    dK2[:, -1] += m2.dH;
    dWx1, dWh1, db1, dWq1, dWk1, dwv1 = tuple(m2.grads);
    dXN = numericGradient(lambda x: sumAll(*forward(m2, x, K)), X);
    dKN = numericGradient(lambda x: sumAll(*forward(m2, X, x)), K);
    dWxN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = x, Wh = Wh, b = b, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, stateful = True, stepwise = True,), X, K)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = x, b = b, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, stateful = True, stepwise = True), X, K)), Wh);
    dbN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = x, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, stateful = True, stepwise = True), X, K)), b);
    dWqN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = x, Wk = Wk, wv = wv, returnSequences = returnSequences, stateful = True, stepwise = True), X, K)), Wq);
    dWkN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = Wq, Wk = x, wv = wv, returnSequences = returnSequences, stateful = True, stepwise = True), X, K)), Wk);
    dwvN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = Wq, Wk = Wk, wv = x, returnSequences = returnSequences, stateful = True, stepwise = True), X, K)), wv);
    print(f"BahdanauAttentionLstmLayer, stepwise, numericGradient, Y error: {np.sum(np.abs(Y1 - Y2))}, dX error: {np.sum(np.abs(dX1 - dX2))}, dK error: {np.sum(np.abs(dK1 - dK2))}, dXN error: {np.sum(np.abs(dX2 - dXN))}, dKN error: {np.sum(np.abs(dK2 - dKN))}, dWx error: {np.sum(np.abs(dWx1 - dWxN))}, dWh error: {np.sum(np.abs(dWh1 - dWhN))}, db error: {np.sum(np.abs(db1 - dbN))}, dWq error: {np.sum(np.abs(dWq1 - dWqN))}, dWk error: {np.sum(np.abs(dWk1 - dWkN))}, dwv error: {np.sum(np.abs(dwv1 - dwvN))}");
    print("\n");


def testBahdanauAttentionLstmLayerGradient_Stepwise_State(returnSequences = False):
    def forward(model : INetModule, X2 : np.ndarray, K2 : np.ndarray):
        Y2 = np.zeros((N, T2, outputSize));
        H2, C2 = K2[:, -1], None;

        model.resetStepState();
        for t2 in range(T2):
            Y2[:, t2], H2, C2 = model.forward(X2[:, t2], K2, H2, C2);

        return (Y2, H2, C2) if returnSequences else (Y2[:, -1], H2, C2);


    N, T1, T2, inputSize, outputSize = 32, 11, 12, 22, 24;
    X, K = np.random.randn(N, T2, inputSize), np.random.randn(N, T1, outputSize);
    m1 = BahdanauAttentionLstmLayer(inputSize, outputSize, returnSequences = returnSequences, returnState = True, stateful = True);
    Y1, H1, C1 = m1.forward(X, K, K[:, -1], None);
    dX1, dK1, dH1, dC1 = m1.backward(np.ones_like(Y1), np.ones_like(H1), np.ones_like(C1));
    dK1[:, -1] += dH1;
    Wx, Wh, b, Wq, Wk, wv = tuple(m1.params);
    m2 = BahdanauAttentionLstmLayer(inputSize, outputSize, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True);
    m2.params = m1.params;
    Y2, H2, C2 = forward(m2, X, K);
    dX2, dK2, dH2, dC2 = np.zeros_like(X), np.zeros_like(K), np.ones_like(H2), np.ones_like(C2);
    for t in reversed(range(T2)):
        if returnSequences:
            dX2[:, t], dK, dH2, dC2 = m2.backward(np.ones_like(Y2[:, t]), dH2, dC2);
        else:
            dX2[:, t], dK, dH2, dC2 = m2.backward(np.ones_like(Y2) if t == T2 - 1 else np.zeros_like(Y2), dH2, dC2);
        dK2 += dK;
    dK2[:, -1] += dH2;
    dWx1, dWh1, db1, dWq1, dWk1, dwv1 = tuple(m2.grads);
    dXN = numericGradient(lambda x: sumAll(*forward(m2, x, K)), X);
    dKN = numericGradient(lambda x: sumAll(*forward(m2, X, x)), K);
    dWxN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = x, Wh = Wh, b = b, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True,), X, K)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = x, b = b, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True), X, K)), Wh);
    dbN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = x, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True), X, K)), b);
    dWqN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = x, Wk = Wk, wv = wv, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True), X, K)), Wq);
    dWkN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = Wq, Wk = x, wv = wv, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True), X, K)), Wk);
    dwvN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = Wq, Wk = Wk, wv = x, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True), X, K)), wv);
    print(f"BahdanauAttentionLstmLayer, stepwise state, numericGradient, Y error: {np.sum(np.abs(Y1 - Y2))}, H error: {np.sum(np.abs(H1 - H2))}, C error: {np.sum(np.abs(C1 - C2))}, dX error: {np.sum(np.abs(dX1 - dX2))}, dK error: {np.sum(np.abs(dK1 - dK2))}, dH error: {np.sum(np.abs(dH1 - dH2))}, dC error: {np.sum(np.abs(dC1 - dC2))}, dXN error: {np.sum(np.abs(dX2 - dXN))}, dKN error: {np.sum(np.abs(dK2 - dKN))}, dWx error: {np.sum(np.abs(dWx1 - dWxN))}, dWh error: {np.sum(np.abs(dWh1 - dWhN))}, db error: {np.sum(np.abs(db1 - dbN))}, dWq error: {np.sum(np.abs(dWq1 - dWqN))}, dWk error: {np.sum(np.abs(dWk1 - dWkN))}, dwv error: {np.sum(np.abs(dwv1 - dwvN))}");
    print("\n");


def testBahdanauAttentionLstmLayerGradient_Stepwise_State_Dropout(returnSequences = False):
    def forward(model : INetModule, X2 : np.ndarray, K2 : np.ndarray):
        model.context.isTrainingMode = True;
        model.setInputDropoutMask(inputMask);
        model.setRecurrentDropoutMask(recurrentMask);

        Y2 = np.zeros((N, T2, outputSize));
        H2, C2 = K2[:, -1], None;

        model.resetStepState();
        for t2 in range(T2):
            Y2[:, t2], H2, C2 = model.forward(X2[:, t2], K2, H2, C2);

        return (Y2, H2, C2) if returnSequences else (Y2[:, -1], H2, C2);


    N, T1, T2, inputSize, outputSize = 32, 11, 12, 22, 24;
    X, K = np.random.randn(N, T2, inputSize), np.random.randn(N, T1, outputSize);
    inputDropout, recurrentDropout = 0.5, 0.5;
    inputMask = getDropoutMask(np.ones_like(K[:, -1]), inputDropout);
    recurrentMask = getDropoutMask(np.ones_like(K[:, -1]), recurrentDropout);
    m1 = BahdanauAttentionLstmLayer(inputSize, outputSize, returnSequences = returnSequences, returnState = True, stateful = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout);
    m1.context.isTrainingMode = True;
    m1.setInputDropoutMask(inputMask);
    m1.setRecurrentDropoutMask(recurrentMask);
    Y1, H1, C1 = m1.forward(X, K, K[:, -1], None);
    dX1, dK1, dH1, dC1 = m1.backward(np.ones_like(Y1), np.ones_like(H1), np.ones_like(C1));
    dK1[:, -1] += dH1;
    Wx, Wh, b, Wq, Wk, wv = tuple(m1.params);
    m2 = BahdanauAttentionLstmLayer(inputSize, outputSize, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout);
    m2.params = m1.params;
    m2.context.isTrainingMode = True;
    m2.setInputDropoutMask(inputMask);
    m2.setRecurrentDropoutMask(recurrentMask);
    Y2, H2, C2 = forward(m2, X, K);
    dX2, dK2, dH2, dC2 = np.zeros_like(X), np.zeros_like(K), np.ones_like(H2), np.ones_like(C2);
    for t in reversed(range(T2)):
        if returnSequences:
            dX2[:, t], dK, dH2, dC2 = m2.backward(np.ones_like(Y2[:, t]), dH2, dC2);
        else:
            dX2[:, t], dK, dH2, dC2 = m2.backward(np.ones_like(Y2) if t == T2 - 1 else np.zeros_like(Y2), dH2, dC2);
        dK2 += dK;
    dK2[:, -1] += dH2;
    dWx1, dWh1, db1, dWq1, dWk1, dwv1 = tuple(m2.grads);
    dXN = numericGradient(lambda x: sumAll(*forward(m2, x, K)), X);
    dKN = numericGradient(lambda x: sumAll(*forward(m2, X, x)), K);
    dWxN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = x, Wh = Wh, b = b, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout), X, K)), Wx);
    dWhN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = x, b = b, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout), X, K)), Wh);
    dbN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = x, Wq = Wq, Wk = Wk, wv = wv, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout), X, K)), b);
    dWqN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = x, Wk = Wk, wv = wv, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout), X, K)), Wq);
    dWkN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = Wq, Wk = x, wv = wv, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout), X, K)), Wk);
    dwvN = numericGradient(lambda x: sumAll(*forward(BahdanauAttentionLstmLayer(inputSize, outputSize, Wx = Wx, Wh = Wh, b = b, Wq = Wq, Wk = Wk, wv = x, returnSequences = returnSequences, returnState = True, stateful = True, stepwise = True, inputDropout = inputDropout, recurrentDropout = recurrentDropout), X, K)), wv);
    print(f"BahdanauAttentionLstmLayer, stepwise state dropout, numericGradient, Y error: {np.sum(np.abs(Y1 - Y2))}, H error: {np.sum(np.abs(H1 - H2))}, C error: {np.sum(np.abs(C1 - C2))}, dX error: {np.sum(np.abs(dX1 - dX2))}, dK error: {np.sum(np.abs(dK1 - dK2))}, dH error: {np.sum(np.abs(dH1 - dH2))}, dC error: {np.sum(np.abs(dC1 - dC2))}, dXN error: {np.sum(np.abs(dX2 - dXN))}, dKN error: {np.sum(np.abs(dK2 - dKN))}, dWx error: {np.sum(np.abs(dWx1 - dWxN))}, dWh error: {np.sum(np.abs(dWh1 - dWhN))}, db error: {np.sum(np.abs(db1 - dbN))}, dWq error: {np.sum(np.abs(dWq1 - dWqN))}, dWk error: {np.sum(np.abs(dWk1 - dWkN))}, dwv error: {np.sum(np.abs(dwv1 - dwvN))}");
    print("\n");


def testBiRnnLayerGradient1_Gru_InnerState_Sequence():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = BiRnnLayer(inputSize, hiddenSize, GruLayer, stateful = False, returnSequence = True, returnState = False);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"BiRnnLayer, numericGradient1, Gru, InnerState, Sequence {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient1, Gru, InnerState, Sequence", X);
    print("\n");


def testBiRnnLayerGradient2_Gru_InnerState_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = BiRnnLayer(inputSize, hiddenSize, GruLayer, stateful = False, returnSequence = False, returnState = True);
    m.context.isTrainingMode = True;
    S, = m.forward(X);
    dX1, = m.backward(np.ones_like(S));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"BiRnnLayer, numericGradient2, Gru, InnerState, State {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient2, Gru, InnerState, State", X);
    print("\n");


def testBiRnnLayerGradient3_Gru_InnerState_Sequence_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = BiRnnLayer(inputSize, hiddenSize, GruLayer, stateful = False, returnSequence = True, returnState = True);
    m.context.isTrainingMode = True;
    Y, S = m.forward(X);
    dX1, = m.backward(np.ones_like(Y), np.ones_like(S));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"BiRnnLayer, numericGradient3, Gru, InnerState, Sequence and State {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient3, Gru, InnerState, Sequence and State", X);
    print("\n");


def testBiRnnLayerGradient4_Gru_ForeignState_Sequence():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X, H = np.random.randn(T, N, inputSize), np.random.randn(N, 2 * hiddenSize);
    m = BiRnnLayer(inputSize, hiddenSize, GruLayer, stateful = False, returnSequence = True, returnState = False);
    m.context.isTrainingMode = True;
    Y, = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x)), H);
    print(f"BiRnnLayer, numericGradient4, Gru, ForeignState, Sequence {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient4, Gru, ForeignState, Sequence", X, H);
    print("\n");


def testBiRnnLayerGradient5_Gru_ForeignState_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X, H = np.random.randn(T, N, inputSize), np.random.randn(N, 2 * hiddenSize);
    m = BiRnnLayer(inputSize, hiddenSize, GruLayer, stateful = False, returnSequence = False, returnState = True);
    m.context.isTrainingMode = True;
    S, = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(S));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x)), H);
    print(f"BiRnnLayer, numericGradient5, Gru, ForeignState, State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient5, Gru, ForeignState, State", X, H);
    print("\n");


def testBiRnnLayerGradient6_Gru_ForeignState_Sequence_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(N, 2 * hiddenSize), np.random.randn(T, N, 2 * hiddenSize);
    CY, CH = np.random.randn(T, N, 2 * hiddenSize), np.random.randn(N, 2 * hiddenSize);
    m = SequentialContainer(
        BiRnnLayer(inputSize, hiddenSize, GruLayer, stateful = False, returnSequence = True, returnState = True),
        FunctionalNetModule("*C", lambda x: x * CH if x.ndim == 2 else x * CY, lambda x, y, dy: dy * CH if dy.ndim == 2 else dy * CY),
    );
    m.context.isTrainingMode = True;
    Y, S = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(Y), np.ones_like(S));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x)), H);
    print(f"BiRnnLayer, numericGradient6, Gru, ForeignState, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient6, Gru, ForeignState, Sequence and State", X, H);
    print("\n");


def testBiRnnLayerGradient7_LstmLayer_InnerState_Sequence():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = BiRnnLayer(inputSize, hiddenSize, LstmLayer, stateful = False, returnSequence = True, returnState = False);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"BiRnnLayer, numericGradient7, Lstm, InnerState, Sequence {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient7, Lstm, InnerState, Sequence", X);
    print("\n");


def testBiRnnLayerGradient8_LstmLayer_InnerState_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = BiRnnLayer(inputSize, hiddenSize, LstmLayer, stateful = False, returnSequence = False, returnState = True);
    m.context.isTrainingMode = True;
    S, C = m.forward(X);
    dX1, = m.backward(np.ones_like(S), np.ones_like(C));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"BiRnnLayer, numericGradient8, Lstm, InnerState, State {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient8, Lstm, InnerState, State", X);
    print("\n");


def testBiRnnLayerGradient9_LstmLayer_InnerState_Sequence_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = BiRnnLayer(inputSize, hiddenSize, LstmLayer, stateful = False, returnSequence = True, returnState = True);
    m.context.isTrainingMode = True;
    Y, S, C = m.forward(X);
    dX1, = m.backward(np.ones_like(Y), np.ones_like(S), np.ones_like(C));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"BiRnnLayer, numericGradient9, Lstm, InnerState, Sequence and State {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient9, Lstm, InnerState, Sequence and State", X);
    print("\n");


def testBiRnnLayerGradient10_LstmLayer_ForeignState_Sequence():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(N, 2 * hiddenSize), np.random.randn(N, 2 * hiddenSize);
    m = BiRnnLayer(inputSize, hiddenSize, LstmLayer, stateful = False, returnSequence = True, returnState = False);
    m.context.isTrainingMode = True;
    Y, = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    print(f"BiRnnLayer, numericGradient10, Lstm, ForeignState, Sequence {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient10, Lstm, ForeignState, Sequence", X, H, C);
    print("\n");


def testBiRnnLayerGradient11_LstmLayer_ForeignState_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(N, 2 * hiddenSize), np.random.randn(N, 2 * hiddenSize);
    m = BiRnnLayer(inputSize, hiddenSize, LstmLayer, stateful = False, returnSequence = False, returnState = True);
    m.context.isTrainingMode = True;
    OS, OC, = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(OS), np.ones_like(OC));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    print(f"BiRnnLayer, numericGradient11, Lstm, ForeignState, State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient11, Lstm, ForeignState, State", X, H, C);
    print("\n");


def testBiRnnLayerGradient12_LstmLayer_ForeignState_Sequence_State():
    T, N, inputSize, hiddenSize = 4, 32, 24, 48;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(N, 2 * hiddenSize), np.random.randn(N, 2 * hiddenSize);
    CY, CH = np.random.randn(T, N, 2 * hiddenSize), np.random.randn(N, 2 * hiddenSize);
    m = SequentialContainer(
        BiRnnLayer(inputSize, hiddenSize, LstmLayer, stateful = False, returnSequence = True, returnState = True),
        FunctionalNetModule("*C", lambda x: x * CH if x.ndim == 2 else x * CY, lambda x, y, dy: dy * CH if dy.ndim == 2 else dy * CY),
    );
    m.context.isTrainingMode = True;
    Y, OS, OC = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(Y), np.ones_like(OS), np.ones_like(OC));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    print(f"BiRnnLayer, numericGradient12, Lstm, ForeignState, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)}");
    testModuleGradient(m, "BiRnnLayer, numericGradient12, Lstm, ForeignState, Sequence and State", X, H, C);
    print("\n");


def testStackRnnLayerGradient1_Gru_InnerState_Sequence():
    L, T, N, inputSize, hiddenSize = 2, 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = StackRnnLayer(inputSize, hiddenSize, GruLayer, layersNum = L, stateful = False, returnSequence = True, returnState = False);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"StackRnnLayer, numericGradient1, Gru, InnerState, Sequence {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient1, Gru, InnerState, Sequence", X);
    print("\n");


def testStackRnnLayerGradient2_Gru_InnerState_State():
    L, T, N, inputSize, hiddenSize = 2, 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = StackRnnLayer(inputSize, hiddenSize, GruLayer, layersNum = L, stateful = False, returnSequence = False, returnState = True);
    m.context.isTrainingMode = True;
    S, = m.forward(X);
    dX1, = m.backward(np.ones_like(S));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"StackRnnLayer, numericGradient2, Gru, InnerState, State {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient2, Gru, InnerState, State", X);
    print("\n");


def testStackRnnLayerGradient3_Gru_InnerState_Sequence_State():
    L, T, N, inputSize, hiddenSize = 2, 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = StackRnnLayer(inputSize, hiddenSize, GruLayer, layersNum = L, stateful = False, returnSequence = True, returnState = True);
    m.context.isTrainingMode = True;
    Y, S = m.forward(X);
    dX1, = m.backward(np.ones_like(Y), np.ones_like(S));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"StackRnnLayer, numericGradient3, Gru, InnerState, Sequence and State {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient3, Gru, InnerState, Sequence and State", X);
    print("\n");


def testStackRnnLayerGradient4_Gru_ForeignState_Sequence():
    L, T, N, inputSize, hiddenSize = 2, 4, 32, 24, 48;
    X, H = np.random.randn(T, N, inputSize), np.random.randn(L, N, hiddenSize);
    m = StackRnnLayer(inputSize, hiddenSize, GruLayer, layersNum = L, stateful = False, returnSequence = True, returnState = False);
    m.context.isTrainingMode = True;
    Y, = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x)), H);
    print(f"StackRnnLayer, numericGradient4, Gru, ForeignState, Sequence {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient4, Gru, ForeignState, Sequence", X, H);
    print("\n");


def testStackRnnLayerGradient5_Gru_ForeignState_State():
    L, T, N, inputSize, hiddenSize = 2, 4, 32, 24, 48;
    X, H = np.random.randn(T, N, inputSize), np.random.randn(L, N, hiddenSize);
    m = StackRnnLayer(inputSize, hiddenSize, GruLayer, layersNum = L, stateful = False, returnSequence = False, returnState = True);
    m.context.isTrainingMode = True;
    S, = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(S));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x)), H);
    print(f"StackRnnLayer, numericGradient5, Gru, ForeignState, State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient5, Gru, ForeignState, State", X, H);
    print("\n");


def testStackRnnLayerGradient6_Gru_ForeignState_Sequence_State():
    L, T, N, inputSize, hiddenSize = 3, 4, 32, 24, 48;
    X, H = np.random.randn(T, N, inputSize), np.random.randn(L, N, hiddenSize);
    CY, CH = np.random.randn(T, N, hiddenSize), np.random.randn(L, N, hiddenSize);
    m = SequentialContainer(
        StackRnnLayer(inputSize, hiddenSize, GruLayer, layersNum = L, stateful = False, returnSequence = True, returnState = True),
        FunctionalNetModule("*C", lambda x: x * CH if len(x) == L else x * CY, lambda x, y, dy: dy * CH if len(dy) == L else dy * CY),
    );
    m.context.isTrainingMode = True;
    Y, S = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(Y), np.ones_like(S));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x)), H);
    print(f"StackRnnLayer, numericGradient6, Gru, ForeignState, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient6, Gru, ForeignState, Sequence and State", X, H);
    print("\n");


def testStackRnnLayerGradient7_Lstm_InnerState_Sequence():
    L, T, N, inputSize, hiddenSize = 2, 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = StackRnnLayer(inputSize, hiddenSize, LstmLayer, layersNum = L, stateful = False, returnSequence = True, returnState = False);
    m.context.isTrainingMode = True;
    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"StackRnnLayer, numericGradient7, Lstm, InnerState, Sequence {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient7, Lstm, InnerState, Sequence", X);
    print("\n");


def testStackRnnLayerGradient8_Lstm_InnerState_State():
    L, T, N, inputSize, hiddenSize = 2, 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = StackRnnLayer(inputSize, hiddenSize, LstmLayer, layersNum = L, stateful = False, returnSequence = False, returnState = True);
    m.context.isTrainingMode = True;
    S, C = m.forward(X);
    dX1, = m.backward(np.ones_like(S), np.ones_like(C));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"StackRnnLayer, numericGradient8, Lstm, InnerState, State {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient8, Lstm, InnerState, State", X);
    print("\n");


def testStackRnnLayerGradient9_Lstm_InnerState_Sequence_State():
    L, T, N, inputSize, hiddenSize = 2, 4, 32, 24, 48;
    X = np.random.randn(T, N, inputSize);
    m = StackRnnLayer(inputSize, hiddenSize, LstmLayer, layersNum = L, stateful = False, returnSequence = True, returnState = True);
    m.context.isTrainingMode = True;
    Y, S, C = m.forward(X);
    dX1, = m.backward(np.ones_like(Y), np.ones_like(S), np.ones_like(C));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x)), X);
    print(f"StackRnnLayer, numericGradient9, Lstm, InnerState, Sequence and State {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient9, Lstm, InnerState, Sequence and State", X);
    print("\n");


def testStackRnnLayerGradient10_Lstm_ForeignState_Sequence():
    L, T, N, inputSize, hiddenSize = 2, 4, 32, 24, 48;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(L, N, hiddenSize), np.random.randn(L, N, hiddenSize);
    m = StackRnnLayer(inputSize, hiddenSize, LstmLayer, layersNum = L, stateful = False, returnSequence = True, returnState = False);
    m.context.isTrainingMode = True;
    Y, = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    print(f"StackRnnLayer, numericGradient10, Lstm, ForeignState, Sequence {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient10, Lstm, ForeignState, Sequence", X, H, C);
    print("\n");


def testStackRnnLayerGradient11_Lstm_ForeignState_State():
    L, T, N, inputSize, hiddenSize = 2, 4, 32, 24, 48;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(L, N, hiddenSize), np.random.randn(L, N, hiddenSize);
    m = StackRnnLayer(inputSize, hiddenSize, LstmLayer, layersNum = L, stateful = False, returnSequence = False, returnState = True);
    m.context.isTrainingMode = True;
    OH, OC = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(OH), np.ones_like(OC));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    print(f"StackRnnLayer, numericGradient11, Lstm, ForeignState, State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient11, Lstm, ForeignState, State", X, H, C);
    print("\n");


def testStackRnnLayerGradient12_Lstm_ForeignState_Sequence_State():
    L, T, N, inputSize, hiddenSize = 3, 4, 32, 24, 48;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(L, N, hiddenSize), np.random.randn(L, N, hiddenSize);
    CY, CH = np.random.randn(T, N, hiddenSize), np.random.randn(L, N, hiddenSize);
    m = SequentialContainer(
        StackRnnLayer(inputSize, hiddenSize, LstmLayer, layersNum = L, stateful = False, returnSequence = True, returnState = True),
        FunctionalNetModule("*C", lambda x: x * CH if len(x) == L else x * CY, lambda x, y, dy: dy * CH if len(dy) == L else dy * CY),
    );
    m.context.isTrainingMode = True;
    Y, OH, OC = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(Y), np.ones_like(OH), np.ones_like(OC));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    print(f"StackRnnLayer, numericGradient12, Lstm, ForeignState, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient12, Lstm, ForeignState, Sequence and State", X, H, C);
    print("\n");


def testStackRnnLayerGradient13_BiGru_ForeignState_Sequence_State():
    L, T, N, inputSize, hiddenSize = 3, 4, 32, 12, 24;
    X, H = np.random.randn(T, N, inputSize), np.random.randn(L, N, 2 * hiddenSize);
    CY, CH = np.random.randn(T, N, 2 * hiddenSize), np.random.randn(L, N, 2 * hiddenSize);
    m = SequentialContainer(
        StackRnnLayer(inputSize, hiddenSize, GruLayer, layersNum = L, stateful = False, returnSequence = True, returnState = True, bidirectional = True),
        FunctionalNetModule("*C", lambda x: x * CH if len(x) == L else x * CY, lambda x, y, dy: dy * CH if len(dy) == L else dy * CY),
    );
    m.context.isTrainingMode = True;
    Y, S = m.forward(X, H);
    dX1, dH1 = m.backward(np.ones_like(Y), np.ones_like(S));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x)), H);
    print(f"StackRnnLayer, numericGradient13, BiGru, ForeignState, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient13, BiGru, ForeignState, Sequence and State", X, H);
    print("\n");


def testStackRnnLayerGradient14_BiLstm_ForeignState_Sequence_State():
    L, T, N, inputSize, hiddenSize = 3, 4, 32, 12, 24;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(L, N, 2 * hiddenSize), np.random.randn(L, N, 2 * hiddenSize);
    CY, CH = np.random.randn(T, N, 2 * hiddenSize), np.random.randn(L, N, 2 * hiddenSize);
    m = SequentialContainer(
        StackRnnLayer(inputSize, hiddenSize, LstmLayer, layersNum = L, stateful = False, returnSequence = True, returnState = True, bidirectional = True),
        FunctionalNetModule("*C", lambda x: x * CH if len(x) == L else x * CY, lambda x, y, dy: dy * CH if len(dy) == L else dy * CY),
    );
    m.context.isTrainingMode = True;
    Y, OH, OC = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(Y), np.ones_like(OH), np.ones_like(OC));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    print(f"StackRnnLayer, numericGradient14, BiLstm, ForeignState, Sequence and State {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient14, BiLstm, ForeignState, Sequence and State", X, H, C);
    print("\n");


def testStackRnnLayerGradient15_Lstm_ForeignState_Sequence_State_Normal():
    L, T, N, inputSize, hiddenSize = 3, 4, 32, 12, 24;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(L, N, hiddenSize), np.random.randn(L, N, hiddenSize);
    CY, CH = np.random.randn(T, N, hiddenSize), np.random.randn(L, N, hiddenSize);
    m = SequentialContainer(
        StackRnnLayer(inputSize, hiddenSize, LstmLayer, LayerNormalizationLayer, layersNum = L, stateful = False, returnSequence = True, returnState = True, bidirectional = False),
        FunctionalNetModule("*C", lambda x: x * CH if len(x) == L else x * CY, lambda x, y, dy: dy * CH if len(dy) == L else dy * CY),
    );
    m.context.isTrainingMode = True;
    Y, OH, OC = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(Y), np.ones_like(OH), np.ones_like(OC));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    print(f"StackRnnLayer, numericGradient15, Lstm, ForeignState, Sequence and State, Normal {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient15, Lstm, ForeignState, Sequence and State, Normal", X, H, C);
    print("\n");


def testStackRnnLayerGradient16_BiLstm_ForeignState_Sequence_State_Normal():
    L, T, N, inputSize, hiddenSize = 3, 4, 32, 12, 24;
    X, H, C = np.random.randn(T, N, inputSize), np.random.randn(L, N, 2 * hiddenSize), np.random.randn(L, N, 2 * hiddenSize);
    CY, CH = np.random.randn(T, N, 2 * hiddenSize), np.random.randn(L, N, 2 * hiddenSize);
    m = SequentialContainer(
        StackRnnLayer(inputSize, hiddenSize, LstmLayer, LayerNormalizationLayer, layersNum = L, stateful = False, returnSequence = True, returnState = True, bidirectional = True),
        FunctionalNetModule("*C", lambda x: x * CH if len(x) == L else x * CY, lambda x, y, dy: dy * CH if len(dy) == L else dy * CY),
    );
    m.context.isTrainingMode = True;
    Y, OH, OC = m.forward(X, H, C);
    dX1, dH1, dC1 = m.backward(np.ones_like(Y), np.ones_like(OH), np.ones_like(OC));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, H, C)), X);
    dHN = numericGradient(lambda x: sumAll(*m.forward(X, x, C)), H);
    dCN = numericGradient(lambda x: sumAll(*m.forward(X, H, x)), C);
    print(f"StackRnnLayer, numericGradient16, BiLstm, ForeignState, Sequence and State, Normal {getErrorText('dX error', dX1, dXN)} {getErrorText('dH error', dH1, dHN)} {getErrorText('dC error', dC1, dCN)}");
    testModuleGradient(m, "StackRnnLayer, numericGradient16, BiLstm, ForeignState, Sequence and State, Normal", X, H, C);
    print("\n");


# def testStackLstmLayerGradient_State(returnSequences = False):
#     L, N, T, inputSize, outputSize = 1, 32, 10, 12, 16;
#     X, HS, CS = np.random.randn(N, T, inputSize), np.random.randn(L, N, outputSize), np.random.randn(L, N, outputSize);
#     m = StackRnnLayer(inputSize, outputSize, LstmLayer, layerNum = L, returnSequences = returnSequences, returnState = True, stateful = True);
#     Y1, OHS1, OCS1 = m.forward(X, HS, CS);
#     dX1, dHS1, dCS1 = m.backward(np.ones_like(Y1), np.ones_like(OHS1), np.ones_like(OCS1));
#     dXN = numericGradient(lambda x: sumAll(*m.forward(x, HS, CS)), X);
#     dHSN = numericGradient(lambda x: sumAll(*m.forward(X, x, CS)), HS);
#     dCSN = numericGradient(lambda x: sumAll(*m.forward(X, HS, x)), CS);
#     print(f"StackLstmLayer, state, numericGradient, dX error: {np.sum(np.abs(dX1 - dXN))}, dHS error: {np.sum(np.abs(dHS1 - dHSN))}, dCS error: {np.sum(np.abs(dCS1 - dCSN))}");
#     testModuleGradient(m, "StackLstmLayer state, numericGradient", X, HS, CS);
#     print("\n");
#
#
# def testStackLstmLayerGradient_State_Dropout(returnSequences = False):
#     L, N, T, inputSize, outputSize = 1, 32, 10, 12, 16;
#     X, HS, CS = np.random.randn(N, T, inputSize), np.random.randn(L, N, outputSize), np.random.randn(L, N, outputSize);
#     m = StackRnnLayer(inputSize, outputSize, LstmLayer, layerNum = L, returnSequences = returnSequences, returnState = True, stateful = True, inputDropout = 0.5, recurrentDropout = 0.5);
#     m.context.isTrainingMode = True;
#     Y1, OHS1, OCS1 = m.forward(X, HS, CS);
#     dX1, dHS1, dCS1 = m.backward(np.ones_like(Y1), np.ones_like(OHS1), np.ones_like(OCS1));
#     dXN = numericGradient(lambda x: sumAll(*m.forward(x, HS, CS)), X);
#     dHSN = numericGradient(lambda x: sumAll(*m.forward(X, x, CS)), HS);
#     dCSN = numericGradient(lambda x: sumAll(*m.forward(X, HS, x)), CS);
#     print(f"StackLstmLayer, state dropout, numericGradient, dX error: {np.sum(np.abs(dX1 - dXN))}, dHS error: {np.sum(np.abs(dHS1 - dHSN))}, dCS error: {np.sum(np.abs(dCS1 - dCSN))}");
#     testModuleGradient(m, "StackLstmLayer state dropout, numericGradient", X, HS, CS);
#     print("\n");


def testAdditiveAttentionModule1():
    batchSize, queryNum, keyNum = 32, 20, 21;
    querySize, keySize, valueSize, hiddenSize = 22, 23, 24, 25;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    m1 = AdditiveAttentionModule(querySize, keySize, hiddenSize);
    Y1, = m1.forward(Q, K, V);

    m2 = AdditiveAttentionModule(querySize, keySize, hiddenSize);
    m2.params = m1.params;
    Y2, = m2.forward(Q, K, V);
    assert len(m1.params) == len(m2.params);
    for p1, p2 in zip(m1.params, m2.params):
        assert np.sum(p1.value - p2.value) < 1e-6;

    m3 = m1.copy(False);
    Y3, = m3.forward(Q, K, V);
    assert len(m1.params) == len(m3.params);
    for p1, p3 in zip(m1.params, m3.params):
        assert np.sum(p1.value - p3.value) < 1e-6;

    print(f"AdditiveAttentionModule, error1, Y2 error: {np.sum(np.abs(Y1 - Y2))}, Y3 error: {np.sum(np.abs(Y1 - Y3))}");
    print("\n");


def testAdditiveAttentionModuleGradient1():
    batchSize, queryNum, keyNum = 32, 20, 21;
    querySize, keySize, valueSize, hiddenSize = 22, 23, 24, 25;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    m = AdditiveAttentionModule(querySize, keySize, hiddenSize);
    Wq, Wk, wv = tuple([p.value for p in m.params]);

    Y, = m.forward(Q, K, V);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dWq1, dWk1, dwv1 = tuple([p.grad for p in m.params]);
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x)[0]), V);
    dWqN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = x, Wk = Wk, wv = wv).forward(Q, K, V)[0]), Wq);
    dWkN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = x, wv = wv).forward(Q, K, V)[0]), Wk);
    dwvN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = Wk, wv = x).forward(Q, K, V)[0]), wv);
    print(f"AdditiveAttentionModule, numericGradient1 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)} {getErrorText('dWq error', dWq1, dWqN)} {getErrorText('dWk error', dWk1, dWkN)} {getErrorText('dwv error', dwv1, dwvN)}");
    print("\n");


def testAdditiveAttentionModuleGradient2():
    def getLenMask(queryNum : int, keyNum, validLen : np.ndarray) -> np.ndarray:
        if len(validLen.shape) == 1:
            validLen = np.repeat(np.expand_dims(validLen, axis = -1), queryNum, axis = -1);
        validLen = np.expand_dims(validLen, axis = -1);

        return np.arange(keyNum, dtype = np.int32) < validLen;

    batchSize, queryNum, keyNum = 32, 20, 21;
    querySize, keySize, valueSize, hiddenSize = 22, 23, 24, 25;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    M = getLenMask(queryNum, keyNum, np.random.randint(1, keyNum + 1, batchSize));
    m = AdditiveAttentionModule(querySize, keySize, hiddenSize);
    Wq, Wk, wv = tuple([p.value for p in m.params]);

    Y, = m.forward(Q, K, V, M);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dWq1, dWk1, dwv1 = tuple([p.grad for p in m.params]);
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, M)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, M)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, M)[0]), V);
    dWqN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = x, Wk = Wk, wv = wv).forward(Q, K, V, M)[0]), Wq);
    dWkN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = x, wv = wv).forward(Q, K, V, M)[0]), Wk);
    dwvN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = Wk, wv = x).forward(Q, K, V, M)[0]), wv);
    print(f"AdditiveAttentionModule, numericGradient2 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)} {getErrorText('dWq error', dWq1, dWqN)} {getErrorText('dWk error', dWk1, dWkN)} {getErrorText('dwv error', dwv1, dwvN)}");
    print("\n");


def testAdditiveAttentionModuleGradient3():
    def getLenMask(queryNum : int, keyNum, validLen : np.ndarray) -> np.ndarray:
        if len(validLen.shape) == 1:
            validLen = np.repeat(np.expand_dims(validLen, axis = -1), queryNum, axis = -1);
        validLen = np.expand_dims(validLen, axis = -1);

        return np.arange(keyNum, dtype = np.int32) < validLen;

    batchSize, queryNum, keyNum = 32, 20, 21;
    querySize, keySize, valueSize, hiddenSize = 22, 23, 24, 25;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    M = getLenMask(queryNum, keyNum, np.random.randint(1, keyNum + 1, (batchSize, queryNum)));
    m = AdditiveAttentionModule(querySize, keySize, hiddenSize);
    Wq, Wk, wv = tuple([p.value for p in m.params]);

    Y, = m.forward(Q, K, V, M);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dWq1, dWk1, dwv1 = tuple([p.grad for p in m.params]);
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, M)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, M)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, M)[0]), V);
    dWqN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = x, Wk = Wk, wv = wv).forward(Q, K, V, M)[0]), Wq);
    dWkN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = x, wv = wv).forward(Q, K, V, M)[0]), Wk);
    dwvN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = Wk, wv = x).forward(Q, K, V, M)[0]), wv);
    print(f"AdditiveAttentionModule, numericGradient3 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)} {getErrorText('dWq error', dWq1, dWqN)} {getErrorText('dWk error', dWk1, dWkN)} {getErrorText('dwv error', dwv1, dwvN)}");
    print("\n");


def testAdditiveAttentionModuleGradient4():
    batchSize, sequenceNum, queryNum, keyNum = 2, 3, 4, 5;
    querySize, keySize, valueSize, hiddenSize = 6, 7, 8, 9;
    Q = np.random.randn(batchSize, sequenceNum, queryNum, querySize);
    K = np.random.randn(batchSize, sequenceNum, keyNum, keySize);
    V = np.random.randn(batchSize, sequenceNum, keyNum, valueSize);
    M = getAttentionMaskByValidLength(queryNum, keyNum, np.random.randint(1, keyNum + 1, (batchSize, sequenceNum, queryNum)));
    m = AdditiveAttentionModule(querySize, keySize, hiddenSize);
    Wq, Wk, wv = tuple([p.value for p in m.params]);

    Y, = m.forward(Q, K, V, M);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dWq1, dWk1, dwv1 = tuple([p.grad for p in m.params]);
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, M)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, M)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, M)[0]), V);
    dWqN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = x, Wk = Wk, wv = wv).forward(Q, K, V, M)[0]), Wq);
    dWkN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = x, wv = wv).forward(Q, K, V, M)[0]), Wk);
    dwvN = numericGradient(lambda x: np.sum(AdditiveAttentionModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = Wk, wv = x).forward(Q, K, V, M)[0]), wv);
    print(f"AdditiveAttentionModule, numericGradient4 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)} {getErrorText('dWq error', dWq1, dWqN)} {getErrorText('dWk error', dWk1, dWkN)} {getErrorText('dwv error', dwv1, dwvN)}");
    print("\n");


def testDotProductAttentionModuleGradient1():
    batchSize, queryNum, keyNum = 32, 20, 21;
    querySize, keySize, valueSize = 22, 22, 23;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    m = DotProductAttentionModule();

    Y, = m.forward(Q, K, V);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x)[0]), V);
    print(f"DotProductAttentionModule, numericGradient1 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)}");
    print("\n");


def testDotProductAttentionModuleGradient2():
    def getLenMask(queryNum : int, keyNum, validLen : np.ndarray) -> np.ndarray:
        if len(validLen.shape) == 1:
            validLen = np.repeat(np.expand_dims(validLen, axis = -1), queryNum, axis = -1);
        validLen = np.expand_dims(validLen, axis = -1);

        return np.arange(keyNum, dtype = np.int32) < validLen;

    batchSize, queryNum, keyNum = 32, 20, 21;
    querySize, keySize, valueSize = 22, 22, 23;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    M = getLenMask(queryNum, keyNum, np.random.randint(1, keyNum + 1, (batchSize, queryNum)));
    m = DotProductAttentionModule();

    Y, = m.forward(Q, K, V, M);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, M)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, M)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, M)[0]), V);
    print(f"DotProductAttentionModule, numericGradient2 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)}");
    print("\n");


def testDotProductAttentionModuleGradient3():
    def getLenMask(queryNum : int, keyNum, validLen : np.ndarray) -> np.ndarray:
        if len(validLen.shape) == 1:
            validLen = np.repeat(np.expand_dims(validLen, axis = -1), queryNum, axis = -1);
        validLen = np.expand_dims(validLen, axis = -1);

        return np.arange(keyNum, dtype = np.int32) < validLen;

    batchSize, queryNum, keyNum = 32, 20, 21;
    querySize, keySize, valueSize = 22, 22, 23;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    M = getLenMask(queryNum, keyNum, np.random.randint(1, keyNum + 1, (batchSize, queryNum)));
    m = DotProductAttentionModule();

    Y, = m.forward(Q, K, V, M);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, M)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, M)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, M)[0]), V);
    print(f"DotProductAttentionModule, numericGradient3 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)}");
    print("\n");


def testDotProductAttentionModuleGradient4():
    batchSize, sequenceNum, queryNum, keyNum = 2, 3, 4, 5;
    querySize, keySize, valueSize = 6, 6, 7;
    Q = np.random.randn(batchSize, sequenceNum, queryNum, querySize);
    K = np.random.randn(batchSize, sequenceNum, keyNum, keySize);
    V = np.random.randn(batchSize, sequenceNum, keyNum, valueSize);
    M = getAttentionMaskByValidLength(queryNum, keyNum, np.random.randint(1, keyNum + 1, (batchSize, sequenceNum, queryNum)));
    m = DotProductAttentionModule();

    Y, = m.forward(Q, K, V, M);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, M)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, M)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, M)[0]), V);
    print(f"DotProductAttentionModule, numericGradient4 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)}");
    print("\n");


def testMultiHeadAttentionModule1():
    batchSize, headNum, queryNum, keyNum = 32, 8, 16, 17;
    querySize, keySize, valueSize = 21, 22, 23;
    queryHiddenSize, keyHiddenSize, valueHiddenSize, additiveHiddenSize, outputHiddenSize = 24, 25, 26, 33, 44;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    attentionModule = AdditiveAttentionModule(queryHiddenSize, keyHiddenSize, additiveHiddenSize);
    m1 = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum);
    Y1, = m1.forward(Q, K, V);

    attentionModule = AdditiveAttentionModule(queryHiddenSize, keyHiddenSize, additiveHiddenSize);
    m2 = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum);
    m2.params = m1.params;
    Y2, = m2.forward(Q, K, V);
    assert len(m1.params) == len(m2.params);
    for p1, p2 in zip(m1.params, m2.params):
        assert np.sum(p1.value - p2.value) < 1e-6;

    m3 = m1.copy(False);
    Y3, = m3.forward(Q, K, V);
    assert len(m1.params) == len(m3.params);
    for p1, p3 in zip(m1.params, m3.params):
        assert np.sum(p1.value - p3.value) < 1e-6;

    print(f"MultiHeadAttentionModule, error1 {getErrorText('Y2 error', Y1, Y2)} {getErrorText('Y3 error', Y1, Y3)}");
    print("\n");


def testMultiHeadAttentionModule2():
    batchSize, headNum, queryNum, keyNum = 32, 8, 16, 17;
    querySize, keySize, valueSize = 21, 22, 23;
    queryHiddenSize, keyHiddenSize, valueHiddenSize, additiveHiddenSize, outputHiddenSize = 24, 24, 25, 33, 44;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    attentionModule = DotProductAttentionModule();
    m1 = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum);
    Y1, = m1.forward(Q, K, V);

    attentionModule = DotProductAttentionModule();
    m2 = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum);
    m2.params = m1.params;
    Y2, = m2.forward(Q, K, V);
    assert len(m1.params) == len(m2.params);
    for p1, p2 in zip(m1.params, m2.params):
        assert np.sum(p1.value - p2.value) < 1e-6;

    m3 = m1.copy(False);
    Y3, = m3.forward(Q, K, V);
    assert len(m1.params) == len(m3.params);
    for p1, p3 in zip(m1.params, m3.params):
        assert np.sum(p1.value - p3.value) < 1e-6;

    print(f"MultiHeadAttentionModule, error2 {getErrorText('Y2 error', Y1, Y2)} {getErrorText('Y3 error', Y1, Y3)}");
    print("\n");


def testMultiHeadAttentionModule3():
    batchSize, headNum, queryNum, keyNum = 32, 8, 16, 17;
    querySize, keySize, valueSize = 21, 22, 23;
    queryHiddenSize, keyHiddenSize, valueHiddenSize, additiveHiddenSize, outputHiddenSize = 24, 25, 26, 33, 44;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    attentionModule = AdditiveAttentionModule(queryHiddenSize, keyHiddenSize, additiveHiddenSize);

    Wqs, Wks, Wvs, Cs = [], [], [], [];
    for _ in range(headNum):
        Wqs.append(Wq := np.random.randn(querySize, queryHiddenSize));
        Wks.append(Wk := np.random.randn(keySize, keyHiddenSize));
        Wvs.append(Wv := np.random.randn(valueSize, valueHiddenSize));

        QH, KH, VH = Q @ Wq, K @ Wk, V @ Wv;
        Cs.append(attentionModule.forward(QH, KH, VH)[0]);
    Wo = np.random.randn(headNum * valueHiddenSize, outputHiddenSize);
    Y1 = np.concatenate(Cs, axis = -1) @ Wo;

    m = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum,
                                 Wq = np.concatenate(Wqs, axis = -1), Wk = np.concatenate(Wks, axis = -1), Wv = np.concatenate(Wvs, axis = -1), Wo = Wo);
    Y2, = m.forward(Q, K, V);

    print(f"MultiHeadAttentionModule, value3 {getErrorText('Y error', Y1, Y2)}");
    print("\n");


def testMultiHeadAttentionModule4():
    batchSize, headNum, queryNum, keyNum = 32, 8, 16, 17;
    querySize, keySize, valueSize = 21, 22, 23;
    queryHiddenSize, keyHiddenSize, valueHiddenSize, additiveHiddenSize, outputHiddenSize = 24, 25, 26, 33, 44;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    M = np.random.randint(0, 2, (batchSize, queryNum, keyNum));
    attentionModule = AdditiveAttentionModule(queryHiddenSize, keyHiddenSize, additiveHiddenSize);

    Wqs, Wks, Wvs, Cs = [], [], [], [];
    for _ in range(headNum):
        Wqs.append(Wq := np.random.randn(querySize, queryHiddenSize));
        Wks.append(Wk := np.random.randn(keySize, keyHiddenSize));
        Wvs.append(Wv := np.random.randn(valueSize, valueHiddenSize));

        QH, KH, VH = Q @ Wq, K @ Wk, V @ Wv;
        Cs.append(attentionModule.forward(QH, KH, VH, M)[0]);
    Wo = np.random.randn(headNum * valueHiddenSize, outputHiddenSize);
    Y1 = np.concatenate(Cs, axis = -1) @ Wo;

    m = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum,
                                 Wq = np.concatenate(Wqs, axis = -1), Wk = np.concatenate(Wks, axis = -1), Wv = np.concatenate(Wvs, axis = -1), Wo = Wo);
    Y2, = m.forward(Q, K, V, M);

    print(f"MultiHeadAttentionModule, value4 {getErrorText('Y error', Y1, Y2)}");
    print("\n");


def testMultiHeadAttentionModuleGradient1():
    batchSize, headNum, queryNum, keyNum = 6, 8, 16, 17;
    querySize, keySize, valueSize = 21, 22, 23;
    queryHiddenSize, keyHiddenSize, valueHiddenSize, additiveHiddenSize, outputHiddenSize = 24, 25, 26, 33, 44;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    attentionModule = AdditiveAttentionModule(queryHiddenSize, keyHiddenSize, additiveHiddenSize);
    m = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum);

    Y, = m.forward(Q, K, V);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x)[0]), V);
    print(f"MultiHeadAttentionModule, numericGradient1 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)}");
    testModuleGradient(m, "MultiHeadAttentionModule, numericGradient1", Q, K, V);
    print("\n");


def testMultiHeadAttentionModuleGradient2():
    batchSize, headNum, queryNum, keyNum = 6, 8, 16, 17;
    querySize, keySize, valueSize = 21, 22, 23;
    queryHiddenSize, keyHiddenSize, valueHiddenSize, additiveHiddenSize, outputHiddenSize = 24, 25, 26, 33, 44;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    M = np.random.randint(0, 2, (batchSize, queryNum, keyNum));
    attentionModule = AdditiveAttentionModule(queryHiddenSize, keyHiddenSize, additiveHiddenSize);
    m = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum);

    Y, = m.forward(Q, K, V, M);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, M)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, M)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, M)[0]), V);
    print(f"MultiHeadAttentionModule, numericGradient2 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)}");
    testModuleGradient(m, "MultiHeadAttentionModule, numericGradient2", Q, K, V, M);
    print("\n");


def testMultiHeadAttentionModuleGradient3():
    batchSize, headNum, queryNum, keyNum = 6, 8, 16, 17;
    querySize, keySize, valueSize = 21, 22, 23;
    queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize = 24, 24, 24, 33;
    Q = np.random.randn(batchSize, queryNum, querySize);
    K = np.random.randn(batchSize, keyNum, keySize);
    V = np.random.randn(batchSize, keyNum, valueSize);
    M = np.random.randint(0, 2, (batchSize, queryNum, keyNum));
    attentionModule = DotProductAttentionModule();
    m = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum);

    Y, = m.forward(Q, K, V, M);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, M)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, M)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, M)[0]), V);
    print(f"MultiHeadAttentionModule, numericGradient3 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)}");
    testModuleGradient(m, "MultiHeadAttentionModule, numericGradient3", Q, K, V, M);
    print("\n");


def testMultiHeadAttentionModuleGradient4():
    batchSize, sequenceNum, headNum, queryNum, keyNum = 2, 3, 4, 5, 6;
    querySize, keySize, valueSize = 7, 8, 9;
    queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize = 10, 10, 10, 11;
    Q = np.random.randn(batchSize, sequenceNum, queryNum, querySize);
    K = np.random.randn(batchSize, sequenceNum, keyNum, keySize);
    V = np.random.randn(batchSize, sequenceNum, keyNum, valueSize);
    M = getAttentionMaskByValidLength(queryNum, keyNum, np.random.randint(1, keyNum + 1, (batchSize, sequenceNum, queryNum)));
    attentionModule = DotProductAttentionModule();
    m = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum);

    Y, = m.forward(Q, K, V, M);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, M)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, M)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, M)[0]), V);
    print(f"MultiHeadAttentionModule, numericGradient4 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)}");
    testModuleGradient(m, "MultiHeadAttentionModule, numericGradient4", Q, K, V, M);
    print("\n");


def testMultiHeadAttentionModuleGradient5():
    batchSize, sequenceNum, headNum, queryNum, keyNum = 2, 3, 1, 5, 6;
    querySize, keySize, valueSize = 7, 8, 9;
    queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize = 10, 10, 10, 11;
    Q = np.random.randn(batchSize, sequenceNum, queryNum, querySize);
    K = np.random.randn(batchSize, sequenceNum, keyNum, keySize);
    V = np.random.randn(batchSize, sequenceNum, keyNum, valueSize);
    M = getAttentionMaskByValidLength(queryNum, keyNum, np.random.randint(1, keyNum + 1, (batchSize, sequenceNum, queryNum)));
    attentionModule = DotProductAttentionModule();
    m = MultiHeadAttentionModule(attentionModule, querySize, keySize, valueSize, (queryHiddenSize, keyHiddenSize, valueHiddenSize, outputHiddenSize), headNum = headNum);

    Y, = m.forward(Q, K, V, M);
    dQ1, dK1, dV1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, M)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, M)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, M)[0]), V);
    print(f"MultiHeadAttentionModule, numericGradient5 {getErrorText('dQ error', dQ1, dQN)} {getErrorText('dK error', dK1, dKN)} {getErrorText('dV error', dV1, dVN)}");
    testModuleGradient(m, "MultiHeadAttentionModule, numericGradient5", Q, K, V, M);
    print("\n");


def testSelfAttentionModuleGradient1():
    def getLenMask(queryNum : int, keyNum, validLen : np.ndarray) -> np.ndarray:
        if len(validLen.shape) == 1:
            validLen = np.repeat(np.expand_dims(validLen, axis = -1), queryNum, axis = -1);
        validLen = np.expand_dims(validLen, axis = -1);

        return np.arange(keyNum, dtype = np.int32) < validLen;

    batchSize, sequenceLength, sequenceDimension, hiddenSize = 32, 10, 11, 12;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    M = getLenMask(sequenceLength, sequenceLength, np.random.randint(1, sequenceLength + 1, batchSize));
    m = SelfAttentionModule(AdditiveAttentionModule(sequenceDimension, sequenceDimension, hiddenSize));

    Y, = m.forward(X, M);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"SelfAttentionModule, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSelfAttentionModuleGradient2():
    def getLenMask(queryNum: int, keyNum, validLen: np.ndarray) -> np.ndarray:
        if len(validLen.shape) == 1:
            validLen = np.repeat(np.expand_dims(validLen, axis = -1), queryNum, axis = -1);
        validLen = np.expand_dims(validLen, axis = -1);

        return np.arange(keyNum, dtype = np.int32) < validLen;


    batchSize, sequenceLength, sequenceDimension = 32, 10, 11;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    M = getLenMask(sequenceLength, sequenceLength, np.random.randint(1, sequenceLength + 1, (batchSize, sequenceLength)));
    m = SelfAttentionModule(DotProductAttentionModule());

    Y, = m.forward(X, M);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"SelfAttentionModule, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSelfAttentionModuleGradient3():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 11;
    headNum, queryHiddenSize, keyHiddenSize, valueHiddenSize, additiveHiddenSize = 8, 12, 13, 14, 15;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    M = np.random.randint(0, 2, (batchSize, sequenceLength, sequenceLength));
    m = SelfAttentionModule(MultiHeadAttentionModule(AdditiveAttentionModule(queryHiddenSize, keyHiddenSize, additiveHiddenSize), sequenceDimension, sequenceDimension, sequenceDimension, (queryHiddenSize, keyHiddenSize, valueHiddenSize, sequenceDimension), headNum = headNum));

    Y, = m.forward(X, M);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"SelfAttentionModule, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSelfAttentionModuleGradient4():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 11;
    headNum, queryHiddenSize, keyHiddenSize, valueHiddenSize = 8, 12, 12, 12;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    M = np.random.randint(0, 2, (batchSize, sequenceLength, sequenceLength));
    m = SelfAttentionModule(MultiHeadAttentionModule(DotProductAttentionModule(), sequenceDimension, sequenceDimension, sequenceDimension, (queryHiddenSize, keyHiddenSize, valueHiddenSize, sequenceDimension), headNum = headNum));

    Y, = m.forward(X, M);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"SelfAttentionModule, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "SelfAttentionModule, numericGradient4", X, M);
    print("\n");


def testSelfAttentionModuleGradient5():
    batchSize, sequenceNum, sequenceLength, sequenceDimension = 2, 3, 4, 5;
    headNum, queryHiddenSize, keyHiddenSize, valueHiddenSize = 6, 7, 7, 7;
    X = np.random.randn(batchSize, sequenceNum, sequenceLength, sequenceDimension);
    M = getAttentionMaskByValidLength(sequenceLength, sequenceLength, np.random.randint(1, sequenceLength + 1, (batchSize, sequenceNum, sequenceLength)));
    m = SelfAttentionModule(MultiHeadAttentionModule(DotProductAttentionModule(), sequenceDimension, sequenceDimension, sequenceDimension, (queryHiddenSize, keyHiddenSize, valueHiddenSize, sequenceDimension), headNum = headNum));

    Y, = m.forward(X, M);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"SelfAttentionModule, numericGradient5 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "SelfAttentionModule, numericGradient5", X, M);
    print("\n");


def testSinePositionalEncodingModuleGradient1():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    m = SinePositionalEncodingModule(sequenceDimension);

    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"SinePositionalEncodingModule, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSinePositionalEncodingModuleGradient2():
    batchSize, headNum, sequenceLength, sequenceDimension = 32, 8, 10, 21;
    X = np.random.randn(batchSize, headNum, sequenceLength, sequenceDimension);
    m = SinePositionalEncodingModule(sequenceDimension);

    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"SinePositionalEncodingModule, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSinePositionalEncodingModuleGradient3():
    batchSize, headNum, sequenceLength, sequenceDimension = 32, 8, 10, 21;
    X = np.random.randn(batchSize, headNum, sequenceLength, sequenceDimension);
    startIndex = np.array(13);
    m = SinePositionalEncodingModule(sequenceDimension);

    Y, = m.forward(X, startIndex);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, startIndex)[0]), X);
    print(f"SinePositionalEncodingModule, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testSinePositionalEncodingModuleGradient4():
    batchSize, sequenceNum, headNum, sequenceLength, sequenceDimension = 2, 3, 4, 5, 6;
    X = np.random.randn(batchSize, sequenceNum, headNum, sequenceLength, sequenceDimension);
    startIndex = np.array(13);
    m = SinePositionalEncodingModule(sequenceDimension);

    Y, = m.forward(X, startIndex);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, startIndex)[0]), X);
    print(f"SinePositionalEncodingModule, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testTransformerAddNormalizationModuleGradient1():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    F = np.random.randn(*X.shape);
    C = np.random.randn(*X.shape); # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerAddNormalizationModule(sequenceDimension),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, F);
    dX1, dF1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, F)[0]), X);
    dFN = numericGradient(lambda x: np.sum(m.forward(X, x)[0]), F);
    print(f"TransformerAddNormalizationModule, numericGradient1 {getErrorText('dX error', dX1, dXN)}, dF error: {np.sum(np.abs(dF1 - dFN))}");
    testModuleGradient(m, "TransformerAddNormalizationModule, numericGradient1", X, F);
    print("\n");


def testTransformerAddNormalizationModuleGradient2():
    batchSize, sequenceNum, sequenceLength, sequenceDimension = 2, 3, 4, 5;
    X = np.random.randn(batchSize, sequenceNum, sequenceLength, sequenceDimension);
    F = np.random.randn(*X.shape);
    C = np.random.randn(*X.shape); # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerAddNormalizationModule(sequenceDimension),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, F);
    dX1, dF1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, F)[0]), X);
    dFN = numericGradient(lambda x: np.sum(m.forward(X, x)[0]), F);
    print(f"TransformerAddNormalizationModule, numericGradient2 {getErrorText('dX error', dX1, dXN)}, dF error: {np.sum(np.abs(dF1 - dFN))}");
    testModuleGradient(m, "TransformerAddNormalizationModule, numericGradient2", X, F);
    print("\n");


def testTransformerPositionwiseFFNModuleGradient1():
    batchSize, sequenceNum, sequenceLength, sequenceDimension, hiddenSize = 2, 3, 4, 5, 6;
    X = np.random.randn(batchSize, sequenceNum, sequenceLength, sequenceDimension);
    m = TransformerPositionwiseFFNModule(sequenceDimension, hiddenSize, activationFuncSelector = lambda size: PReluLayer(outputSize = size));

    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"TransformerPositionwiseFFNModule, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "TransformerPositionwiseFFNModule, numericGradient1", X);
    print("\n");


def testTransformerEncoderBlockGradient1():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum = 22, 23, 8;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    C = np.random.randn(*X.shape); # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEncoderBlock(sequenceDimension, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"TransformerEncoderBlock, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "TransformerEncoderBlock, numericGradient1", X);
    print("\n");


def testTransformerEncoderBlockGradient2():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum = 22, 23, 8;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    M = np.random.randint(0, 2, (batchSize, sequenceLength, sequenceLength));
    C = np.random.randn(*X.shape); # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEncoderBlock(sequenceDimension, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, M);
    dX1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"TransformerEncoderBlock, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "TransformerEncoderBlock, numericGradient2", X, M);
    print("\n");


def testTransformerEncoderBlockGradient3():
    batchSize, sequenceLength, sequenceDimension = 32, 20, 16;
    attentionHiddenSize, ffnHiddenSize, headNum = 17, 18, 8;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    validLength = np.random.randint(1, sequenceLength + 1, batchSize);
    M = getAttentionMaskByValidLength(sequenceLength, sequenceLength, validLength);
    C = np.random.randn(*X.shape); # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEncoderBlock(sequenceDimension, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, M);
    dX1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"TransformerEncoderBlock, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "TransformerEncoderBlock, numericGradient3", X, M);
    print("\n");


def testTransformerEncoderBlockGradient4():
    batchSize, sequenceNum, sequenceLength, sequenceDimension = 2, 3, 4, 5;
    attentionHiddenSize, ffnHiddenSize, headNum = 6, 7, 8;
    X = np.random.randn(batchSize, sequenceNum, sequenceLength, sequenceDimension);
    validLength = np.random.randint(1, sequenceLength + 1, (batchSize, sequenceNum));
    M = getAttentionMaskByValidLength(sequenceLength, sequenceLength, validLength, onlyBatch = True);
    C = np.random.randn(*X.shape); # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEncoderBlock(sequenceDimension, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, M);
    dX1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"TransformerEncoderBlock, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "TransformerEncoderBlock, numericGradient4", X, M);
    print("\n");


def testTransformerEncoderGradient1():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 22, 23, 8, 2;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    C = np.random.randn(*X.shape);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEncoder(sequenceDimension, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum, blockNum = blockNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"TransformerEncoder, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "TransformerEncoder, numericGradient1", X);
    print("\n");


def testTransformerEncoderGradient2():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 22, 23, 8, 2;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    M = np.random.randint(0, 2, (batchSize, sequenceLength, sequenceLength));
    C = np.random.randn(*X.shape);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEncoder(sequenceDimension, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum, blockNum = blockNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, M);
    dX1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"TransformerEncoder, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "TransformerEncoder, numericGradient2", X, M);
    print("\n");


def testTransformerEncoderGradient3():
    batchSize, sequenceLength, sequenceDimension = 8, 20, 16;
    attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 17, 18, 8, 2;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    validLength = np.random.randint(1, sequenceLength + 1, batchSize);
    M = getAttentionMaskByValidLength(sequenceLength, sequenceLength, validLength);
    C = np.random.randn(*X.shape);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEncoder(sequenceDimension, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum, blockNum = blockNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, M);
    dX1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"TransformerEncoder, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "TransformerEncoder, numericGradient3", X, M);
    print("\n");


def testTransformerEncoderGradient4():
    batchSize, sequenceNum, sequenceLength, sequenceDimension = 2, 3, 4, 5;
    attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 6, 7, 8, 9;
    X = np.random.randn(batchSize, sequenceNum, sequenceLength, sequenceDimension);
    validLength = np.random.randint(1, sequenceLength + 1, (batchSize, sequenceNum));
    M = getAttentionMaskByValidLength(sequenceLength, sequenceLength, validLength, onlyBatch = True);
    C = np.random.randn(*X.shape);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEncoder(sequenceDimension, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum, blockNum = blockNum, ffnActivationFuncSelector = lambda size: SwishLayer(outputSize = size)),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, M);
    dX1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"TransformerEncoder, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "TransformerEncoder, numericGradient4", X, M);
    print("\n");


def testTransformerDecoderBlockGradient1():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum = 22, 23, 8;
    Q = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    K = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    V = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    encoderY = np.random.randn(batchSize, sequenceLength + 1, sequenceDimension + 2);
    C = np.random.randn(*Q.shape);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerDecoderBlock(sequenceDimension, sequenceDimension + 2, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    m.context.isTrainingMode = True;

    Y, = m.forward(Q, K, V, encoderY);
    dQ1, dK1, dV1, dEncoderY1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, encoderY)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, encoderY)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, encoderY)[0]), V);
    dEncoderYN = numericGradient(lambda x: np.sum(m.forward(Q, K, V, x)[0]), encoderY);
    print(f"TransformerDecoderBlock, numericGradient1, dQ error: {np.sum(np.abs(dQ1 - dQN))}, dK error: {np.sum(np.abs(dK1 - dKN))}, dV error: {np.sum(np.abs(dV1 - dVN))}, dEncoderY error: {np.sum(np.abs(dEncoderY1 - dEncoderYN))}");
    testModuleGradient(m, "TransformerEncoderBlock, numericGradient1", Q, K, V, encoderY);
    print("\n");


def testTransformerDecoderBlockGradient2():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum = 22, 23, 8;
    Q = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    K = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    V = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    encoderY = np.random.randn(batchSize, sequenceLength + 1, sequenceDimension + 2);
    encoderM = np.random.randint(0, 2, (batchSize, sequenceLength, sequenceLength + 1));
    C = np.random.randn(*Q.shape); # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerDecoderBlock(sequenceDimension, sequenceDimension + 2, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    m.context.isTrainingMode = True;

    Y, = m.forward(Q, K, V, encoderY, encoderM);
    dQ1, dK1, dV1, dEncoderY1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, encoderY, encoderM)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, encoderY, encoderM)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, encoderY, encoderM)[0]), V);
    dEncoderYN = numericGradient(lambda x: np.sum(m.forward(Q, K, V, x, encoderM)[0]), encoderY);
    print(f"TransformerDecoderBlock, numericGradient2, dQ error: {np.sum(np.abs(dQ1 - dQN))}, dK error: {np.sum(np.abs(dK1 - dKN))}, dV error: {np.sum(np.abs(dV1 - dVN))}, dEncoderY error: {np.sum(np.abs(dEncoderY1 - dEncoderYN))}");
    testModuleGradient(m, "TransformerEncoderBlock, numericGradient2", Q, K, V, encoderY, encoderM);
    print("\n");


def testTransformerDecoderBlockGradient3():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum = 22, 23, 8;
    Q = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    K = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    V = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    encoderY = np.random.randn(batchSize, sequenceLength + 1, sequenceDimension + 2);
    encoderValidLength = np.random.randint(1, sequenceLength + 2, batchSize);
    encoderM = getAttentionMaskByValidLength(sequenceLength, sequenceLength + 1, encoderValidLength);
    C = np.random.randn(*Q.shape); # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerDecoderBlock(sequenceDimension, sequenceDimension + 2, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    m.context.isTrainingMode = True;

    Y, = m.forward(Q, K, V, encoderY, encoderM);
    dQ1, dK1, dV1, dEncoderY1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, encoderY, encoderM)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, encoderY, encoderM)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, encoderY, encoderM)[0]), V);
    dEncoderYN = numericGradient(lambda x: np.sum(m.forward(Q, K, V, x, encoderM)[0]), encoderY);
    print(f"TransformerDecoderBlock, numericGradient3, dQ error: {np.sum(np.abs(dQ1 - dQN))}, dK error: {np.sum(np.abs(dK1 - dKN))}, dV error: {np.sum(np.abs(dV1 - dVN))}, dEncoderY error: {np.sum(np.abs(dEncoderY1 - dEncoderYN))}");
    testModuleGradient(m, "TransformerEncoderBlock, numericGradient3", Q, K, V, encoderY, encoderM);
    print("\n");


def testTransformerDecoderBlockGradient4():
    batchSize, sequenceNum, sequenceLength, sequenceDimension = 2, 3, 4, 5;
    attentionHiddenSize, ffnHiddenSize, headNum = 6, 7, 8;
    Q = np.random.randn(batchSize, sequenceNum, sequenceLength, sequenceDimension);
    K = np.random.randn(batchSize, sequenceNum, sequenceLength, sequenceDimension);
    V = np.random.randn(batchSize, sequenceNum, sequenceLength, sequenceDimension);
    encoderY = np.random.randn(batchSize, sequenceNum, sequenceLength + 1, sequenceDimension + 2);
    encoderValidLength = np.random.randint(1, sequenceLength + 2, (batchSize, sequenceNum));
    encoderM = getAttentionMaskByValidLength(sequenceLength, sequenceLength + 1, encoderValidLength, onlyBatch = True);
    C = np.random.randn(*Q.shape); # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerDecoderBlock(sequenceDimension, sequenceDimension + 2, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    m.context.isTrainingMode = True;

    Y, = m.forward(Q, K, V, encoderY, encoderM);
    dQ1, dK1, dV1, dEncoderY1 = m.backward(np.ones_like(Y));
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K, V, encoderY, encoderM)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x, V, encoderY, encoderM)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m.forward(Q, K, x, encoderY, encoderM)[0]), V);
    dEncoderYN = numericGradient(lambda x: np.sum(m.forward(Q, K, V, x, encoderM)[0]), encoderY);
    print(f"TransformerDecoderBlock, numericGradient4, dQ error: {np.sum(np.abs(dQ1 - dQN))}, dK error: {np.sum(np.abs(dK1 - dKN))}, dV error: {np.sum(np.abs(dV1 - dVN))}, dEncoderY error: {np.sum(np.abs(dEncoderY1 - dEncoderYN))}");
    testModuleGradient(m, "TransformerEncoderBlock, numericGradient4", Q, K, V, encoderY, encoderM);
    print("\n");


def testTransformerDecoder1():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 22, 23, 8, 3;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    encoderY = np.random.randn(batchSize, sequenceLength + 1, sequenceDimension + 2);
    m = TransformerDecoder(sequenceDimension, sequenceDimension + 2, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum, blockNum = blockNum);
    m.context.isTrainingMode = True;

    Y1, = m.forward(X, encoderY);

    m.reset();
    m.context.isTrainingMode = False;
    blockInputs = [None] * blockNum;
    Y2 = np.concatenate(tuple(m.predict(X[:, i: i + 1, :], encoderY, blockInputs = blockInputs)[0] for i in range(X.shape[-2])), axis = -2);

    print(f"TransformerDecoder, value1, Y error: {np.sum(np.abs(Y1 - Y2))}");
    print("\n");


def testTransformerDecoder2():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 22, 23, 8, 3;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    encoderY = np.random.randn(batchSize, sequenceLength + 1, sequenceDimension + 2);
    encoderM = np.random.randint(0, 2, (batchSize, sequenceLength, sequenceLength + 1));
    m = TransformerDecoder(sequenceDimension, sequenceDimension + 2, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum, blockNum = blockNum);
    m.context.isTrainingMode = True;

    Y1, = m.forward(X, encoderY, encoderM);

    m.reset();
    m.context.isTrainingMode = False;
    blockInputs = [None] * blockNum;
    Y2 = np.concatenate(tuple(m.predict(X[:, i: i + 1, :], encoderY, encoderM[:, i: i + 1, :], blockInputs = blockInputs)[0] for i in range(X.shape[-2])), axis = -2);

    print(f"TransformerDecoder, value2, Y error: {np.sum(np.abs(Y1 - Y2))}");
    print("\n");


def testTransformerDecoderGradient1():
    batchSize, sequenceLength, sequenceDimension = 32, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 22, 23, 8, 3;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    encoderY = np.random.randn(batchSize, sequenceLength + 1, sequenceDimension + 2);
    C = np.random.randn(*X.shape);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerDecoder(sequenceDimension, sequenceDimension + 2, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum, blockNum = blockNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    m.context.isTrainingMode = True;

    Y, = m.forward(X, encoderY);
    dX1, dEncoderY1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, encoderY)[0]), X);
    dEncoderYN = numericGradient(lambda x: np.sum(m.forward(X, x)[0]), encoderY);
    print(f"TransformerDecoder, numericGradient1 {getErrorText('dX error', dX1, dXN)}, dEncoderY error: {np.sum(np.abs(dEncoderY1 - dEncoderYN))}");
    testModuleGradient(m, "TransformerDecoder, numericGradient1", X, encoderY);
    print("\n");


def testTransformerDecoderGradient2():
    batchSize, sequenceLength, sequenceDimension = 6, 10, 21;
    attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 22, 23, 8, 3;
    X = np.random.randn(batchSize, sequenceLength, sequenceDimension);
    encoderY = np.random.randn(batchSize, sequenceLength + 1, sequenceDimension + 2);
    encoderM = np.random.randint(0, 2, (batchSize, sequenceLength, sequenceLength + 1));
    C = np.random.randn(*X.shape);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerDecoder(sequenceDimension, sequenceDimension + 2, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum, blockNum = blockNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    m.context.isTrainingMode = True;

    Y, = m.forward(X, encoderY, encoderM);
    dX1, dEncoderY1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, encoderY, encoderM)[0]), X);
    dEncoderYN = numericGradient(lambda x: np.sum(m.forward(X, x, encoderM)[0]), encoderY);
    print(f"TransformerDecoder, numericGradient1 {getErrorText('dX error', dX1, dXN)}, dEncoderY error: {np.sum(np.abs(dEncoderY1 - dEncoderYN))}");
    testModuleGradient(m, "TransformerDecoder, numericGradient2", X, encoderY, encoderM);
    print("\n");


def testTransformerDecoderGradient3():
    batchSize, sequenceNum, sequenceLength, sequenceDimension = 2, 3, 4, 5;
    attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 6, 7, 8, 9;
    X = np.random.randn(batchSize, sequenceNum, sequenceLength, sequenceDimension);
    encoderY = np.random.randn(batchSize, sequenceNum, sequenceLength + 1, sequenceDimension + 2);
    encoderValidLength = np.random.randint(1, sequenceLength + 2, (batchSize, sequenceNum));
    encoderM = getAttentionMaskByValidLength(sequenceLength, sequenceLength + 1, encoderValidLength, onlyBatch = True);
    C = np.random.randn(*X.shape);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerDecoder(sequenceDimension, sequenceDimension + 2, attentionHiddenSize, ffnHiddenSize, sequenceDimension, headNum = headNum, blockNum = blockNum, ffnActivationFuncSelector = lambda size: GeluLayer()),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );
    m.context.isTrainingMode = True;

    Y, = m.forward(X, encoderY, encoderM);
    dX1, dEncoderY1 = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, encoderY, encoderM)[0]), X);
    dEncoderYN = numericGradient(lambda x: np.sum(m.forward(X, x, encoderM)[0]), encoderY);
    print(f"TransformerDecoder, numericGradient3 {getErrorText('dX error', dX1, dXN)}, dEncoderY error: {np.sum(np.abs(dEncoderY1 - dEncoderYN))}");
    testModuleGradient(m, "TransformerDecoder, numericGradient3", X, encoderY, encoderM);
    print("\n");


def testTransformerEmbeddingEncoderGradient1():
    batchSize, sequenceLength = 32, 20;
    vocabSize, embeddingSize, attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 120, 16, 17, 18, 8, 2;
    X = np.random.randint(0, vocabSize, (batchSize, sequenceLength));
    C = np.random.randn(batchSize, sequenceLength, embeddingSize);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEmbeddingEncoder(vocabSize, embeddingSize, attentionHiddenSize, ffnHiddenSize, embeddingSize, headNum = headNum, blockNum = blockNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X);
    dX1 = m.backward(np.ones_like(Y));
    testModuleGradient(m, "TransformerEmbeddingEncoder, numericGradient1", X);
    print("\n");


def testTransformerEmbeddingEncoderGradient2():
    batchSize, sequenceLength = 32, 20;
    vocabSize, embeddingSize, attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 120, 16, 17, 18, 8, 2;
    X = np.random.randint(0, vocabSize, (batchSize, sequenceLength));
    validLength = np.random.randint(1, sequenceLength + 1, batchSize);
    C = np.random.randn(batchSize, sequenceLength, embeddingSize);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEmbeddingEncoder(vocabSize, embeddingSize, attentionHiddenSize, ffnHiddenSize, embeddingSize, headNum = headNum, blockNum = blockNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, validLength);
    dX1 = m.backward(np.ones_like(Y));
    testModuleGradient(m, "TransformerEmbeddingEncoder, numericGradient2", X, validLength);
    print("\n");


def testTransformerEmbeddingEncoderGradient3():
    batchSize, sequenceNum, sequenceLength = 2, 3, 4;
    vocabSize, embeddingSize, attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 5, 6, 7, 8, 9, 10;
    X = np.random.randint(0, vocabSize, (batchSize, sequenceNum, sequenceLength));
    validLength = np.random.randint(1, sequenceLength + 1, (batchSize, sequenceNum));
    C = np.random.randn(batchSize, sequenceNum, sequenceLength, embeddingSize);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEmbeddingEncoder(vocabSize, embeddingSize, attentionHiddenSize, ffnHiddenSize, embeddingSize, headNum = headNum, blockNum = blockNum, ffnActivationFuncSelector = lambda size: SwishLayer(outputSize = size)),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, validLength);
    dX1 = m.backward(np.ones_like(Y));
    testModuleGradient(m, "TransformerEmbeddingEncoder, numericGradient3", X, validLength);
    print("\n");


def testTransformerEmbeddingDecoder1():
    batchSize, sequenceLength = 32, 20;
    vocabSize, embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 120, 16, 17, 18, 19, 8, 3;
    X = np.random.randint(0, vocabSize, (batchSize, sequenceLength));
    encoderY = np.random.randn(batchSize, sequenceLength + 1, encoderSize);
    m = TransformerEmbeddingDecoder(vocabSize, embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, embeddingSize, headNum = headNum, blockNum = blockNum);
    m.context.isTrainingMode = True;

    Y1, = m.forward(X, encoderY);

    m.reset();
    m.context.isTrainingMode = False;
    blockInputs = [None] * blockNum;
    Y2 = np.concatenate(tuple(m.predict(X[:, i: i + 1], encoderY, blockInputs = blockInputs)[0] for i in range(X.shape[-1])), axis = -2);

    print(f"TransformerEmbeddingDecoder, value1, Y error: {np.sum(np.abs(Y1 - Y2))}");
    print("\n");


def testTransformerEmbeddingDecoder2():
    batchSize, sequenceLength = 32, 20;
    vocabSize, embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 120, 16, 17, 18, 19, 8, 3;
    X = np.random.randint(0, vocabSize, (batchSize, sequenceLength));
    encoderY = np.random.randn(batchSize, sequenceLength + 1, encoderSize);
    encoderValidLength = np.random.randint(1, sequenceLength + 2, batchSize);
    m = TransformerEmbeddingDecoder(vocabSize, embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, embeddingSize, headNum = headNum, blockNum = blockNum);
    m.context.isTrainingMode = True;

    Y1, = m.forward(X, encoderY, encoderValidLength);

    m.reset();
    m.context.isTrainingMode = False;
    blockInputs = [None] * blockNum;
    Y2 = np.concatenate(tuple(m.predict(X[:, i: i + 1], encoderY, encoderValidLength, blockInputs = blockInputs)[0] for i in range(X.shape[-1])), axis = -2);

    print(f"TransformerEmbeddingDecoder, value1, Y error: {np.sum(np.abs(Y1 - Y2))}");
    print("\n");


def testTransformerEmbeddingDecoderGradient1():
    batchSize, sequenceLength = 32, 20;
    vocabSize, embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 120, 16, 17, 18, 19, 8, 2;
    X = np.random.randint(0, vocabSize, (batchSize, sequenceLength));
    encoderY = np.random.randn(batchSize, sequenceLength + 1, encoderSize);
    C = np.random.randn(batchSize, sequenceLength, embeddingSize);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEmbeddingDecoder(vocabSize, embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, embeddingSize, headNum = headNum, blockNum = blockNum),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, encoderY);
    dX1, dEncoderY1 = m.backward(np.ones_like(Y));
    dEncoderYN = numericGradient(lambda x: np.sum(m.forward(X, x)[0]), encoderY);
    print(f"TransformerEmbeddingDecoder, numericGradient1, dEncoderY error: {np.sum(np.abs(dEncoderY1 - dEncoderYN))}");
    testModuleGradient(m, "TransformerEmbeddingDecoder, numericGradient1", X, encoderY);
    print("\n");


def testTransformerEmbeddingDecoderGradient2():
    batchSize, sequenceLength = 32, 20;
    vocabSize, embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 120, 16, 17, 18, 19, 8, 2;
    X = np.random.randint(0, vocabSize, (batchSize, sequenceLength));
    encoderY = np.random.randn(batchSize, sequenceLength + 1, encoderSize);
    encoderValidLength = np.random.randint(1, sequenceLength + 2, batchSize);
    C = np.random.randn(batchSize, sequenceLength, embeddingSize);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEmbeddingDecoder(vocabSize, embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, embeddingSize, headNum = headNum, blockNum = blockNum, ffnActivationFuncSelector = lambda size: GeluLayer()),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, encoderY, encoderValidLength);
    dX1, dEncoderY1 = m.backward(np.ones_like(Y));
    dEncoderYN = numericGradient(lambda x: np.sum(m.forward(X, x, encoderValidLength)[0]), encoderY);
    print(f"TransformerEmbeddingDecoder, numericGradient2, dEncoderY error: {np.sum(np.abs(dEncoderY1 - dEncoderYN))}");
    testModuleGradient(m, "TransformerEmbeddingDecoder, numericGradient2", X, encoderY, encoderValidLength);
    print("\n");


def testTransformerEmbeddingDecoderGradient3():
    batchSize, sequenceNum, sequenceLength = 2, 3, 4;
    vocabSize, embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, headNum, blockNum = 5, 6, 7, 8, 9, 10, 11;
    X = np.random.randint(0, vocabSize, (batchSize, sequenceNum, sequenceLength));
    encoderY = np.random.randn(batchSize, sequenceNum, sequenceLength + 1, encoderSize);
    encoderValidLength = np.random.randint(1, sequenceLength + 2, (batchSize, sequenceNum));
    C = np.random.randn(batchSize, sequenceNum, sequenceLength, embeddingSize);  # if not multiple C, np.sum(Y) is always zero, dX1 will be zero too.
    m = SequentialContainer(
        TransformerEmbeddingDecoder(vocabSize, embeddingSize, encoderSize, attentionHiddenSize, ffnHiddenSize, embeddingSize, headNum = headNum, blockNum = blockNum, ffnActivationFuncSelector = lambda size: SwishLayer(outputSize = size)),
        FunctionalNetModule("*C", lambda x: x * C, lambda x, y, dy: dy * C),
    );

    Y, = m.forward(X, encoderY, encoderValidLength);
    dX1, dEncoderY1 = m.backward(np.ones_like(Y));
    dEncoderYN = numericGradient(lambda x: np.sum(m.forward(X, x, encoderValidLength)[0]), encoderY);
    print(f"TransformerEmbeddingDecoder, numericGradient3, dEncoderY error: {np.sum(np.abs(dEncoderY1 - dEncoderYN))}");
    testModuleGradient(m, "TransformerEmbeddingDecoder, numericGradient3", X, encoderY, encoderValidLength);
    print("\n");


def testAttentionPoolingLayerGradient1():
    batchSize, sequenceLength, inputSize, hiddenSize = 32, 20, 21, None;
    X = np.random.randn(batchSize, sequenceLength, inputSize);
    m = AttentionPoolingLayer(inputSize, hiddenSize);

    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AttentionPoolingLayer, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "AttentionPoolingLayer, numericGradient1", X);
    print("\n");


def testAttentionPoolingLayerGradient2():
    batchSize, sequenceLength, inputSize, hiddenSize = 32, 20, 21, 22;
    X = np.random.randn(batchSize, sequenceLength, inputSize);
    m = AttentionPoolingLayer(inputSize, hiddenSize);

    Y, = m.forward(X);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x)[0]), X);
    print(f"AttentionPoolingLayer, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "AttentionPoolingLayer, numericGradient2", X);
    print("\n");


def testAttentionPoolingLayerGradient3():
    batchSize, sequenceLength, inputSize, hiddenSize = 32, 20, 21, (22, 23);
    X = np.random.randn(batchSize, sequenceLength, inputSize);
    M = np.random.randint(0, 2, (batchSize, sequenceLength));
    m = AttentionPoolingLayer(inputSize, hiddenSize);

    Y, = m.forward(X, M);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"AttentionPoolingLayer, numericGradient3 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "AttentionPoolingLayer, numericGradient3", X, M);
    print("\n");


def testAttentionPoolingLayerGradient4():
    batchSize, sequenceNum, sequenceLength, inputSize, hiddenSize = 2, 3, 4, 5, (6, 7);
    X = np.random.randn(batchSize, sequenceNum, sequenceLength, inputSize);
    M = np.random.randint(0, 2, (batchSize, sequenceNum, sequenceLength));
    m = AttentionPoolingLayer(inputSize, hiddenSize, activationFuncSelector = lambda x: GeluLayer());

    Y, = m.forward(X, M);
    dX1, = m.backward(np.ones_like(Y));
    dXN = numericGradient(lambda x: np.sum(m.forward(x, M)[0]), X);
    print(f"AttentionPoolingLayer, numericGradient4 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "AttentionPoolingLayer, numericGradient4", X, M);
    print("\n");


def testRnnWithCupy():
    inputSize, rnnHiddenSize, layerNum = 32, 64, 2;
    rnnSelector = lambda ds, hs, sf, rq, rs: LstmLayer(ds, hs, sf, rq, rs);
    # model = StackRnnLayer(inputSize, rnnHiddenSize, rnnSelector, normalSelector = LayerNormalizationLayer, layersNum = layerNum, stateful = False, returnSequence = True, returnState = False);
    model = RnnLayer(inputSize, rnnHiddenSize, stateful = False, returnSequence = True, returnState = False);

    batchSize, sequenceLength = 32, 240;
    data = np.random.randn(batchSize * 100, sequenceLength, inputSize).astype(defaultDType);
    dataIterator = SequentialDataIterator([data], batchSize = batchSize, shuffle = True);

    print(f"dtype: {data.dtype}, device: {data.device}");

    t1 = time.time();
    for _ in range(10):
        for X in dataIterator:
            Y = model.forward(X[0].transpose(1, 0, 2));
    t2 = time.time();
    
    print(f"time: {t2 - t1}s");


def testRnnWithTorch():
    device = torch.device("cpu");
    # device = torch.device("cuda:0");

    inputSize, rnnHiddenSize, layerNum = 32, 64, 2;
    rnnSelector = lambda ds, hs, sf, rq, rs: LstmLayer(ds, hs, sf, rq, rs);
    model = nn.RNN(inputSize, rnnHiddenSize, device = device);

    batchSize, sequenceLength = 32, 240;
    data = torch.randn(batchSize * 100, sequenceLength, inputSize, device = device);
    dataIterator = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size = batchSize, shuffle = True);

    print(f"dtype: {data.dtype}, device: {data.device}");

    t1 = time.time();
    for _ in range(10):
        for X in dataIterator:
            H = torch.zeros(1, batchSize, rnnHiddenSize, device = device);
            Y = model(X[0].transpose(1, 0), H);
    t2 = time.time();
    
    print(f"time: {t2 - t1}s");


def testSelectByWeightModuleGradient():
    V, W = np.random.randn(2, 32, 11, 16), np.random.randn(2, 32, 12, 11);
    m1 = SelectByWeight1TModule();
    m2 = SelectByWeightNTModule();

    Y1 = np.zeros(W.shape[:-1] + (V.shape[-1], ));
    dV1, dW1 = np.zeros_like(V), np.zeros_like(W);
    for t in range(W.shape[-2]):
        Y1[..., t, :] = m1.forward(W[..., t, :], V)[0];
        dW, dV = m1.backward(np.ones_like(Y1[..., t, :]));
        dV1 += dV;
        dW1[..., t, :] = dW;

    Y2 = m2.forward(W, V)[0];
    dW2, dV2 = m2.backward(np.ones_like(Y2));
    print(f"SelectByWeightModule, batch process, Y error: {np.sum(np.abs(Y1 - Y2))}, dV error: {np.sum(np.abs(dV1 - dV2))}, dW error: {np.sum(np.abs(dW1 - dW2))}");

    dVN = numericGradient(lambda x: np.sum(m2.forward(W, x)[0]), V);
    dWN = numericGradient(lambda x: np.sum(m2.forward(x, V)[0]), W);
    print(f"SelectByWeightModule, numericGradient, dV error: {np.sum(np.abs(dV2 - dVN))}, dW error: {np.sum(np.abs(dW2 - dWN))}");
    print("\n");


def testAdditiveAttentionWeight1TModuleGradient():
    N, T1, querySize, keySize, hiddenSize = 32, 8, 12, 16, 24;
    K, Q = np.random.randn(N, T1, keySize), np.random.randn(N, querySize);
    m = AdditiveAttentionWeight1TModule(querySize, keySize, hiddenSize);
    Wq, Wk, wv = tuple(m.params);

    Y = m.forward(Q, K)[0];
    dQ1, dK1 = m.backward(np.ones_like(Y));
    dWq1, dWk1, dwv1 = tuple(m.grads);
    dKN = numericGradient(lambda x: np.sum(m.forward(Q, x)[0]), K);
    dQN = numericGradient(lambda x: np.sum(m.forward(x, K)[0]), Q);
    dWqN = numericGradient(lambda x: np.sum(AdditiveAttentionWeight1TModule(querySize, keySize, hiddenSize, Wq = x, Wk = Wk, wv = wv).forward(Q, K)[0]), Wq);
    dWkN = numericGradient(lambda x: np.sum(AdditiveAttentionWeight1TModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = x, wv = wv).forward(Q, K)[0]), Wk);
    dwvN = numericGradient(lambda x: np.sum(AdditiveAttentionWeight1TModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = Wk, wv = x).forward(Q, K)[0]), wv);
    print(f"AdditiveAttentionWeight1TModule, numericGradient, dK error: {np.sum(np.abs(dK1 - dKN))}, dQ error: {np.sum(np.abs(dQ1 - dQN))}, dWq error: {np.sum(np.abs(dWq1 - dWqN))}, dWk error: {np.sum(np.abs(dWk1 - dWkN))}, dwv error: {np.sum(np.abs(dwv1 - dwvN))}");
    print("\n");


def testAdditiveAttentionWeightNTModuleGradient():
    N, T1, T2, querySize, keySize, hiddenSize = 32, 11, 12, 22, 21, 24;
    K, Q = np.random.randn(N, T1, keySize), np.random.randn(N, T2, querySize);
    m1 = AdditiveAttentionWeight1TModule(querySize, keySize, hiddenSize);
    Wq, Wk, wv = tuple(m1.params);
    m2 = AdditiveAttentionWeightNTModule(querySize, keySize, hiddenSize, Wq, Wk, wv);

    Y1 = np.zeros((N, T2, T1));
    dK1, dQ1 = np.zeros_like(K), np.zeros_like(Q);
    dWq1, dWk1, dwv1 = np.zeros_like(Wq), np.zeros_like(Wk), np.zeros_like(wv);
    for t in range(Q.shape[-2]):
        Y1[..., t, :] = m1.forward(Q[..., t, :], K)[0];
        dQ, dK = m1.backward(np.ones_like(Y1[..., t, :]));
        dK1 += dK;
        dQ1[..., t, :] = dQ;
        dWq1 += m1.grads[0];
        dWk1 += m1.grads[1];
        dwv1 += m1.grads[2];

    Y2 = m2.forward(Q, K)[0];
    dQ2, dK2 = m2.backward(np.ones_like(Y2));
    dWq2, dWk2, dwv2 = tuple(m2.grads);
    print(f"AdditiveAttentionWeightNTModule, batch process, Y error: {np.sum(np.abs(Y1 - Y2))}, dK error: {np.sum(np.abs(dK1 - dK2))}, dQ error: {np.sum(np.abs(dQ1 - dQ2))}, dWq error: {np.sum(np.abs(dWq1 - dWq2))}, dWk error: {np.sum(np.abs(dWk1 - dWk2))}, dwv error: {np.sum(np.abs(dwv1 - dwv2))}");

    dKN = numericGradient(lambda x: np.sum(m2.forward(Q, x)[0]), K);
    dQN = numericGradient(lambda x: np.sum(m2.forward(x, K)[0]), Q);
    dWqN = numericGradient(lambda x: np.sum(AdditiveAttentionWeightNTModule(querySize, keySize, hiddenSize, Wq = x, Wk = Wk, wv = wv).forward(Q, K)[0]), Wq);
    dWkN = numericGradient(lambda x: np.sum(AdditiveAttentionWeightNTModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = x, wv = wv).forward(Q, K)[0]), Wk);
    dwvN = numericGradient(lambda x: np.sum(AdditiveAttentionWeightNTModule(querySize, keySize, hiddenSize, Wq = Wq, Wk = Wk, wv = x).forward(Q, K)[0]), wv);
    print(f"AdditiveAttentionWeightNTModule, numericGradient, dK error: {np.sum(np.abs(dK2 - dKN))}, dQ error: {np.sum(np.abs(dQ2 - dQN))}, dWq error: {np.sum(np.abs(dWq2 - dWqN))}, dWk error: {np.sum(np.abs(dWk2 - dWkN))}, dwv error: {np.sum(np.abs(dwv2 - dwvN))}");
    print("\n");


def testDotProductAttentionWeightModuleGradient():
    N1, N2, T1, T2, querySize = 2, 32, 11, 12, 24;
    K, Q = np.random.randn(N1, N2, T1, querySize), np.random.randn(N1, N2, T2, querySize);
    m1 = DotProductAttentionWeight1TModule();
    m2 = DotProductAttentionWeightNTModule();

    Y1 = np.zeros(Q.shape[:-1] + (K.shape[-2], ));
    dK1, dQ1 = np.zeros_like(K), np.zeros_like(Q);
    for t in range(Q.shape[-2]):
        Y1[..., t, :] = m1.forward(Q[..., t, :], K)[0];
        dQ, dK = m1.backward(np.ones_like(Y1[..., t, :]));
        dK1 += dK;
        dQ1[..., t, :] = dQ;

    Y2 = m2.forward(Q, K)[0];
    dQ2, dK2 = m2.backward(np.ones_like(Y2));
    print(f"DotProductAttentionWeight, batch process, Y error: {np.sum(np.abs(Y1 - Y2))}, dK error: {np.sum(np.abs(dK1 - dK2))}, dQ error: {np.sum(np.abs(dQ1 - dQ2))}");

    dKN = numericGradient(lambda x: np.sum(m2.forward(Q, x)[0]), K);
    dQN = numericGradient(lambda x: np.sum(m2.forward(x, K)[0]), Q);
    print(f"DotProductAttentionWeight, numericGradient, dK error: {np.sum(np.abs(dK2 - dKN))}, dQ error: {np.sum(np.abs(dQ2 - dQN))}");
    print("\n");


def testQKVAttentionLayerGradient():
    K, V, Q = np.random.randn(2, 32, 11, 16), np.random.randn(2, 32, 11, 16), np.random.randn(2, 32, 12, 16);
    m1 = QKVAttentionLayer(DotProductAttentionWeight1TModule(), SelectByWeight1TModule());
    m2 = QKVAttentionLayer(DotProductAttentionWeightNTModule(), SelectByWeightNTModule());

    Y1 = np.zeros_like(Q);
    dK1, dV1, dQ1 = np.zeros_like(K), np.zeros_like(V), np.zeros_like(Q);
    for t in range(Q.shape[-2]):
        Y1[..., t, :] = m1.forward(Q[..., t, :], K, V)[0];
        dQ, dK, dV = m1.backward(np.ones_like(Y1[..., t, :]));
        dK1 += dK;
        dV1 += dV;
        dQ1[..., t, :] = dQ;

    Y2 = m2.forward(Q, K, V)[0];
    dQ2, dK2, dV2 = m2.backward(np.ones_like(Y2));
    print(f"QKVAttention, batch process, Y error: {np.sum(np.abs(Y1 - Y2))}, dQ error: {np.sum(np.abs(dQ1 - dQ2))}, dK error: {np.sum(np.abs(dK1 - dK2))}, dV error: {np.sum(np.abs(dV1 - dV2))}");

    dQN = numericGradient(lambda x: np.sum(m2.forward(x, K, V)[0]), Q);
    dKN = numericGradient(lambda x: np.sum(m2.forward(Q, x, V)[0]), K);
    dVN = numericGradient(lambda x: np.sum(m2.forward(Q, K, x)[0]), V);
    print(f"QKVAttention, numericGradient, dQ error: {np.sum(np.abs(dQ2 - dQN))}, dK error: {np.sum(np.abs(dK2 - dKN))}, dV error: {np.sum(np.abs(dV2 - dVN))}");
    print("\n");


def testGaussianVAELossGradient():
    N, D, H, L = 32, 8, 4, 16;
    X, M, V, E, U = np.random.randn(N, D), np.random.randn(N, H), softplus(np.random.randn(N, H)), np.random.randn(N, L, D), softplus(np.random.randn(N, L, D));

    m = GaussianVAELoss();
    loss = m.forward(X, M, V, E, U);
    dX1, dM1, dV1, dE1, dU1 = m.backward();
    dXN = numericGradient(lambda x: m.forward(x, M, V, E, U), X);
    dMN = numericGradient(lambda x: m.forward(X, x, V, E, U), M);
    dVN = numericGradient(lambda x: m.forward(X, M, x, E, U), V);
    dEN = numericGradient(lambda x: m.forward(X, M, V, x, U), E);
    dUN = numericGradient(lambda x: m.forward(X, M, V, E, x), U);
    print(f"GaussianVAELoss, numericGradient {getErrorText('dX error', dX1, dXN)}, dM error: {np.sum(np.abs(dM1 - dMN))}, dV error: {np.sum(np.abs(dV1 - dVN))}, dE error: {np.sum(np.abs(dE1 - dEN))}, dU error: {np.sum(np.abs(dU1 - dUN))}");
    print("\n");


def testGaussianVAEGradient1():
    N, D, H, L = 32, 8, 4, 16;
    X, epsilon = np.random.randn(N, D), np.random.randn(N, L, H);

    m = GaussianVAE(AggregateNetModule(
        AffineLayer(D, 32, W = np.random.randn(D, 32), b = np.random.randn(32)),
        SigmoidLayer(),
        AffineLayer(32, 16, W = np.random.randn(32, 16), b = np.random.randn(16)),
        SigmoidLayer(),
        AffineLayer(16, 2 * H, W = np.random.randn(16, 2 * H), b = np.random.randn(2 * H)),
    ), AggregateNetModule(
        AffineLayer(H, 16, W = np.random.randn(H, 16), b = np.random.randn(16)),
        SigmoidLayer(),
        AffineLayer(16, 32, W = np.random.randn(16, 32), b = np.random.randn(32)),
        SigmoidLayer(),
        AffineLayer(32, 2 * D, W = np.random.randn(32, 2 * D), b = np.random.randn(2 * D)),
    ), H, L);
    X, M, V, E, U = m.forward(X, epsilon);
    dX1 = m.backward(np.ones_like(X), np.ones_like(M), np.ones_like(V), np.ones_like(E), np.ones_like(U))[0];
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, epsilon)), X);
    print(f"GaussianVAE, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "GaussianVAE, numericGradient1", X, epsilon);
    print("\n");


def testGaussianVAEGradient2():
    N, D, H, L = 32, 8, 4, 16;
    X, epsilon = np.random.randn(N, D), np.random.randn(N, L, H);

    m = GaussianVAE(AggregateNetModule(
        AffineLayer(D, 32),#, W = np.random.randn(D, 32), b = np.random.randn(32)),
        SoftplusLayer(),
        AffineLayer(32, 16),#, W = np.random.randn(32, 16), b = np.random.randn(16)),
        SoftplusLayer(),
        AffineLayer(16, 2 * H)#, W = np.random.randn(16, 2 * H), b = np.random.randn(2 * H)),
    ), AggregateNetModule(
        AffineLayer(H, 16),#, W = np.random.randn(H, 16), b = np.random.randn(16)),
        SoftplusLayer(),
        AffineLayer(16, 32),#, W = np.random.randn(16, 32), b = np.random.randn(32)),
        SoftplusLayer(),
        AffineLayer(32, 2 * D)#, W = np.random.randn(32, 2 * D), b = np.random.randn(2 * D)),
    ), H, L);
    loss = GaussianVAELoss();
    loss.forward(*m.forward(X, epsilon));
    dX1 = m.backward(*loss.backward())[0];
    dXN = numericGradient(lambda x: loss.forward(*m.forward(x, epsilon)), X);
    print(f"GaussianVAE, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testGaussianVAE_Anomaly_HTRU2():
    dataset = np.loadtxt("/media/WindowsE/Data/UCI/HTRU2/HTRU_2.csv", delimiter = ",").astype(defaultDType);

    D = dataset.shape[-1] - 1;
    filename = f"GaussianVAE_Anomaly_HTRU2.result";
    hiddenSize1, latentSize, batchSize, maxEpoch = 2 * D, D // 2, 32, 100;
    model = GaussianVAE(AggregateNetModule(
        AffineLayer(D, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, 2 * latentSize),
    ), AggregateNetModule(
        AffineLayer(latentSize, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, 2 * D),
    ), latentSize);

    X = dataset[dataset[:, -1] == 0, : -1];
    np.random.shuffle(X);
    trainSize = int(len(X) * 0.8);
    X_train, X_test_normal, X_test_anomaly = X[: trainSize], X[trainSize:], dataset[dataset[:, -1] == 1, : -1];

    scaler = StandardScaler();
    scaler.fit(X_train);
    X_train, X_test_normal, X_test_anomaly = scaler.transform(X_train), scaler.transform(X_test_normal), scaler.transform(X_test_anomaly);

    lossFunc = GaussianVAELoss();
    optimizer = Adam();
    trainingIterator = SequentialDataIterator([X_train], batchSize);
    model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, plot = True);

    with open(filename, "wb") as file:
        pickle.dump((model.params, X_train, X_test_normal, X_test_anomaly), file);

    # if os.path.isfile(filename):
    #     with open(filename, "br") as file:
    #         params, X_train, X_test_normal, X_test_anomaly = pickle.load(file);
    #     if isinstance(params[0], cp.ndarray):
    #         params = [cp.asnumpy(p) for p in params];
    #     for i in range(len(params)):
    #         params[i] = params[i].astype(defaultDType);
    #     model.params = params;
    #
    # M, V = model.encode(X_train);
    #
    # X_test_normal_rp = model.reconstructionProbability(X_test_normal);
    # X_test_anomaly_rp = model.reconstructionProbability(X_test_anomaly);
    # X_train_rp = np.concatenate(tuple([model.reconstructionProbability(item[0]) for item in SequentialDataIterator([X_train], batchSize, False)]), axis = 0);
    #
    # plt.figure(figsize = (12, 14));
    # plt.hist(X_train_rp, bins = 1000, color = "k");
    # plt.hist(X_test_normal_rp, bins = 1000, label = "normal", color = "b");
    # plt.hist(X_test_anomaly_rp, bins = 1000, label = "anomaly", color = "r");
    # plt.show(block = True);
    #
    # fprX, tprY = [], [];
    # for alpha in np.arange(0, 1.001, 0.001):
    #     threshold = np.quantile(X_train_rp, alpha);
    #     X_test_normal_class = X_test_normal_rp < threshold;
    #     X_test_anomaly_class = X_test_anomaly_rp < threshold;
    #
    #     TP, FP = np.sum(X_test_anomaly_class == True), np.sum(X_test_normal_class == True);
    #     FN, TN = np.sum(X_test_anomaly_class == False), np.sum(X_test_normal_class == False);
    #     precision, recall = TP / (TP + FP), TP / (TP + FN);
    #     fpr, tpr = FP / (FP + TN), recall;
    #     fprX.append(fpr);
    #     tprY.append(tpr);
    #     print(f"alpha = {alpha}, precision = {precision}, recall = {recall}, fpr = {fpr}, tpr = {tpr}");
    #
    # plt.figure(figsize = (12, 14));
    # plt.plot(fprX, tprY);
    # plt.xlabel("FPR");
    # plt.ylabel("TPR");
    # plt.title(f"AUC={auc(-X_test_anomaly_rp, -X_test_normal_rp)}");
    # plt.show(block = True);

    print("exit.");


def testBernoulliVAELossGradient():
    N, D, H, L = 32, 8, 4, 16;
    X, M, V, Y = np.random.randn(N, D), np.random.randn(N, H), softplus(np.random.randn(N, H)), np.random.randn(N, L, D);

    m = BernoulliVAELoss();
    loss = m.forward(X, M, V, Y);
    dX1, dM1, dV1, dY1 = m.backward();
    dXN = numericGradient(lambda x: m.forward(x, M, V, Y), X);
    dMN = numericGradient(lambda x: m.forward(X, x, V, Y), M);
    dVN = numericGradient(lambda x: m.forward(X, M, x, Y), V);
    dYN = numericGradient(lambda x: m.forward(X, M, V, x), Y);
    print(f"BernoulliVAELoss, numericGradient {getErrorText('dX error', dX1, dXN)}, dM error: {np.sum(np.abs(dM1 - dMN))}, dV error: {np.sum(np.abs(dV1 - dVN))}, dY error: {np.sum(np.abs(dY1 - dYN))}");
    print("\n");


def testBernoulliVAEGradient1():
    N, D, H, L = 32, 8, 4, 16;
    X, epsilon = np.random.randn(N, D), np.random.randn(N, L, H);

    m = BernoulliVAE(AggregateNetModule(
        AffineLayer(D, 32, W = np.random.randn(D, 32), b = np.random.randn(32)),
        SigmoidLayer(),
        AffineLayer(32, 16, W = np.random.randn(32, 16), b = np.random.randn(16)),
        SigmoidLayer(),
        AffineLayer(16, 2 * H, W = np.random.randn(16, 2 * H), b = np.random.randn(2 * H)),
    ), AggregateNetModule(
        AffineLayer(H, 16, W = np.random.randn(H, 16), b = np.random.randn(16)),
        SigmoidLayer(),
        AffineLayer(16, 32, W = np.random.randn(16, 32), b = np.random.randn(32)),
        SigmoidLayer(),
        AffineLayer(32, D, W = np.random.randn(32, D), b = np.random.randn(D)),
    ), H, L);
    X, M, V, Y = m.forward(X, epsilon);
    dX1 = m.backward(np.ones_like(X), np.ones_like(M), np.ones_like(V), np.ones_like(Y))[0];
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, epsilon)), X);
    print(f"BernoulliVAE, numericGradient1 {getErrorText('dX error', dX1, dXN)}");
    testModuleGradient(m, "BernoulliVAE, numericGradient1", X, epsilon);
    print("\n");


def testBernoulliVAEGradient2():
    N, D, H, L = 32, 8, 4, 16;
    X, epsilon = np.random.randn(N, D), np.random.randn(N, L, H);

    m = BernoulliVAE(AggregateNetModule(
        AffineLayer(D, 32), #, W = np.random.randn(D, 32), b = np.random.randn(32)),
        SoftplusLayer(),
        AffineLayer(32, 16), #, W = np.random.randn(32, 16), b = np.random.randn(16)),
        SoftplusLayer(),
        AffineLayer(16, 2 * H), #, W = np.random.randn(16, 2 * H), b = np.random.randn(2 * H)),
    ), AggregateNetModule(
        AffineLayer(H, 16), #, W = np.random.randn(H, 16), b = np.random.randn(16)),
        SoftplusLayer(),
        AffineLayer(16, 32), #, W = np.random.randn(16, 32), b = np.random.randn(32)),
        SoftplusLayer(),
        AffineLayer(32, D)#, W = np.random.randn(32, D), b = np.random.randn(D)),
    ), H, L);
    loss = BernoulliVAELoss();
    loss.forward(*m.forward(X, epsilon));
    dX1 = m.backward(*loss.backward())[0];
    dXN = numericGradient(lambda x: loss.forward(*m.forward(x, epsilon)), X);
    print(f"BernoulliVAE, numericGradient2 {getErrorText('dX error', dX1, dXN)}");
    print("\n");


def testBernoulliVAE_MNIST():
    mnist = MNIST("/media/WindowsE/Data/MNIST", flatten = True, normalize = True);
    X_train, X_test = mnist.trainX, mnist.testX;
    Y_train, Y_test = mnist.trainY, mnist.testY;

    D = X_train.shape[-1];
    filename = "BernoulliVAE_MNIST.weights";
    hiddenSize1, hiddenSize2, latentSize, batchSize, maxEpoch = 1000, 400, 200, 32, 10;
    model = BernoulliVAE(AggregateNetModule(
        AffineLayer(D, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, hiddenSize2),
        ReluLayer(),
        AffineLayer(hiddenSize2, 2 * latentSize),
    ), AggregateNetModule(
        AffineLayer(latentSize, hiddenSize2),
        ReluLayer(),
        AffineLayer(hiddenSize2, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, D),
    ), latentSize);

    # if os.path.isfile(filename):
    #     with open(filename, "br") as file:
    #         params = pickle.load(file);
    #     if isinstance(params[0], cp.ndarray):
    #         params = [cp.asnumpy(p) for p in params];
    #     for i in range(len(params)):
    #         params[i] = params[i].astype(defaultDType);
    #     model.params = params;
    #
    # M, V = model.encode(X_train);
    # plt.figure(figsize = (12, 14));
    # plt.scatter(M[:, 0], M[:, 1], c = Y_train);
    # # plt.scatter(V[:, 0], V[:, 1], c = Y_train);
    # plt.colorbar();
    # plt.xlabel("Z[0]");
    # plt.ylabel("Z[1]");
    # plt.show(block = True);
    #
    # digitSize, digitNumber, scale = 28, 30, 1;
    #
    # P = np.concatenate(tuple([model.reconstructionProbability(item[0]) for item in SequentialDataIterator([X_train], batchSize, False)]), axis = 0);
    # plt.figure(figsize = (12, 14));
    # plt.hist(P, bins = 1000);
    # plt.show(block = True);
    #
    # gridX = np.linspace(-scale, scale, digitNumber);
    # gridY = np.linspace(-scale, scale, digitNumber)[::-1];
    # images = np.zeros((digitSize * digitNumber, digitSize * digitNumber));
    # for i, x in enumerate(gridX):
    #     for j, y in enumerate(gridY):
    #         xHat = model.decode(np.array([[x, y]]), toProbability = True);
    #         images[j * digitSize : (j + 1) * digitSize, i * digitSize : (i + 1) * digitSize] = xHat.reshape(digitSize, digitSize);
    # plt.figure(figsize = (12, 14));
    # plt.imshow(images, cmap = "Greys_r");
    # plt.xlabel("M[0]");
    # plt.ylabel("M[1]");
    # plt.axis("off");
    # plt.show(block = True);
    #
    # L, R = 8, 3;
    # np.random.shuffle(X_test);
    # generatedData = model.generate(X_test[:10], L);
    # for k in range(len(generatedData)):
    #     figures = np.concatenate((X_test[k].reshape(1, -1), generatedData[k]), axis = 0);
    #     images = np.zeros((digitSize * R, digitSize * R));
    #     for i in range(R):
    #         for j in range(R):
    #             images[i * digitSize: (i + 1) * digitSize, j * digitSize: (j + 1) * digitSize] = figures[i * R + j].reshape(digitSize, digitSize);
    #     plt.figure(figsize = (12, 14));
    #     plt.imshow(images, cmap = "Greys_r");
    #     plt.axis("off");
    #     plt.show(block = True);

    lossFunc = BernoulliVAELoss();
    optimizer = Adam();
    trainingIterator = SequentialDataIterator([X_train], batchSize);
    model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, plot = True);

    with open(filename, "wb") as file:
        pickle.dump(model.params, file);

    print("exit.");


def testBernoulliVAE_Anomaly_MNIST():
    mnist = MNIST("/media/WindowsE/Data/MNIST", flatten = True, normalize = True);

    anomalyIndex, D = 9, mnist.trainX.shape[-1];
    filename = f"BernoulliVAE_Anomaly_MNIST_{anomalyIndex}.result";
    hiddenSize1, latentSize, batchSize, maxEpoch = 400, 200, 32, 10;
    model = BernoulliVAE(AggregateNetModule(
        AffineLayer(D, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, 2 * latentSize),
    ), AggregateNetModule(
        AffineLayer(latentSize, hiddenSize1),
        ReluLayer(),
        AffineLayer(hiddenSize1, D),
    ), latentSize);

    # X = mnist.trainX[mnist.trainY != anomalyIndex];
    # np.random.shuffle(X);
    # trainSize = int(len(X) * 0.8);
    # X_train, X_test_normal, X_test_anomaly = X[: trainSize], X[trainSize:], mnist.trainX[mnist.trainY == anomalyIndex];
    #
    # lossFunc = BernoulliVAELoss();
    # optimizer = Adam();
    # trainingIterator = SequentialDataIterator([X_train], batchSize);
    # model.fit(trainingIterator, lossFunc, optimizer, maxEpoch, plot = True);
    #
    # with open(filename, "wb") as file:
    #     pickle.dump((model.params, X_train, X_test_normal, X_test_anomaly), file);

    if os.path.isfile(filename):
        with open(filename, "br") as file:
            params, X_train, X_test_normal, X_test_anomaly = pickle.load(file);
        if isinstance(params[0], cp.ndarray):
            params = [cp.asnumpy(p) for p in params];
        for i in range(len(params)):
            params[i] = params[i].astype(defaultDType);
        model.params = params;

    M, V = model.encode(X_train);

    X_test_normal_rp = model.reconstructionProbability(X_test_normal);
    X_test_anomaly_rp = model.reconstructionProbability(X_test_anomaly);
    X_train_rp = np.concatenate(tuple([model.reconstructionProbability(item[0]) for item in SequentialDataIterator([X_train], batchSize, False)]), axis = 0);

    plt.figure(figsize = (12, 14));
    plt.hist(X_train_rp, bins = 1000, color = "k");
    plt.hist(X_test_normal_rp, bins = 1000, label = "normal", color = "b");
    plt.hist(X_test_anomaly_rp, bins = 1000, label = "anomaly", color = "r");
    plt.show(block = True);

    fprX, tprY = [], [];
    for alpha in np.arange(0, 1.001, 0.001):
        threshold = np.quantile(X_train_rp, alpha);
        X_test_normal_class = X_test_normal_rp < threshold;
        X_test_anomaly_class = X_test_anomaly_rp < threshold;

        TP, FP = np.sum(X_test_anomaly_class == True), np.sum(X_test_normal_class == True);
        FN, TN = np.sum(X_test_anomaly_class == False), np.sum(X_test_normal_class == False);
        precision, recall = TP / (TP + FP), TP / (TP + FN);
        fpr, tpr = FP / (FP + TN), recall;
        fprX.append(fpr);
        tprY.append(tpr);
        print(f"alpha = {alpha}, precision = {precision}, recall = {recall}, fpr = {fpr}, tpr = {tpr}");

    plt.figure(figsize = (12, 14));
    plt.plot(fprX, tprY);
    plt.xlabel("FPR");
    plt.ylabel("TPR");
    plt.title(f"AUC={auc(-X_test_anomaly_rp, -X_test_normal_rp)}");
    plt.show(block = True);

    print("exit.");


def testSeqAttentionDecoderGradient():
    N, T1, T2, vocabSize, vectorSize, hiddenSize = 32, 11, 12, 12, 14, 16;
    H, X = np.random.randn(N, T1, hiddenSize), np.random.randint(0, vocabSize, N * T2).reshape(N, T2);
    m = SeqAttentionDecoderLayer(vocabSize , vectorSize, hiddenSize);
    m.modules[0].params = [np.random.randn(vocabSize, vectorSize)];
    Y = m.forward(H, X)[0];
    dH1, dX1 = m.backward(np.ones_like(Y));
    dHN = numericGradient(lambda x: np.sum(m.forward(x, X)[0]), H);
    print(f"SeqAttentionDecoder, numericGradient, dH error: {np.sum(np.abs(dH1 - dHN))}");
    print("\n");


def testSeqBahdanauAttentionDecoderGradient():
    N, T1, T2, vocabSize, vectorSize, hiddenSize = 32, 11, 12, 13, 14, 24;
    H, X = np.random.randn(N, T1, hiddenSize), np.random.randint(0, vocabSize, N * T2).reshape(N, T2);
    # H, X = np.random.randn(N, T1, hiddenSize), np.arange(0, vocabSize).reshape(N, T2);
    m = SeqBahdanauAttentionDecoderLayer(vocabSize , vectorSize, hiddenSize);
    m.modules[0].params = [np.random.randn(vocabSize, vectorSize)];
    Y = m.forward(H, X)[0];
    dH1, dX1 = m.backward(np.ones_like(Y));
    dHN = numericGradient(lambda x: np.sum(m.forward(x, X)[0]), H);
    print(f"SeqAttentionDecoder, numericGradient, dH error: {np.sum(np.abs(dH1 - dHN))}");
    print("\n");


def testSeqTSEncoderLayerGradient():
    N, T, inputSize, hiddenSize = 32, 11, 16, 24;
    X = np.random.randn(N, T, inputSize);
    m = SeqTSEncoderLayer(inputSize, hiddenSize);
    Y1, Y2, Y3 = m.forward(X);
    dX1 = m.backward(np.ones_like(Y1), np.ones_like(Y2), np.ones_like(Y3))[0];
    dXN = numericGradient(lambda x: np.sum(np.concatenate(m.forward(x), axis = -1)), X);
    print(f"SeqTSEncoderLayer, numericGradient, dX error: {np.sum(np.abs(dX1 - dXN))}");
    print("\n");


def testSeqTSDecoderLayerGradient():
    N, T1, T2, inputSize, hiddenSize = 32, 11, 12, 16, 24;
    H1, H2, H3, X = np.random.randn(N, T1, hiddenSize), np.random.randn(N, T1, hiddenSize), np.random.randn(N, T1, hiddenSize), np.random.randn(N, T2, inputSize);
    m = SeqTSDecoderLayer(inputSize, hiddenSize);
    Y = m.forward(H1, H2, H3, X)[0];
    dH1, dH2, dH3, dX1 = m.backward(np.ones_like(Y));
    dH1N = numericGradient(lambda x: np.sum(m.forward(x, H2, H3, X)[0]), H1);
    dH2N = numericGradient(lambda x: np.sum(m.forward(H1, x, H3, X)[0]), H2);
    dH3N = numericGradient(lambda x: np.sum(m.forward(H1, H2, x, X)[0]), H3);
    dXN = numericGradient(lambda x: np.sum(m.forward(H1, H2, H3, x)[0]), X);
    print(f"SeqTSDecoderLayer, numericGradient, dH1 error: {np.sum(np.abs(dH1 - dH1N))}, dH2 error: {np.sum(np.abs(dH2 - dH2N))}, dH3 error: {np.sum(np.abs(dH3 - dH3N))}, dX error: {np.sum(np.abs(dX1 - dXN))}");
    print("\n");

    # N, T1, T2, inputSize, hiddenSize = 32, 11, 12, 16, 24;
    # H, X = np.random.randn(N, T1, hiddenSize), np.random.randn(N, T2, inputSize);
    # m = SeqTSDecoderLayer(inputSize, hiddenSize);
    # Y = m.forward(H, X)[0];
    # dH1, dX1 = m.backward(np.ones_like(Y));
    # dHN = numericGradient(lambda x: np.sum(m.forward(x, X)[0]), H);
    # dXN = numericGradient(lambda x: np.sum(m.forward(H, x)[0]), X);
    # print(f"SeqTSDecoderLayer, numericGradient, dH error: {np.sum(np.abs(dH1 - dHN))}, dX error: {np.sum(np.abs(dX1 - dXN))}");

    # N, T1, T2, inputSize, hiddenSize = 32, 11, 12, 16, 24;
    # H, X = np.random.randn(N, T1, hiddenSize), np.random.randn(N, T2, inputSize);
    # m1 = SeqTSBahdanauAttentionDecoderLayer(inputSize, hiddenSize);
    # Y1 = m1.forward(H, X)[0];
    # dH1, dX1 = m1.backward(np.ones_like(Y1));
    # m2 = SeqTSDecoderLayer(inputSize, hiddenSize);
    # m2.params = m1.params;
    # Y2 = m2.forward(H, X)[0];
    # dH2, dX2 = m2.backward(np.ones_like(Y2));
    #
    # print(np.sum(np.abs(Y1 - Y2)));
    # print(np.sum(np.abs(dH1 - dH2)));
    # print(np.sum(np.abs(dX1 - dX2)));
    # print("\n");
    #
    # for i in range(len(m1.grads)):
    #     print(np.sum(np.abs(m1.grads[i] - m2.grads[i])));


def testSeq2SeqTSModel1():
    N, T1, T2, inputSize, hiddenSize, outputSize = 2, 11, 12, 10, 32, 5;
    X, T = np.random.randn(N, T1, inputSize), np.random.randn(N, T2 + 1, outputSize);
    m = Seq2SeqTSModel_Seq2Seq_OnlyStateInput(inputSize, hiddenSize, outputSize);
    Y1, = m.forward(X, T);
    dX1, dT1 = m.backward(np.ones_like(Y1));
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, T)), X);
    dTN = numericGradient(lambda x: sumAll(*m.forward(X, x)), T);
    print(f"Seq2SeqTSModel, numericGradient, dX error: {np.sum(np.abs(dX1 - dXN))}, dT error: {np.sum(np.abs(dT1 - dTN))}");
    testModuleGradient(m, "Seq2SeqTSModel, numericGradient", X, T);
    print("\n");


def testSeq2SeqTSModel2():
    N, T1, T2, inputSize, hiddenSize, outputSize = 32, 11, 12, 10, 32, 5;
    X, T = np.random.randn(N, T1, inputSize), np.random.randn(N, T2, outputSize);
    m = Seq2SeqTSModel_Seq2Seq_OnlyStateInput(inputSize, hiddenSize, outputSize);
    Y1, = m.forward(X, T);
    dX1 = m.backward(np.ones_like(Y1))[0];
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, T)), X);
    print(f"Seq2SeqTSModel_Seq2Seq_OnlyStateInput, numericGradient, dX error: {np.sum(np.abs(dX1 - dXN))}");
    testModuleGradient(m, "Seq2SeqTSModel_Seq2Seq_OnlyStateInput, numericGradient", X, T);
    print("\n");


def testSeq2SeqTSModel_Dropout():
    N, T1, T2, inputSize, hiddenSize, outputSize = 32, 11, 12, 16, 24, 14;
    X, T = np.random.randn(N, T1, inputSize), np.random.randn(N, T2, outputSize);
    m = Seq2SeqTSModel(inputSize, hiddenSize, outputSize, inputDropout = 0.5, recurrentDropout = 0.5);
    m.context.isTrainingMode = True;
    Y1, = m.forward(X, T);
    dX1 = m.backward(np.ones_like(Y1))[0];
    dXN = numericGradient(lambda x: sumAll(*m.forward(x, T)), X);
    print(f"Seq2SeqTSModel dropout, numericGradient, dX error: {np.sum(np.abs(dX1 - dXN))}");
    testModuleGradient(m, "Seq2SeqTSModel dropout, numericGradient", X, T);
    print("\n");


if __name__ == "__main__":
    test();
