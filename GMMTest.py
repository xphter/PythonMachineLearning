import time;
import datetime;
import os.path;

from Functions import *;
from NN import *;
from Random import *;
from GMM import *;
import numpy as np;
import matplotlib;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;


SAMPLE_SIZE = int(1e4);
FIT_SIZE = int(1e5);
MAX_EPOCH = 1;


def _mae(p0 : np.ndarray, p1 : np.ndarray):
    return float(np.sum(np.abs(p0 - p1))) / p0.size;


def plotSample(X : np.ndarray, K : int):
    D = X.shape[-1];

    if D == 1:
        plt.figure(1, (14, 8));
        plt.hist(X, bins = 1000);
        plt.title(f"sample histogram, K = {K}");
        plt.show(block = True);
        plt.close();
    elif D == 2:
        plt.figure(1, (14, 8));
        plt.scatter(X[:, 0], X[:, 1]);
        plt.title(f"sample scatter diagram, K = {K}");
        plt.show(block = True);
        plt.close();
    elif D == 3:
        fig = plt.figure(1, (14, 8));
        ax = plt.axes(projection = "3d");
        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2]);
        plt.title(f"sample 3D diagram, K = {K}");
        plt.show(block = True);
        plt.close();


def plotGMM(X : np.ndarray, gmm : GaussianMixture, alpha : float = 0.001, title : str = None, filename : str = None):
    D = X.shape[-1];
    P = gmm.pdf(X);
    Q = np.quantile(P, alpha);
    A = X[P <= Q];

    if D == 1:
        fig = plt.figure(1, (14, 8));
        ax1 = fig.add_subplot(111);
        ax1.set_xlabel("bins");
        ax1.set_ylabel('hist');
        bins = ax1.hist(X, bins = 1000)[1];
        ax2 = ax1.twinx();
        ax2.set_ylabel('pdf');
        ax2.plot(bins, bp := gmm.pdf(bins.reshape(-1, 1)), "r", label = "pdf");
        for mu in gmm.mu:
            ax2.axvline(x = mu[0], ymin = 0, ymax = np.amax(bp) / 5, c = "red", lw = 5);
        for a in A:
            ax1.axvline(x = a, ymin = 0, ymax = np.amax(bp) / 5, c = "orange");
        plt.title(title if title is not None else f"histogram and pdf, K = {gmm.K}");
        if filename is not None:
            plt.savefig(filename);
        else:
            plt.show(block = True);
        plt.close();
    elif D == 2:
        x, y = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100));
        z = np.zeros_like(x);
        for i in range(len(x)):
            for j in range(x.shape[-1]):
                z[i, j] = gmm.pdf(np.array([x[i, j], y[i, j]]).reshape(1, 2));

        fig = plt.figure(1, (14, 8));
        plt.scatter(X[:, 0], X[:, 1]);
        plt.scatter(A[:, 0], A[:, 1], c = "orange");
        plt.contour(x, y, z);
        for mu in gmm.mu:
            plt.scatter(mu[0], mu[1], s = 100, marker = "x", c = "red");
        plt.title(title if title is not None else f"scatter and contour, K = {gmm.K}");
        if filename is not None:
            plt.savefig(filename);
        else:
            plt.show(block = True);
        plt.close();

    if filename is None:
        plt.figure(1, (14, 8));
        plt.hist(P, bins = 1000);
        plt.title("pdf histogram");
        plt.show(block = True);
        plt.close();


def testSample(N : int, D : int, K : int, w : np.ndarray = None, m : np.ndarray = None, s : np.ndarray = None):
    weight = softmax(np.random.randn(K)) if w is None else w;
    mu = np.random.randn(K * D).reshape(K, D) if m is None else m;
    sigma = np.concatenate(tuple([randomPDM(D) for k in range(K)]), axis = 0).reshape(K, D, D) if s is None else s;

    gmm = GaussianMixture(weight, mu, sigma);
    X = gmm.sample(N);

    plotSample(X, K);


def testFit(N : int, D : int, K : int, sample : np.ndarray = None, w : np.ndarray = None, m : np.ndarray = None, s : np.ndarray = None, maxEpoch : int = 1):
    weight0 = softmax(np.random.randn(K)) if w is None else w;
    mu0 = np.random.randn(K * D).reshape(K, D) if m is None else m;
    sigma0 = np.concatenate(tuple([randomPDM(D) for k in range(K)]), axis = 0).reshape(K, D, D) if s is None else s;
    X = GaussianMixture(weight0, mu0, sigma0).sample(N) if sample is None else sample;

    plotSample(X, K);

    gmm = GaussianMixture();
    valuesList = gmm.fit(X, K, maxEpoch = maxEpoch, maxIteration = 1000, outDebugInfo = True, verbose = True);

    if sample is None:
        print(f"actual weight: {weight0}\n\nestimate weight: {gmm.weight}\n");
        print(f"actual mu: {mu0}\n\nestimate mu: {gmm.mu}\n");
        print(f"actual sigma: {sigma0}\n\nestimate sigma: {gmm.sigma}\n");
        print(f"{D}D, {K}K weight MAE: {_mae(weight0, gmm.weight)}, mu MAE: {_mae(mu0, gmm.mu)}, sigma MAE: {_mae(sigma0, gmm.sigma)}");

    plotGMM(X, gmm);

    plt.figure(1, (14, 8));
    for qValues, pValues in valuesList:
        plt.plot(qValues);
    plt.title("value of Q function");
    plt.show(block = True);
    plt.close();

    plt.figure(1, (14, 8));
    for qValues, pValues in valuesList:
        plt.plot(pValues);
    plt.title("log-likelihood");
    plt.show(block = True);
    plt.close();


def testOptimalK(N : int, D : int, K : int, maxK : int, sample : np.ndarray = None, w : np.ndarray = None, m : np.ndarray = None, s : np.ndarray = None, maxEpoch : int = 1):
    weight0 = softmax(np.random.randn(K)) if w is None else w;
    mu0 = np.random.randn(K * D).reshape(K, D) if m is None else m;
    sigma0 = np.concatenate(tuple([randomPDM(D) for k in range(K)]), axis = 0).reshape(K, D, D) if s is None else s;
    X = GaussianMixture(weight0, mu0, sigma0).sample(N) if sample is None else sample;

    plotSample(X, K);

    gmm = GaussianMixture.optimalK(X, maxK, maxEpoch = maxEpoch, maxIteration = 1000, outDebugInfo = False, verbose = True);
    print(f"the optimal K is {gmm.K}");

    if sample is None:
        print(f"actual weight: {weight0}\n\nestimate weight: {gmm.weight}\n");
        print(f"actual mu: {mu0}\n\nestimate mu: {gmm.mu}\n");
        print(f"actual sigma: {sigma0}\n\nestimate sigma: {gmm.sigma}\n");
        print(f"{D}D, {K}K weight MAE: {_mae(weight0, gmm.weight)}, mu MAE: {_mae(mu0, gmm.mu)}, sigma MAE: {_mae(sigma0, gmm.sigma)}");

    plotGMM(X, gmm);


def testSample1D_1K():
    testSample(SAMPLE_SIZE, 1, 1);


def testSample1D_3K():
    testSample(SAMPLE_SIZE, 1, 3, w = np.array([0.4, 0.2, 0.4]), m = np.array([[-3], [0], [3]]));


def testSample2D_1K():
    testSample(SAMPLE_SIZE, 2, 1);


def testSample2D_3K():
    testSample(SAMPLE_SIZE, 2, 3, w = np.array([0.4, 0.2, 0.4]), m = np.array([[-3, -3], [0, 0], [3, 3]]));


def testSample3D_1K():
    testSample(SAMPLE_SIZE, 3, 1);


def testSample3D_3K():
    testSample(SAMPLE_SIZE, 3, 3, w = np.array([0.4, 0.2, 0.4]), m = np.array([[-3, -3, -3], [0, 0, 0], [3, 3, 3]]));


def testFit1D_1K():
    testFit(FIT_SIZE, 1, 1, maxEpoch = MAX_EPOCH);


def testFit1D_3K():
    testFit(FIT_SIZE, 1, 3, w = np.array([0.4, 0.2, 0.4]), m = np.array([[-3], [0], [3]]), maxEpoch = MAX_EPOCH);


def testFit1D_3K_Random():
    testFit(FIT_SIZE, 1, 3, maxEpoch = MAX_EPOCH);


def testFit1D_Exp():
    testFit(FIT_SIZE, 1, 6, sample = np.random.exponential(1/2, FIT_SIZE).reshape(-1, 1), maxEpoch = MAX_EPOCH);


def testFit1D_Chisq():
    testFit(FIT_SIZE, 1, 4, sample = np.random.chisquare(10, FIT_SIZE).reshape(-1, 1), maxEpoch = MAX_EPOCH);


def testFit2D_1K():
    testFit(FIT_SIZE, 2, 1, maxEpoch = MAX_EPOCH);


def testFit2D_3K():
    testFit(FIT_SIZE, 2, 3, w = np.array([0.4, 0.2, 0.4]), m = np.array([[-3, -3], [0, 0], [3, 3]]), maxEpoch = MAX_EPOCH);


def testFit2D_3K_Random():
    testFit(FIT_SIZE, 2, 3, maxEpoch = MAX_EPOCH);


def testFit3D_1K():
    testFit(FIT_SIZE, 3, 1, maxEpoch = MAX_EPOCH);


def testFit3D_3K():
    testFit(FIT_SIZE, 3, 3, w = np.array([0.4, 0.2, 0.4]), m = np.array([[-3, -3, -3], [0, 0, 0], [3, 3, 3]]), maxEpoch = MAX_EPOCH);


def testFit3D_3K_Random():
    testFit(FIT_SIZE, 3, 3, maxEpoch = MAX_EPOCH);


def testFitND_2NK_Random(D : int):
    testFit(FIT_SIZE, D, 2 * D, maxEpoch = MAX_EPOCH);


def testOptimalK1D_3K():
    testOptimalK(FIT_SIZE, 1, 3, 6, w = np.array([0.4, 0.2, 0.4]), m = np.array([[-3], [0], [3]]), maxEpoch = MAX_EPOCH);


def testOptimalK1D_3K_Random():
    testOptimalK(FIT_SIZE, 1, 3, 6, maxEpoch = MAX_EPOCH);


def testOptimalK1D_Exp():
    testOptimalK(FIT_SIZE, 1, 3, 12, sample = np.random.exponential(1/2, FIT_SIZE).reshape(-1, 1), maxEpoch = MAX_EPOCH);


def testOptimalK1D_Chisq():
    testOptimalK(FIT_SIZE, 1, 3, 12, sample = np.random.chisquare(5, FIT_SIZE).reshape(-1, 1), maxEpoch = MAX_EPOCH);


def testOptimalK2D_3K():
    testOptimalK(FIT_SIZE, 2, 3, 6, w = np.array([0.4, 0.2, 0.4]), m = np.array([[-3, -3], [0, 0], [3, 3]]), maxEpoch = MAX_EPOCH);


def testOptimalK3D_3K():
    testOptimalK(FIT_SIZE, 3, 3, 6, w = np.array([0.4, 0.2, 0.4]), m = np.array([[-3, -3, -3], [0, 0, 0], [3, 3, 3]]), maxEpoch = MAX_EPOCH);


def loadTagData(tagNames : List[str], remove : bool = False, plot : bool = False) -> Tuple[np.ndarray, np.ndarray]:
    data, marks = [], [];

    for tagName in tagNames:
        d = np.load(f"/media/WindowsE/Data/PARS/JNLH/AiModel/isys_data_20210701_20220516_180/1CH/__JNRTDB_{tagName}.npy");
        data.append(d[:, 0]);
        marks.append(d[:, 1]);

    data, marks = np.column_stack(tuple(data)), np.column_stack(tuple(marks));
    data1, data2 = data[np.all(marks, axis = -1)], None;

    if remove and len(data1) > 0:
        Q1, Q3 = np.quantile(data1, 0.25, axis = 0), np.quantile(data1, 0.75, axis = 0);
        data2 = data1[np.all(np.logical_and(data1 >= Q1 - 5 * (Q3 - Q1), data1 <= Q3 + 5 * (Q3 - Q1)), axis = -1)];

        if len(data2) < 0.85 * len(data1):
            data2 = data1;
    else:
        data2 = data1;

    if plot:
        D = data1.shape[-1];

        if D == 1:
            plt.figure(1, (14, 8));
            plt.hist(data1, bins = 1000);
            if data2 is not None:
                plt.hist(data2, bins = 1000);
            plt.title("sample histogram");
            plt.show(block = True);
            plt.close();
        elif D == 2:
            plt.figure(1, (14, 8));
            plt.scatter(data1[:, 0], data1[:, 1]);
            if data2 is not None:
                plt.scatter(data2[:, 0], data2[:, 1]);
            plt.title("scatter histogram");
            plt.show(block = True);
            plt.close();

    return data1, data2;


def testTagData(tagNames : List[str], maxK : int = None, K : int = None, maxEpoch : int = 1, plot : bool = False, filename : str = None):
    title = ', '.join(tagNames);

    Z, X = loadTagData(tagNames, remove = True, plot = plot);

    if len(X) == 0:
        print(f"ignore {title} due to empty data");
        return;
    if np.all(X == X[0]):
        print(f"ignore {title} due to duplicate data");
        return;

    if maxK is None:
        maxK = int((20 if X.shape[-1] == 1 else 10) ** len(tagNames));

    if K is None:
        gmm = GaussianMixture.optimalK(X, maxK, maxEpoch = maxEpoch, maxIteration = 1000, outDebugInfo = False, verbose = True);
        print(f"the maxK is {maxK}, the optimal K of {title} is {gmm.K}");
    else:
        gmm = GaussianMixture();
        gmm.fit(X, K, maxEpoch = maxEpoch, maxIteration = 1000, outDebugInfo = False, verbose = False);
        print(f"the fix K of {title} is {gmm.K}");
    print(f"the weight of {title} is: {gmm.weight}");

    plotGMM(X, gmm, title = f"{title} histogram and pdf, K = {gmm.K}", filename = filename);

    # p = gmm.pdf(Z);
    # Q1000 = np.quantile(p, 0.001);
    # np.save("anomaly_data.npy", Z[p <= Q1000]);


def testOneTag():
    with open("/media/WindowsE/Data/PARS/JNLH/AiModel/isys_data_20210701_20220516_180/1CH.txt", "rt", encoding = "utf-8") as file:
        tagNames = [line.strip() for line in file.readlines()];

    count = len(tagNames);
    for i, tagName in enumerate(tagNames):
        if len(tagName) == 0:
            continue;

        if os.path.isfile(filename := f"{tagName}.png"):
            continue;

        try:
            testTagData([tagName], maxEpoch = MAX_EPOCH, plot = False, filename = filename);
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} success to fit {tagName} {i + 1}/{count}");
        except Exception as ex:
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} fail to fit {tagName} {i + 1}/{count}: {ex}");


if __name__ == '__main__':
    startTime = time.time();

    # testSample1D_1K();
    # testSample1D_3K();
    # testSample2D_1K();
    # testSample2D_3K();
    # testSample3D_1K();
    # testSample3D_3K();

    # testFit1D_1K();
    # testFit1D_3K();
    # testFit1D_Exp();
    # testFit1D_Chisq();
    # testFit1D_3K_Random();
    # testFit2D_1K();
    # testFit2D_3K();
    # testFit2D_3K_Random();
    # testFit3D_1K();
    # testFit3D_3K();
    # testFit3D_3K_Random();
    # testFitND_2NK_Random(4);

    # testOptimalK1D_3K();
    # testOptimalK1D_Exp();
    # testOptimalK1D_Chisq();
    # testOptimalK2D_3K();
    # testOptimalK3D_3K();

    testTagData(["PI6217.PV"], maxEpoch = MAX_EPOCH, plot = True);
    # testTagData(["FIC6270.PV", "TIC6208.PV"], maxEpoch = MAX_EPOCH, plot = True);
    # testTagData(["TI6113A.PV", "TI6116A.PV", "TIC6115.PV", "AI6103.PV"], maxEpoch = MAX_EPOCH);
    # testOneTag();

    print(f"elapsed time: {time.time() - startTime}");
