import numpy as np;
import numpy.linalg as npl;

import DataHelper;


# # each column is a sample
# def projectData(dataSet, percent):
#     count = dataSet.shape[1];
#     cov = np.matmul(dataSet, dataSet.T) / count;
#     u, s, v = npl.svd(cov);
#
#     p, totalVar = None, sum(s);
#
#     for i in range(0, len(s)):
#         if sum(s[0:i+1]) >= percent * totalVar:
#             p = u[:, 0:i+1];
#             break;
#
#     result = np.matmul(p.T, dataSet);
#
#     return result, p;


def projectData(X, percent):
    X, mu, sigma = DataHelper.normalizeFeatures(X);

    count = X.shape[0];
    cov = (X.T * X) / count;
    u, s, v = npl.svd(cov);

    p, totalVar = None, sum(s);

    for i in range(0, len(s)):
        if sum(s[0:i+1]) >= percent * totalVar:
            p = u[:, 0:i+1];
            break;

    result = X * p;

    return X, mu, sigma, result, p;


def performProject(X, mu, sigma, p):
    Y = DataHelper.performNormalize(X, mu, sigma);

    return Y * p;


def recoverData(X, p):
    return X * p.T;


