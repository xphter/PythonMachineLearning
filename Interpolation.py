import enum;
import numpy as np;
import matplotlib.pyplot as plt;

from typing import Union, List;


"""
Cubic Spline Interpolation
inputs:
a = x_0 ≤ x_1 ≤ ... ≤ x_n-1 ≤ x_n = b and y_i, i ∈ [0, n]
h_i = x_i+1 - x_i, i ∈ [0, n]

cubic functions:
f_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3, i ∈ [0, n - 1]
a_i = y_i
b_i = (y_i+1 - y_i) / h_i - (h_i / 2)m_i - (h_i / 6)(m_i+1 - m_i)
c_i = m_i / 2
d_i = (m_i+1 - m_i) / (6 h_i)

linear equations:
h_i m_i + 2(h_i + h_i+1)m_i+1 + h_i+1 m_i+2 = 6 ((y_i+2 - y_i+1) / h_i+1 - (y_i+1 - y_i) / h_i), , i ∈ [0, n - 2]

natural boundary: 
m_0 = 0
m_n = 0

not-a-knot boundary: 
-h_1 m_0 + (h_0 + h_1) m_1 - h_0 m_2 = 0
-h_n-1 m_n-2 + (h_n-2 + h_n-1) m_n-1 - h_n-2 m_n = 0

"""

class CubicSplineBoundary(enum.IntEnum):
    Natural = 0x10;
    NotAKnot = 0x30;


class CubicSpline:
    def __init__(self, x : np.ndarray = None, y : np.ndarray = None, W : np.ndarray = None):
        self._x = x;
        self._y = y;
        self._W = W;


    @property
    def x(self) -> np.ndarray:
        return self._x;


    @property
    def y(self) -> np.ndarray:
        return self._y;


    @property
    def W(self) -> np.ndarray:
        return self._W;


    def _checkError(self, n : int, h : np.ndarray, boundary : CubicSplineBoundary):
        if n < 1:
            raise ValueError("at least 2 points are required");
        if boundary == CubicSplineBoundary.NotAKnot and n < 3:
            raise ValueError("a minimum of 4 points are required when using Not-A-Knot boundary");

        if np.any(h == 0):
            raise ValueError("x has duplicated values");


    def fit(self, x : Union[List[float], np.ndarray], y : Union[List[float], np.ndarray], boundary : CubicSplineBoundary = CubicSplineBoundary.NotAKnot):
        n, x, y = len(x) - 1, np.array(x), np.array(y);
        idx = np.argsort(x);
        x, y = x[idx], y[idx];
        h = np.diff(x);

        self._checkError(n, h, boundary);

        idx = np.arange(n - 1);
        h0, h1 = h[0: n - 1], h[1: n];
        y0, y1, y2 = y[0: n - 1], y[1: n], y[2: n + 1];

        A = np.zeros((n + 1, n + 1));
        A[idx, idx] = h0;
        A[idx, idx + 1] = 2 * (h0 + h1);
        A[idx, idx + 2] = h1;

        m = None;
        b = np.zeros(n + 1);
        b[idx] = 6 * ((y2 - y1) / h1 - (y1 - y0) / h0);

        if boundary == CubicSplineBoundary.Natural:
            A[n - 1, 0] = 1;
            A[n, n] = 1;

            m = np.linalg.inv(A) @ b;
        elif boundary == CubicSplineBoundary.NotAKnot:
            A[n - 1, np.arange(3)] = np.array([-h[1], h[0] + h[1], -h[0]]);
            A[n, np.arange(n - 2, n + 1)] = np.array([-h[n - 1], h[n - 2] + h[n - 1], -h[n - 2]]);

            try:
                m = np.linalg.solve(A, b);
            except np.linalg.LinAlgError:
                m = np.linalg.lstsq(A, b);
        else:
            raise ValueError(f"invalid boundary: {boundary}");

        h0 = h[0: n];
        y0, y1 = y[0: n], y[1: n + 1];
        m0, m1 = m[0: n], m[1: n + 1];

        W = np.zeros((4, n));
        W[0] = y0;  # a_i
        W[1] = (y1 - y0) / h0 - h0 * m0 / 2 - h0 * (m1 - m0) / 6;  # b_i
        W[2] = m0 / 2;  # c_i
        W[3] = (m1 - m0) / (6 * h0);  # d_i

        self._x = x;
        self._y = y;
        self._W = W;


    def predict(self, x : Union[List[float], np.ndarray], plot : bool = False) -> (np.ndarray, np.ndarray):
        x = np.array(x);
        includeMax = np.any(x == self._x[-1]);
        x = np.unique(x[np.logical_and(x >= self._x[0], x < self._x[-1])]);
        y = np.zeros_like(x);

        a, j = [], 0;
        idx = len(x) - 1;
        for i in range(len(x)):
            a.append(v := x[i]);

            while v >= self._x[j + 1]:
                j += 1;

            if i == idx or x[i + 1] >= self._x[j + 1]:
                z = np.array(a) - self._x[j];
                y[i + 1 - len(a): i + 1] = np.vstack((np.ones_like(z), z, z ** 2, z ** 3)).T @ self._W[:, j];

                j += 1;
                a.clear();

        if includeMax:
            x = np.append(x, self._x[-1]);
            y = np.append(y, self._y[-1]);

        if plot:
            plt.figure(1);
            plt.plot(x, y, "-ok");
            plt.scatter(self._x, self._y, marker = "x", c = "red", linewidths = 20);
            plt.show(block = True);
            plt.close();

        return x, y;
