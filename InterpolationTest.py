import numpy as np;

from Interpolation import *;


def test():
    n, sd = 10, 100;
    x = np.random.randn(n) * sd;
    # x = np.arange(n);
    x.sort();
    y = np.random.randn(n) * sd;

    cs = CubicSpline();
    cs.fit(x, y);

    cs.predict(np.random.rand(200) * (x[-1] - x[0]) + x[0], plot = True);
    # cs.predict(np.concatenate((np.arange(x[0], x[-1], 1), x)), plot = True);
