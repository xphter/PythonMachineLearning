import numpy as np;

from WeibullDistribution import *;


def test():
    n = 100;
    beta0, eta0 = 2.5, 3.5;
    x = eta0 * np.random.weibull(beta0, n);
    print(f"β0 = {beta0}, η0 = {eta0}");

    wd1 = WeibullDistribution();
    wd1.fit(x, method = WeibullDistributionFitMethod.OLS);
    print(f"β1 = {wd1.beta}, η1 = {wd1.eta}");

    wd2 = WeibullDistribution();
    wd2.fit(x, method = WeibullDistributionFitMethod.MLE);
    print(f"β2 = {wd2.beta}, η2 = {wd2.eta}");


if __name__ == '__main__':
    test();
