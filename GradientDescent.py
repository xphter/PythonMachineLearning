import math;


class GradientDescent:
    def __init__(self, valueProvider, gradientProvider):
        if valueProvider is None or gradientProvider is None:
            raise ValueError();

        self.__currentValue = None;
        self.__currentGradient = None;
        self.__valueProvider = valueProvider;
        self.__gradientProvider = gradientProvider;


    def search(self, theta, alpha, minDifference, maxCount):
        if minDifference is None and maxCount is None:
            raise ValueError();

        functionValue = None;
        iterationCount = 0;

        while True:
            iterationCount += 1;

            self.__currentValue = self.__valueProvider(theta);
            self.__currentGradient = self.__gradientProvider(theta);
            print("{0}\tfunction value: {1}".format(iterationCount, self.__currentValue));

            if maxCount is not None and iterationCount >= maxCount or \
                    minDifference is not None and functionValue is not None and abs(self.__currentValue - functionValue) <= minDifference:
                break;

            if functionValue is not None and self.__currentValue - functionValue > 0:
                alpha *= 0.1;

            functionValue = self.__currentValue;
            theta -= alpha * self.__currentGradient;

        return theta, self.__currentValue, self.__currentGradient;
