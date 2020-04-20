import abc;
import math;


class OptimizerBase(metaclass = abc.ABCMeta):
    def __init__(self, epsilon):
        if epsilon is None or epsilon <= 0:
            raise ValueError();

        self._target = None;
        self._epsilon = epsilon;


    def setTarget(self, target):
        if target is None or not isinstance(target, IOptimizerTarget):
            raise ValueError();

        self._target = target;

    @abc.abstractmethod
    def search(self, theta):
        pass;


class IOptimizerTarget(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def getTargetValue(self, theta):
        pass;


class IGradientProvider(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def getGradient(self, theta):
        pass;


class IHessianMatrixProvider(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def getHessianMatrix(self, theta):
        pass;


class GradientDescent(OptimizerBase):
    def __init__(self, epsilon, alpha = None):
        if alpha is not None and alpha <= 0:
            raise ValueError();

        super().__init__(epsilon);

        self.__alpha = alpha;


    def __derivativeOfAlpha(self, alpha, theta, gradientValue):
        return -(self._target.getGradient(theta - alpha * gradientValue).T * gradientValue)[0, 0];


    def __findAlpha(self, theta, gradientValue, epsilon):
        t, a, b = 2, 0, 1;
        while self.__derivativeOfAlpha(b, theta, gradientValue) < 0:
            a, b = b, b * t;

        previous = 0;
        alpha = (a + b) / 2;
        derivative = self.__derivativeOfAlpha(alpha, theta, gradientValue);
        while math.fabs(derivative) > epsilon and b > a and derivative != previous:
            a, b = (alpha, b) if derivative < 0 else (a, alpha);

            alpha = (a + b) / 2;
            previous = derivative;
            derivative = self.__derivativeOfAlpha(alpha, theta, gradientValue);

        return alpha;


    def setTarget(self, target):
        if target is None or not isinstance(target, IGradientProvider):
            raise ValueError();

        super().setTarget(target);


    def search(self, theta):
        alpha = None;
        iterationCount = 0;
        previousValue, functionValue = None, None;
        gradientValue, gradientLength = None, None;

        while True:
            iterationCount += 1;

            functionValue = self._target.getTargetValue(theta);
            gradientValue = self._target.getGradient(theta);
            gradientLength = math.sqrt((gradientValue.T * gradientValue)[0, 0]);
            print("{0}\tvalue of function: {1}\tlength of gradient: {2}".format(iterationCount, functionValue, gradientLength));

            if gradientLength <= self._epsilon:
                break;

            alpha = self.__alpha if self.__alpha is not None else self.__findAlpha(theta, gradientValue, self._epsilon);
            if previousValue is not None and math.fabs(functionValue - previousValue) <= self._epsilon or alpha * gradientLength <= self._epsilon:
                break;

            if self.__alpha is not None and previousValue is not None and functionValue - previousValue > 0:
                self.__alpha *= 0.1;

            previousValue = functionValue;
            theta -= alpha * gradientValue;

        return theta, functionValue, gradientValue;


class NewtonMethod(OptimizerBase):
    def __init__(self, epsilon):
        super().__init__(epsilon);


    def setTarget(self, target):
        if target is None or not isinstance(target, IGradientProvider) or not isinstance(target, IHessianMatrixProvider):
            raise ValueError();

        super().setTarget(target);


    def search(self, theta):
        delta = None;
        iterationCount = 0;
        previousValue, functionValue = None, None;
        gradientValue, gradientLength = None, None;

        while True:
            iterationCount += 1;

            functionValue = self._target.getTargetValue(theta);
            gradientValue = self._target.getGradient(theta);
            gradientLength = math.sqrt((gradientValue.T * gradientValue)[0, 0]);
            print("{0}\tvalue of function: {1}\tlength of gradient: {2}".format(iterationCount, functionValue, gradientLength));

            if gradientLength <= self._epsilon:
                break;

            delta = self._target.getHessianMatrix(theta).I * gradientValue;
            if previousValue is not None and math.fabs(functionValue - previousValue) <= self._epsilon or math.sqrt((delta.T * delta)[0, 0]) <= self._epsilon:
                break;

            previousValue = functionValue;
            theta -= delta;

        return theta, functionValue, gradientValue;
