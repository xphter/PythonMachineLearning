import inspect;


class Event:
    def __init__(self, name):
        self.name = name;
        self.__listeners = [];


    def __repr__(self):
        return self.name;


    def __str__(self):
        return self.__repr__();


    def addListener(self, listener):
        if listener is None or not inspect.isfunction(listener):
            raise ValueError();

        if listener in self.__listeners:
            return;

        self.__listeners.append(listener);


    def removeListener(self, listener):
        if listener is None or not inspect.isfunction(listener):
            raise ValueError();

        if listener not in self.__listeners:
            return;

        self.__listeners.remove(listener);


    def trigger(self, *args):
        result = [];

        for listener in self.__listeners:
            result.append(listener(*args));

        return tuple(result);



