import numpy as np

""" INTER_FUNCTION """


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.add(a, b)


def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.subtract(a, b)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.multiply(a, b)


def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b)


def power(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.power(a, b)


""" BASIC_FUNCTION """

def mean(x: np.ndarray) -> np.ndarray:
    return np.mean(x)


def std(x: np.ndarray) -> np.ndarray:
    return x.std()


def var(x: np.ndarray) -> np.ndarray:
    return x.var()


def max(x: np.ndarray) -> np.ndarray:
    return x.max()


def min(x: np.ndarray) -> np.ndarray:
    return x.min()
