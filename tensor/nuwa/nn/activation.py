import numpy as np


class Activation:
    """
    Activation function
    It is a wrapper around the activation function that can be used in the optim.btree

    The child class of Activation should implement the gradient function: grad and __call__ function.
    It should also need to specify the name of activation
    """
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def grad(self, *args, **kwargs):
        raise NotImplementedError


class ReLU(Activation):
    """
    ReLu activation function: f = max(0., X)
    """
    def __init__(self):
        super(ReLU, self).__init__("ReLU")

    def __call__(self, X: np.ndarray):
        return np.maximum(0., X)

    def grad(self, X, grad):
        grad[np.where(X < 0)] = 0
        return grad


"""implemented activation functions"""
ACTIVATIONS = {"ReLU": ReLU}
