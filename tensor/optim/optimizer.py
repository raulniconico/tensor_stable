from typing import Any

import numpy as np
from numba import njit
from numpy import ndarray, dtype, floating, unsignedinteger

from .optim_btree import Optim_Btree
from ..nuwa.util import tensorise, matricise


class Optimizer:
    NAME = ["SGD", "Adam", "RMSprop"]

    def __init__(self, btree: Optim_Btree, name: str, dtype=np.float64) -> None:
        self.btree = btree
        self.name = name
        self.dtype = dtype

    @property
    def btree(self):
        return self._btree

    @btree.setter
    def btree(self, btree: Optim_Btree):
        self._btree = btree

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        assert name in self.NAME, "Optimizer name must be one of: {}".format(self.NAME)
        self._name = name

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype=np.float64):
        self._dtype = dtype

    def init(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, btree: Optim_Btree,
                 lr: float,
                 momentum: float = 0.9,
                 decay: bool = False,
                 decay_rate: float = 1,
                 decay_min: float=1e-4):

        super().__init__(btree=btree, name="SGD")
        self._lr_0 = lr
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.decay_rate = decay_rate
        self.decay_min = decay_min

        self.nodes = None
        self.params = None
        self.momentum_buffer = None
        self.grad_buffer = None
        self.iter = None


        self.init()

    @property
    def iter(self):
        return self._iter

    @iter.setter
    def iter(self, iter: int):
        self._iter = iter

    @property
    def lr(self):
        if self.decay:
            lr = self._lr_0 * self.decay_rate / np.sqrt(self.iter)
        else:
            lr = self._lr_0

        if lr < self.decay_min:
            lr = self.decay_min
        self.lr = lr
        return lr

    @lr.setter
    def lr(self, lr: float):
        self._lr = lr

    def init(self):

        self.iter = 0
        self.params = []
        self.grad_buffer = []
        self.momentum_buffer = []

        self.nodes = self.btree.get_non_leaf()

        for i, node in enumerate(self.nodes):
            B_t = node.get_cache("B_t")
            self.momentum_buffer.append([np.zeros(B_t.shape).astype(self.dtype)])

            if self.btree.BN and node.is_inner():
                Beta = node.get_cache("Beta")
                Gamma = node.get_cache("Gamma")
                self.momentum_buffer[-1].append(np.zeros(Beta.shape).astype(self.dtype))
                self.momentum_buffer[-1].append(np.zeros(Gamma.shape).astype(self.dtype))

    def step(self):
        self.iter += 1
        for i, node in enumerate(self.nodes):
            B_t = node.get_cache("B_t")
            B_t_grad = node.get_cache("B_t_grad")

            B_t, self.momentum_buffer[i][0] = self.SGD(B_t,
                                                       B_t_grad,
                                                       self.lr,
                                                       self.momentum_buffer[i][0],
                                                       self.momentum)
            node.update_cache({"B_t": B_t})

            if self.btree.BN and node.is_inner():
                Beta = node.get_cache("Beta")
                Beta_grad = node.get_cache("Beta_grad")
                Gamma = node.get_cache("Gamma")
                Gamma_grad = node.get_cache("Gamma_grad")
                Beta, self.momentum_buffer[i][1] = self.SGD(Beta,
                                                            Beta_grad,
                                                            self.lr,
                                                            self.momentum_buffer[i][1],
                                                            self.momentum)
                Gamma, self.momentum_buffer[i][2] = self.SGD(Gamma,
                                                             Gamma_grad,
                                                             self.lr,
                                                             self.momentum_buffer[i][2],
                                                             self.momentum)
                node.update_cache({"Beta": Beta, "Gamma": Gamma})

    def zero_grad(self):
        for i, buffer in enumerate(self.grad_buffer):
            self.grad_buffer[i] = np.zeros(buffer.shape)

    @staticmethod
    @njit(fastmath=True, parallel=True, nogil=True)
    def SGD(param: np.ndarray,
            grad: np.ndarray,
            lr: float,
            momentum_buffer: np.ndarray,
            momentum: float = 0.9) -> tuple[
        ndarray[Any, dtype[unsignedinteger[Any]]], ndarray[Any, dtype[floating[Any]]]]:

        assert momentum_buffer.shape == param.shape, "Momentum shape must be the same as param shape"
        momentum_buffer = momentum * momentum_buffer + (1 - momentum) * grad
        param = param - lr * momentum_buffer
        return param, momentum_buffer


class RMSProp(Optimizer):
    def __init__(self, btree: Optim_Btree, lr: float, velocity: float = 0.9, decay: float = 0):
        super().__init__(btree=btree, name="SGD")
        self.lr = lr
        self.decay = decay
        self.velocity = velocity

        self.params = None
        self.velocity_buffer = None
        self.grad_buffer = None
        self.iter = None

        self.init()

    @property
    def iter(self):
        return self._iter

    @iter.setter
    def iter(self, iter: int):
        self._iter = iter

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr: float):
        self._lr = lr

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: float):
        self._velocity = velocity

    def init(self):

        self.iter = 0
        self.params = []
        self.grad_buffer = []
        self.velocity_buffer = []

        self.nodes = self.btree.get_non_leaf()

        for i, node in enumerate(self.nodes):
            B_t = node.get_cache("B_t")
            # B_t_mat = B_t.reshape((B_t.shape[1] * B_t.shape[2], B_t.shape[0]))
            B_t_mat = np.zeros((B_t.shape[1] * B_t.shape[2], B_t.shape[0]))
            B_t_mat = matricise(B_t, B_t_mat)

            self.params.append(B_t_mat.astype(self.dtype))
            self.grad_buffer.append(np.zeros(B_t_mat.shape).astype(self.dtype))
            self.velocity_buffer.append(np.zeros(B_t_mat.shape).astype(self.dtype))

    def step(self):
        self.iter += 1
        for i, node in enumerate(self.nodes):
            self.grad_buffer[i] = node.get_cache("grad")
            self.params[i], self.velocity_buffer[i] = self.rmsprop(self.params[i],
                                                                   self.grad_buffer[i],
                                                                   self.lr,
                                                                   self.velocity_buffer[i],
                                                                   self.velocity)
            node.update_cache(
                {"B_t": (tensorise(self.params[i], np.zeros(node.get_cache("B_t").shape)).astype(self.dtype))})

    @staticmethod
    @njit(fastmath=True, parallel=True, nogil=True)
    def rmsprop(param: np.ndarray,
                grad: np.ndarray,
                lr: float,
                velocity_buffer: np.ndarray,
                velocity: float = 0.999
                ) -> tuple[ndarray[Any, dtype[Any]] | Any, ndarray[Any, dtype[floating[Any]]] | Any]:
        assert param.shape == velocity_buffer.shape, "velocity must have the same shape as param"
        velocity_buffer = velocity * velocity_buffer + (1 - velocity) * grad ** 2
        param = param - lr * grad / np.sqrt(velocity_buffer)
        return param, velocity_buffer


class Adam(Optimizer):
    EPSILON = 1e-8

    def __init__(self,
                 btree: Optim_Btree,
                 lr: float,
                 Beta_m: float = 0.9,
                 Beta_v: float = 0.999,
                 decay: float = 0.,
                 decay_rate: float = 1,
                 decay_min: float = 0.,
                 dtype: np.dtype = np.float64):

        super().__init__(btree=btree, name="Adam")
        self._lr_0 = lr
        self.lr = lr
        self.Beta_m = Beta_m
        self.Beta_v = Beta_v
        self.decay = decay
        self.decay_rate = decay_rate
        self.decay_min = decay_min
        self.dtype = dtype

        self.nodes = None
        self.params = None
        self.momentum_buffer = None
        self.velocity_buffer = None
        self.grad_buffer = None
        self.iter = None

        self.init()

    @property
    def iter(self):
        return self._iter

    @iter.setter
    def iter(self, iter: int):
        self._iter = iter

    @property
    def lr(self):
        if self.decay:
            lr = self._lr_0 * self.decay_rate / np.sqrt(self.iter)
        else:
            lr = self._lr_0

        if lr < self.decay_min:
            lr = self.decay_min

        self.lr = lr
        return lr

    @lr.setter
    def lr(self, lr: float):
        self._lr = lr

    def init(self):
        self.iter = 0

        self.momentum_buffer = []
        self.velocity_buffer = []

        self.nodes = self.btree.get_non_leaf()

        for i, node in enumerate(self.nodes):
            B_t = node.get_cache("B_t")
            self.momentum_buffer.append([np.zeros(B_t.shape).astype(self.dtype)])
            self.velocity_buffer.append([np.zeros(B_t.shape).astype(self.dtype)])

            # if (self.btree.BN or self.btree.LN) and node.is_inner():
            #     layer_norm = node.get_cache("LN")
            #     Beta = layer_norm.Beta
            #     Gamma = layer_norm.Gamma
            #
            #     self.momentum_buffer[-1].append(np.zeros(Beta.shape).astype(self.dtype))
            #     self.momentum_buffer[-1].append(np.zeros(Gamma.shape).astype(self.dtype))
            #
            #     self.velocity_buffer[-1].append(np.zeros(Beta.shape).astype(self.dtype))
            #     self.velocity_buffer[-1].append(np.zeros(Gamma.shape).astype(self.dtype))

    def step(self):
        self.iter += 1
        for i, node in enumerate(self.nodes):

            B_t = node.get_cache("B_t")
            B_t_grad = node.get_cache("B_t_grad")

            B_t, self.momentum_buffer[i][0], self.velocity_buffer[i][0] = self.Adam(B_t,
                                                                                    B_t_grad,
                                                                                    self.lr,
                                                                                    self.iter + 1,
                                                                                    self.momentum_buffer[i][0],
                                                                                    self.velocity_buffer[i][0],
                                                                                    self.Beta_m,
                                                                                    self.Beta_v,
                                                                                    self.EPSILON)
            node.update_cache({"B_t": B_t})

            if (self.btree.BN or self.btree.LN) and node.is_inner():
                layer_norm = node.get_cache("LN")
                Beta = layer_norm.Beta
                Beta_grad = layer_norm.Beta_grad
                Gamma = layer_norm.Gamma
                Gamma_grad = layer_norm.Gamma_grad
                Beta, self.momentum_buffer[i][1], self.velocity_buffer[i][1] = self.Adam(Beta,
                                                                                         Beta_grad,
                                                                                         self.lr,
                                                                                         self.iter + 1,
                                                                                         self.momentum_buffer[i][1],
                                                                                         self.velocity_buffer[i][1],
                                                                                         self.Beta_m,
                                                                                         self.Beta_v,
                                                                                         self.EPSILON)

                Gamma, self.momentum_buffer[i][2], self.velocity_buffer[i][2] = self.Adam(Gamma,
                                                                                          Gamma_grad,
                                                                                          self.lr,
                                                                                          self.iter + 1,
                                                                                          self.momentum_buffer[i][2],
                                                                                          self.velocity_buffer[i][2],
                                                                                          self.Beta_m,
                                                                                          self.Beta_v,
                                                                                          self.EPSILON)
                layer_norm.Beta = Beta
                layer_norm.Gamma = Gamma

    def zero_grad(self):
        for i, buffer in enumerate(self.grad_buffer):
            self.grad_buffer[i] = np.zeros(buffer.shape)

    @staticmethod
    @njit(fastmath=True, parallel=True, nogil=True)
    def Adam(param: np.ndarray,
             grad: np.ndarray,
             lr: float,
             iter: int,
             momentum: np.ndarray,
             velocity: np.ndarray,
             Beta_m: float = 0.9,
             Beta_v: float = 0.999,
             epsilon: float = EPSILON,
             ) -> tuple[ndarray[Any, dtype[Any]] | Any, ndarray[Any, dtype[floating[Any]]], ndarray[
        Any, dtype[floating[Any]]] | float | Any]:

        assert momentum.shape == param.shape, "Momentum shape must be the same as param shape"
        assert param.shape == velocity.shape, "velocity must have the same shape as param"

        momentum = Beta_m * momentum + (1 - Beta_m) * grad
        velocity = Beta_v * velocity + (1 - Beta_v) * grad ** 2

        momentum_cor = momentum / (1 - Beta_m ** iter)
        velocity_cor = velocity / (1 - Beta_v ** iter)

        param = param - lr * momentum_cor / (np.sqrt(velocity_cor) + epsilon)

        return param, momentum, velocity


class AdaMax(Optimizer):
    def __init__(self,
                 btree: Optim_Btree,
                 lr: float,
                 Beta_m: float = 0.9,
                 Beta_u: float = 0.999,
                 decay: float = 0.,
                 dtype: np.dtype = np.float64):

        super().__init__(btree=btree, name="Adam")
        self.lr = lr
        self.Beta_m = Beta_m
        self.Beta_u = Beta_u
        self.decay = decay
        self.dtype = dtype

        self.nodes = None
        self.params = None
        self.momentum_buffer = None
        self.exp_weight_buffer = None
        self.grad_buffer = None
        self.iter = None

        self.init()

    @property
    def iter(self):
        return self._iter

    @iter.setter
    def iter(self, iter: int):
        self._iter = iter

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr: float):
        self._lr = lr

    def init(self):
        self.iter = 0

        self.params = []
        self.grad_buffer = []
        self.momentum_buffer = []
        self.exp_weight_buffer = []

        self.nodes = self.btree.get_non_leaf()

        for i, node in enumerate(self.nodes):
            B_t = node.get_cache("B_t")
            # B_t_mat = B_t.reshape((B_t.shape[1] * B_t.shape[2], B_t.shape[0]))
            B_t_mat = np.zeros((B_t.shape[1] * B_t.shape[2], B_t.shape[0]))
            B_t_mat = matricise(B_t, B_t_mat)

            self.params.append(B_t_mat.astype(self.dtype))
            self.grad_buffer.append(np.zeros(B_t_mat.shape).astype(self.dtype))
            self.momentum_buffer.append(np.zeros(B_t_mat.shape).astype(self.dtype))
            self.exp_weight_buffer.append(np.zeros(B_t_mat.shape).astype(self.dtype))

    def step(self):

        self.iter += 1

        for i, node in enumerate(self.nodes):
            self.grad_buffer[i] = node.get_cache("grad")
            self.params[i], self.momentum_buffer[i], self.exp_weight_buffer[i] = self.AdaMax(self.params[i],
                                                                                             self.grad_buffer[i],
                                                                                             self.lr,
                                                                                             self.iter + 1,
                                                                                             self.momentum_buffer[i],
                                                                                             self.exp_weight_buffer[i],
                                                                                             self.Beta_m,
                                                                                             self.Beta_u)

            node.update_cache(
                {"B_t": (tensorise(self.params[i], np.zeros(node.get_cache("B_t").shape)).astype(self.dtype))})

        self.lr = self.lr * (1 - self.decay)

    def zero_grad(self):
        for i, buffer in enumerate(self.grad_buffer):
            self.grad_buffer[i] = np.zeros(buffer.shape)

    @staticmethod
    @njit(fastmath=True, parallel=True, nogil=True)
    def AdaMax(param: np.ndarray,
               grad: np.ndarray,
               lr: float,
               iter: int,
               momentum: np.ndarray,
               exp_weight: np.ndarray,
               Beta_m: float = 0.9,
               Beta_u: float = 0.999,
               ) -> tuple[ndarray[Any, dtype[Any]] | Any, ndarray[Any, dtype[floating[Any]]], ndarray[
        Any, dtype[floating[Any]]] | float | Any]:

        assert momentum.shape == param.shape, "Momentum shape must be the same as param shape"
        assert param.shape == exp_weight.shape, "exponentially weight must have the same shape as param"

        momentum = Beta_m * momentum + (1 - Beta_m) * grad
        exp_weight = np.maximum(Beta_u * exp_weight, np.abs(grad))

        param = param - (lr / (1 - Beta_m ** iter)) * (momentum / exp_weight)

        return param, momentum, exp_weight
