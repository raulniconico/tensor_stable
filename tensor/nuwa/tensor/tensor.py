import numpy as np
import sys
sys.path.append("core")
from ..btree.btree import Node

from .core import numpy_core as numpy_core
# import nuwa.tensor.core.torch_core as torch_core

from typing_extensions import Self, Union



class Tensor(Node):
    INTER_FUNCTION = ["add", "subtract", "multiply", "divide", "power"]
    BASIC_FUNCTION = ["mean", "std", "var", "max", "min"]
    NP_DTYPE = {"int": np.int64, "float32": np.float32, "float64": np.float64}
    # TORCH_DTYPE = {"float32": torch.float32, "float64": torch.float64}
    DEVICE = ["cpu", "gpu"]

    def __init__(self, value, require_grad=False, device="cpu", dtype="float64"):
        super().__init__()
        self.value = value
        self.grad = None

        self.device = device
        self.dtype = dtype

        self.require_grad = require_grad
        if require_grad:
            self.grad = np.zeros_like(self.value)

    def copy(self):
        return self.__init__(value=self.value, require_grad=self.require_grad, device=self.device, dtype=self.dtype)

    def __add__(self, other: Union[Self, np.ndarray]) -> Self:
        if isinstance(other, Tensor):
            value = self._add(self.value, other.value)
            t = self.copy()
            t.value = value
        else:
            value = self._add(self.value, other)
            t = self.copy()
            t.value = value
        return t

    def __sub__(self, other: Union[Self, np.ndarray]) -> Self:
        if isinstance(other, Tensor):
            value = self._subtract(self.value, other.value)
            t = self.copy()
            t.value = value
        else:
            value = self._subtract(self.value, other)
            t = self.copy()
            t.value = value
        return t

    @staticmethod
    def _add(self, other):
        raise NotImplementedError

    @staticmethod
    def _subtract(self, other):
        raise NotImplementedError

    @staticmethod
    def _multiply(self, other):
        raise NotImplementedError

    @staticmethod
    def _divide(self, other):
        raise NotImplementedError

    def mean(self):
        return self._mean(self.value)

    def _mean(self, x):
        raise NotImplementedError

    def std(self):
        return self._mean(self.value)

    def _std(self, x):
        raise NotImplementedError

    def var(self):
        return self._var(self.value)

    def _var(self, x):
        raise NotImplementedError

    def max(self):
        return self._max(self.value)

    def _max(self, x):
        raise NotImplementedError

    def min(self):
        return self._min(self.value)

    def _min(self, x):
        raise NotImplementedError

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        assert device in ["cpu", "gpu"]
        # assert isinstance(self.value, torch.Tensor) or isinstance(self.value, np.ndarray)

        if device == "cpu":
            assert isinstance(self.value, np.ndarray)
            for func in self.BASIC_FUNCTION:
                self.__setattr__("_" + func, getattr(numpy_core, func))
            for func in self.INTER_FUNCTION:
                self.__setattr__("_" + func, getattr(numpy_core, func))

        # if device == "gpu":
        #     gpu_device = torch.device("cuda:0")
        #     if isinstance(self.value, np.ndarray):
        #         self.value = torch.Tensor(self.value).to(gpu_device)
        #     for func in self.INTER_FUNCTION:
        #         self.__setattr__("_" + func, getattr(torch_core, func))
        #
        #     for func in self.INTER_FUNCTION:
        #         self.__setattr__("_" + func, getattr(torch_core, func))

        self._device = device

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        # assert dtype in self.NP_DTYPE.keys() or dtype in self.TORCH_DTYPE.keys()
        assert dtype in self.NP_DTYPE.keys()
        if self._device == "cpu":
            self._dtype = self.NP_DTYPE[dtype]
        # elif self._device == "gpu":
        #     self._dtype = self.TORCH_DTYPE[dtype]
        else:
            raise NotImplementedError

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad):
        self._grad = grad

    @property
    def T(self):
        return self.value.T

    @T.setter
    def T(self, T):
        pass
