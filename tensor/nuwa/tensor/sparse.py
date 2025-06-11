from typing_extensions import Self, Union

import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict

from .tensor import Tensor


from .util import use_value


class SparseTensor(Tensor):
    KEY_TYPE = {"int64": types.int64, "float64": types.float64, "float64[:]": types.float64[:]}

    def __init__(self, index, value, shape=None, require_grad=False, key_type="int64", device="cpu", dtype="float64"):
        super(SparseTensor, self).__init__(value=value, require_grad=require_grad, device=device, dtype=dtype)
        self.index = index
        self.shape = shape

        self.key_type = key_type
        self.tdict = None
        self.tgrad = None

        if require_grad:
            self.tgrad = self.grad

    def copy(self):
        return self.__init__(index=self.index,
                             value=self.value,
                             shape=self.shape,
                             require_grad=self.require_grad,
                             device=self.device,
                             dtype=self.dtype)

    def __add__(self, other: Union[Self]) -> Self:
        if isinstance(other, Tensor):
            self.value = self._add(self.value, other.value)
        elif isinstance(other, SparseTensor):
            assert self.shape == other.shape, "self.shape and other.shape must be equal"
            assert np.array_equal(self.index, other.index), "self.index and other.index must be equal"

            self.value = self._add(self.value, other.value)
        else:
            self.value = self._add(self.value, other)
        return self

    __radd__ = __add__

    def __sub__(self, other: Union[Self]) -> Self:
        if isinstance(other, Tensor):
            self.value = self._subtract(self.value, other.value)
        elif isinstance(other, SparseTensor):
            assert self.shape == other.shape, "self.shape and other.shape must be equal"
            assert np.array_equal(self.index, other.index), "self.index and other.index must be equal"
            self.value = self._subtract(self.value, other.value)
        else:
            self.value = self._subtract(self.value, other)
        return self

    __rsub__ = __sub__

    def __mul__(self, other):
        if isinstance(other, Tensor):
            self.value = self._multiply(self.value, other.value)
        elif isinstance(other, SparseTensor):
            assert self.shape == other.shape, "self.shape and other.shape must be equal"
            assert np.array_equal(self.index, other.index), "self.index and other.index must be equal"
            self.value = self._multiply(self.value, other.value)
        else:
            self.value = self._multiply(self.value, other)
        return self

    def __rmul__(self, other):
        if isinstance(other, Tensor):
            self.value = self._multiply(other.value, self.value)
        elif isinstance(other, SparseTensor):
            assert self.shape == other.shape, "self.shape and other.shape must be equal"
            assert np.array_equal(self.index, other.index), "self.index and other.index must be equal"
            self.value = self._multiply(other.value, self.value)
        else:
            self.value = self._multiply(other, self.value)
        return self

    def __truediv__(self, other: Union[Self]) -> Self:
        if isinstance(other, Tensor):
            self.value = self._divide(self.value, other.value)
        elif isinstance(other, SparseTensor):
            assert self.shape == other.shape, "self.shape and other.shape must be equal"
            assert np.array_equal(self.index, other.index), "self.index and other.index must be equal"
            self.value = self._divide(self.value, other.value)
        else:
            self.value = self._divide(self.value, other)
        return self

    __rtruediv__ = __truediv__

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: np.ndarray) -> None:
        # assert index.ndim == 1, "the index should be a 1D np.ndarray"
        self._index = index
        self._tdict = self.value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self._tdict = value

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def value_shape(self):
        return self.value.shape if self.value is not None else None

    @value_shape.setter
    def value_shape(self, shape):
        pass

    @property
    def key_type(self):
        return self._key_type

    @key_type.setter
    def key_type(self, key_type: str):
        assert key_type in self.KEY_TYPE.keys(), "key_type must be one of %s" % self.KEY_TYPE.keys()
        self._key_type = self.KEY_TYPE[key_type]

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad=None):
        self._grad = np.zeros(self.value_shape)
        self._tgrad = self._grad

    @staticmethod
    @njit(nogil=True, cache=True)
    def register(index, value, key_type=types.int64):
        d = Dict.empty(
            key_type=key_type,
            value_type=types.float64[:],
        )
        for i in prange(index.shape[0]):
            d[index[i]] = value[i, :]
        return d

    @property
    def tdict(self):
        return self._tdict

    @tdict.setter
    def tdict(self, value=None):
        d = self.register(index=self.index, value=self.value, key_type=self.key_type)
        self._tdict = d

    @property
    def tgrad(self):
        return self._tgrad

    @tgrad.setter
    def tgrad(self, grad=None):
        d = self.register(self.index, self.grad, self.key_type)
        self._tgrad = d
