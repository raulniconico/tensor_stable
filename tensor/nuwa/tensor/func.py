import numpy as np

from .tensor import Tensor
from .sparse import SparseTensor


def interface(func, *args, **kwargs):
    """
    interface function is the interface of Tensor class with numpy and pytorch.
    It provides an interface for creating new Tensor from numpy and pytorch by parsing numpy or pytorch function
    :return:
    """
    return


def zeros(shape, require_grad=False, device="cpu", dtype="float64"):
    """
    It takes shape as parameter

    :param shape:
    :param require_grad:
    :param device:
    :param dtype:
    :return:
    """
    assert device in Tensor.DEVICE, f"device must be one of: {Tensor.DEVICE}"
    if device == "cpu":
        value = np.zeros(shape)
        tensor = Tensor(value=value, require_grad=require_grad, device=device, dtype=dtype)
        return tensor
    # elif device == "gpu":
    #     value = torch.zeros(shape)
    #     tensor = Tensor(value=value, require_grad=require_grad, device=device, dtype=dtype)
    #     return tensor
    else:
        print("device not supported")
        raise NotImplementedError


def ones(shape, require_grad=False, device="cpu", dtype="float64"):
    assert device in Tensor.DEVICE, f"device must be one of: {Tensor.DEVICE}"
    if device == "cpu":
        value = np.ones(shape)
        tensor = Tensor(value=value, require_grad=require_grad, device=device, dtype=dtype)
        return tensor
    # elif device == "gpu":
    #     value = torch.ones(shape)
    #     tensor = Tensor(value=value, require_grad=require_grad, device=device, dtype=dtype)
    #     return tensor
    else:
        print("device not supported")
        raise NotImplementedError


def zeros_sparse(index, value_shape, shape=None, dtype="float64"):
    zero = np.zeros(value_shape, dtype=dtype)
    t = SparseTensor(index=index, value=zero, shape=shape, dtype=dtype)
    return t


def zeros_like_sparse(A: SparseTensor) -> SparseTensor:
    t = SparseTensor(A.index, np.zeros_like(A.value), A.shape)
    return t