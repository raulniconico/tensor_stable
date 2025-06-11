from numba import prange, njit

from .kernel import Kernel

import numpy as np

from tensor.nuwa import Node
from tensor.nuwa.tensor import SparseTensor

"""
Decrypted

"""


class Multi_Single_Entry_Kernel(Kernel):
    """
    IN CODING !!!

    Single entry kernel with multiple possibility of index.
    Unlike single entry kernel only accepts int as index of sparse tensor, multi single entry kernel also accepts str.
    """
    def __init__(self):
        super().__init__(name="Multi_Single_Entry_Kernel")

    @staticmethod
    def node_init_index(node: Node, feature: np.ndarray):
        """
        Initialize indices for each node sparse tensor

        :param node:
        :param feature:
        :return:
        """

        """ root """
        if node.is_root():
            node.update_cache({"index": feature})

        """ root & inner """
        if node.is_root() or node.is_inner():
            index = node.get_cache("index")
            length = index.shape[1]

            if length == 1:
                return
            else:
                if length % 2 != 0:
                    node.left.update_cache({"index": np.unique(index[:, : length // 2 + 1], axis=0)})
                    node.right.update_cache({"index": np.unique(index[:, length // 2 + 1:], axis=0)})
                else:
                    node.left.update_cache({"index": np.unique(index[:, : length // 2], axis=0)})
                    node.right.update_cache({"index": np.unique(index[:, length // 2:], axis=0)})

    @staticmethod
    def node_forward(node: Node,
                     feature,
                     activation,
                     BN: bool,
                     DO_rate: float,
                     dtype=np.float64):

        """ root and inner """
        if node.is_inner() or node.is_root():
            U_l_s = node.left.get_cache("U_t_s")
            U_r_s = node.right.get_cache("U_t_s")
            B_t = node.get_cache("B_t")
            index = node.get_cache("index")

            """ root """
            if node.is_root():
                A_predict, kron_prod = t_m_m_with_kron_prod_single_entry(index, B_t, U_l_s, U_r_s)
                node.update_cache({"kron_prod": kron_prod})
                return A_predict.value

            """ inner """
            if node.is_inner():
                U_t_s, kron_prod = t_m_m_with_kron_prod_single_entry(index, B_t, U_l_s, U_r_s)

                node.update_cache({"kron_prod": kron_prod, "U_t_s": U_t_s})

        """ leaf """
        if node.is_leaf():
            index = node.get_cache("index")
            print(index.shape)
            U_t = node.get_cache("U_t")
            shape = node.get_cache("shape")

            U_t_s = SparseTensor(index=index, value=U_t, shape=shape, key_type="float64[:]")
            node.update_cache({"U_t_s": U_t_s})

    @staticmethod
    def node_backward(node: Node, feature, activation, BN, DO_rate, dtype=np.float64):

        """ root """
        if node.is_root():
            """ 0: read cached C and W and initial gradient """
            cache = node.get_cache()
            kron_prod = cache["kron_prod"]
            B_t = cache["B_t"]
            grad_0 = cache["grad_0"]

            U_l_s = node.left.get_cache("U_t_s")
            U_r_s = node.right.get_cache("U_t_s")

            """ 1: calculate C and W gradient """
            grad_B_t = kron_prod.value.T @ grad_0
            kron_prod.grad = grad_0 @ B_t.T

            """ 2: calculate left and right gradient """
            U_l_s, U_r_s = kron_grad_single_entry(kron_prod=kron_prod,
                                                  U_l_s=U_l_s,
                                                  U_r_s=U_r_s)

            node.update_cache({"B_t_grad": grad_B_t})
            node.left.update_cache({"U_t_s": U_l_s})
            node.right.update_cache({"U_t_s": U_r_s})

        """ inner """
        if node.is_inner():
            cache = node.get_cache()
            B_t = cache["B_t"]
            U_t_s = cache["U_t_s"]
            kron_prod = cache["kron_prod"]

            """ 3 inner product to children """
            grad_B_t = kron_prod.T @ U_t_s.grad
            kron_prod.grad = U_t_s.grad @ B_t.T

            """ 4 gradient of child node"""
            U_l_s = node.left.get_cache("U_t_s")
            U_r_s = node.right.get_cache("U_t_s")

            U_l_s, U_r_s = kron_grad_single_entry(kron_prod=kron_prod,
                                                  U_l_s=U_l_s,
                                                  U_r_s=U_r_s)

            node.update_cache({"B_t_grad": grad_B_t.astype(dtype)})
            node.left.update_cache({"U_t_s": U_l_s})
            node.right.update_cache({"U_t_s": U_r_s})


def t_m_m_with_kron_prod_single_entry(index: np.ndarray,
                                      B_t: np.ndarray,
                                      U_l_s: SparseTensor,
                                      U_r_s: SparseTensor):
    """ init U_t_s and kron_prod params """
    U_t_s_shape = (U_l_s.shape[0] * U_r_s.shape[0], B_t.shape[1])
    U_t_s_value_shape = (index.size, B_t.shape[1])
    U_t_s_value = np.zeros(U_t_s_value_shape)

    kron_prod_shape = (U_l_s.shape[0] * U_r_s.shape[0], B_t.shape[0])
    kron_prod_value_shape = (index.size, B_t.shape[0])
    kron_prod_value = np.zeros(kron_prod_value_shape)

    """ calculate index exp and U_t_s and kron_prod value """
    index_exp = index_expand_single_entry(index, U_t_s_shape)
    U_t_s_value, kron_prod_value = _t_m_m_with_kron_prod_single_entry(index=index_exp,
                                                                      B_t=B_t,
                                                                      U_l_s=U_l_s.tdict,
                                                                      U_r_s=U_r_s.tdict,
                                                                      U_t_shape=U_t_s_shape,
                                                                      U_l_shape=U_l_s.shape,
                                                                      U_r_shape=U_r_s.shape,
                                                                      values=U_t_s_value,
                                                                      kron_prod=kron_prod_value)
    """ SparseTensor the U_t_s and kron_prod """
    U_t_s = SparseTensor(index=index,
                         value=U_t_s_value,
                         shape=U_t_s_shape,
                         require_grad=True)

    kron_prod = SparseTensor(index=index,
                             value=kron_prod_value,
                             shape=kron_prod_shape,
                             require_grad=True)

    return U_t_s, kron_prod


def kron_grad_single_entry(kron_prod: SparseTensor, U_l_s: SparseTensor, U_r_s: SparseTensor):
    U_l_s.grad, U_r_s.grad = _kron_grad_single_entry(index=kron_prod.index,
                                                     U_l_s=U_l_s.tdict,
                                                     U_r_s=U_r_s.tdict,
                                                     U_r_shape=U_r_s.shape,
                                                     grad_0=kron_prod.tgrad,
                                                     grad_l=U_l_s.tgrad,
                                                     grad_r=U_r_s.tgrad)

    return U_l_s, U_r_s


def index_expand_single_entry(index: np.ndarray, shape: tuple) -> np.ndarray:
    base = np.tile(index, (1, shape[1]))
    x = np.linspace(0, shape[0] * (shape[1] - 1), shape[1], dtype=int)
    dif = np.repeat(x, index.size, axis=0)
    return (base + dif).reshape((-1,))


@njit(parallel=True, nogil=True, fastmath=True)
def _t_m_m_with_kron_prod_single_entry(index: np.ndarray,
                                       B_t: np.ndarray,
                                       U_l_s: np.ndarray,
                                       U_r_s: np.ndarray,
                                       U_t_shape,
                                       U_l_shape,
                                       U_r_shape,
                                       values: np.ndarray,
                                       kron_prod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U_l_x, U_l_y = U_l_shape
    U_r_x, U_r_y = U_r_shape

    kron_prod_x, kron_prod_y = kron_prod.shape
    len_val = index.size

    for v in prange(len_val):
        for i in prange(B_t.shape[0]):
            " U_l index "
            a = (index[v] % (U_l_x * U_r_x)) // U_r_x
            b = i // U_r_y
            " U_r index "
            x = (index[v] % (U_l_x * U_r_x)) % U_r_x
            y = i % U_r_y

            if v < kron_prod_x:
                kron_prod[v, i] = U_l_s[a][b] * U_r_s[x][y]

            values[v % values.shape[0], v // values.shape[0]] += (B_t[i, index[v] // (U_l_x * U_r_x)] * (U_l_s[a][b] * U_r_s[x][y]))

    return values, kron_prod

# @njit(parallel=True, nogil=True, fastmath=True)
# def _t_m_m_with_kron_prod_multi_single_entry(index: np.ndarray,
#                                        B_t: np.ndarray,
#                                        U_l_s: np.ndarray,
#                                        U_r_s: np.ndarray,
#                                        U_t_shape,
#                                        U_l_shape,
#                                        U_r_shape,
#                                        values: np.ndarray,
#                                        kron_prod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     U_l_x, U_l_y = U_l_shape
#     U_r_x, U_r_y = U_r_shape
#
#     kron_prod_x, kron_prod_y = kron_prod.shape
#     len_val = index.size
#
#     for v in prange(len_val):
#         for i in prange(B_t.shape[0]):
#             " U_l index "
#             a = index[v][]
#             b = i // U_r_y
#
#
#             " U_r index "
#             x = index[v][]
#             y = i % U_r_y
#
#             if v < kron_prod_x:
#                 kron_prod[v, i] = U_l_s[a][b] * U_r_s[x][y]
#
#             values[v % values.shape[0], v // values.shape[0]] += (B_t[i, index[v] // (U_l_x * U_r_x)] * (U_l_s[a][b] * U_r_s[x][y]))
#
#     return values, kron_prod


@njit(fastmath=True, nogil=True, parallel=True)
def _kron_grad_single_entry(index: np.ndarray,
                            U_l_s: np.ndarray,
                            U_r_s: np.ndarray,
                            U_r_shape,
                            grad_0: np.ndarray,
                            grad_l: np.ndarray,
                            grad_r: np.ndarray):
    U_r_x, U_r_y = U_r_shape
    grad_0_x, grad_0_y = grad_0.shape
    len_val = index.size

    for i in prange(len_val):
        for j in prange(grad_0_y):
            " U_l index "
            a = index[i] // U_r_x
            b = j // U_r_y
            " U_r index "
            x = index[i] % U_r_x
            y = j % U_r_y

            if grad_l is not None:
                grad_l[a][b] += grad_0[i][j] * U_r_s[x][y]

            if grad_r is not None:
                grad_r[x][y] += grad_0[i][j] * U_l_s[a][b]

    return grad_l, grad_r