import numpy as np
from numba import prange, njit

from tensor.nuwa import reconstruction_polynomial
from tensor.nuwa import Node
from .kernel import Kernel

from ..NN import LayerNorm

"""
Concatenate kernel

The indices are propagated by the dimension.
"""

class Concatenate_Kernel(Kernel):
    """
    IN PROGRAMING

    """

    def __init__(self):
        super().__init__(name="Concatenate_Kernel")

    @staticmethod
    def node_init_index(node: Node, feature):
        if node.is_leaf():
            mode = node.get_cache("mode")
            node.update_cache({"index": feature[:, mode]})

    @staticmethod
    def node_forward(node: Node,
                     feature,
                     activation,
                     BN,
                     LN,
                     DO_rate,
                     poly_base,
                     dtype=np.float64,
                     *args,
                     **kwargs):
        """
        Forward propagation

        :param dtype:
        :param DO_rate:
        :param BN:
        :param node:
        :return:
        """
        if node.is_root():
            U_l = node.left.get_cache("U_t")
            U_r = node.right.get_cache("U_t")
            B_t = node.get_cache("B_t")

            kron_prod = kron_concatenate(U_l, U_r, np.zeros((U_l.shape[0], U_l.shape[1] * U_r.shape[1])))

            A_predict = kron_prod @ B_t

            node.update_cache({"kron_prod": kron_prod})
            return A_predict

        if node.is_inner():
            """
            index flow in preparation
            """
            U_l = node.left.get_cache("U_t")
            U_r = node.right.get_cache("U_t")
            B_t = node.get_cache("B_t")

            kron_prod = kron_concatenate(U_l=U_l,
                                         U_r=U_r,
                                         prod=np.zeros((U_l.shape[0], U_l.shape[1] * U_r.shape[1])))

            U_t = kron_prod @ B_t

            # activation function
            if activation is not None:
                U_t = activation(U_t)

            # layer normalisation
            if LN:

                layer_norm = LayerNorm(d=U_t.shape[1])
                U_t = layer_norm(U_t)
                node.update_cache({"LN": layer_norm})

            # Dropout layer
            thres = np.random.rand(*U_t.shape)
            U_t[thres <= DO_rate] = 0

            # node.update_cache({"thres": thres})
            node.update_cache({"kron_prod": kron_prod})
            node.update_cache({"U_t": U_t})

        if node.is_leaf():
            """ reconstruct base in each step """
            coor = node.get_cache("index")
            U_c = node.get_cache("U_c")
            U_t = reconstruction_polynomial(U_c, poly_base, coor).astype(dtype)
            node.update_cache({"U_t": U_t})


    @staticmethod
    def node_backward(node: Node,
                      feature,
                      activation,
                      BN,
                      LN,
                      DO_rate,
                      dtype=np.float64):
        if node.is_root():
            # 0: read cached C and W and initial gradient
            cache = node.get_cache()
            kron_prod = cache["kron_prod"]
            B_t = cache["B_t"]
            grad_0 = cache["grad_0"]

            U_l = node.left.get_cache("U_t")
            U_r = node.right.get_cache("U_t")
            U_l_x, U_l_y = U_l.shape
            U_r_x, U_r_y = U_r.shape

            # 1: calculate C and W gradient
            grad_B_t = kron_prod.T @ grad_0
            grad_0 = grad_0 @ B_t.T

            if node.left.is_leaf():
                grad_kron_left = None
            else:
                grad_kron_left = np.zeros((U_l_x, U_l_y))

            if node.right.is_leaf():
                grad_kron_right = None
            else:
                grad_kron_right = np.zeros((U_r_x, U_r_y))

            if grad_kron_left is not None or grad_kron_right is not None:
                grad_kron_left, grad_kron_right = kron_grad_concatenate(sample_index=None,
                                                                        U_l=U_l,
                                                                        U_r=U_r,
                                                                        grad_0=grad_0,
                                                                        grad_l=grad_kron_left,
                                                                        grad_r=grad_kron_right)
            if grad_kron_left is not None:
                node.left.update_cache({"grad_0": grad_kron_left.astype(dtype)})
            if grad_kron_right is not None:
                node.right.update_cache({"grad_0": grad_kron_right.astype(dtype)})

            node.update_cache({"B_t_grad": grad_B_t.astype(dtype)})

            node.clean_cache("kron_prod")

        if node.is_inner():
            # 0. read cache
            cache = node.get_cache()
            B_t = cache["B_t"]
            kron_prod = cache["kron_prod"]
            grad_0 = cache["grad_0"]
            U_t = cache["U_t"]
            # thres = cache["thres"]

            # 1. Drop out gradient
            # grad_0[thres < DO_rate] = 0

            # 2. LN gradient
            if LN:

                layer_norm = node.get_cache("LN")
                grad_0 = layer_norm.backward(grad_0)

            # 2.activation gradient
            if activation is not None:
                grad_0 = activation.grad(U_t, grad_0)

            # 4. inner product to children
            grad_B_t = kron_prod.T @ grad_0
            grad_0 = grad_0 @ B_t.T

            node.update_cache({"B_t_grad": grad_B_t.astype(dtype)})

            # leaf update gradient

            U_l = node.left.get_cache("U_t")
            U_r = node.right.get_cache("U_t")
            U_l_x, U_l_y = U_l.shape
            U_r_x, U_r_y = U_r.shape

            if node.left.is_leaf():
                grad_kron_left = None
            else:
                grad_kron_left = np.zeros((U_l_x, U_l_y))

            if node.right.is_leaf():
                grad_kron_right = None
            else:
                grad_kron_right = np.zeros((U_r_x, U_r_y))

            if grad_kron_left is not None or grad_kron_right is not None:
                grad_kron_left, grad_kron_right = kron_grad_concatenate(sample_index=None,
                                                                        U_l=U_l,
                                                                        U_r=U_r,
                                                                        grad_0=grad_0,
                                                                        grad_l=grad_kron_left,
                                                                        grad_r=grad_kron_right)
            node.left.update_cache({"grad_0": grad_kron_left})
            node.right.update_cache({"grad_0": grad_kron_right})
            node.clean_cache("kron_prod")
            node.clean_cache("U_t")


@njit(parallel=True, nogil=True, fastmath=True)
def kron_concatenate(U_l: np.ndarray,
                     U_r: np.ndarray,
                     prod: np.ndarray) -> np.ndarray:
    U_l_x, U_l_y = U_l.shape
    U_r_x, U_r_y = U_r.shape

    assert U_l_x == U_r_x, "left and right factor should have same shape 0"

    for v in prange(U_l_x):
        for i in prange(U_l_y * U_r_y):
            prod[v, i] = U_l[v, i // U_r_y] * U_r[v, i % U_r_y]
    return prod


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def kron_grad_concatenate(sample_index: None,
                          U_l: np.ndarray,
                          U_r: np.ndarray,
                          grad_0: np.ndarray,
                          grad_l: np.ndarray,
                          grad_r: np.ndarray):
    U_l_x, U_l_y = U_l.shape
    U_r_x, U_r_y = U_r.shape
    grad_0_x, grad_0_y = grad_0.shape

    assert U_l_x == U_r_x and U_r_x == grad_0_x, f"U_l, U_r and grad_0 should have same shape 0"

    for i in prange(grad_0_x):
        for j in prange(grad_0_y):

            if grad_l is not None:
                grad_l[i, j // U_r_y] += grad_0[i, j] * U_r[i, j % U_r_y]

            if grad_r is not None:
                grad_r[i, j % U_r_y] += grad_0[i, j] * U_l[i, j // U_r_y]

    return grad_l, grad_r
