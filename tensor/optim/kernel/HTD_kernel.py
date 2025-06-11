from numba import njit, prange

from .kernel import Kernel
import numpy as np

from tensor.nuwa import Node


class HTD_kernel(Kernel):
    def __init__(self):
        super().__init__(name="HTD_Kernel")

    @staticmethod
    def node_init_index(node: Node, feature):
        if node.is_root():
            node.update_cache({"index": feature.reshape((-1,))})
        else:
            feature = node.parent.get_cache("index")
            if node.parent.left is node:
                right_x = int(node.parent.right.get_cache("shape")[0])
                node.update_cache({"index": np.sort(np.unique(feature // right_x)).astype(int)})
            elif node.parent.right is node:
                right_x = int(node.parent.right.get_cache("shape")[0])
                node.update_cache({"index": np.sort(np.unique(feature % right_x)).astype(int)})
            else:
                raise Exception("The node is neither left or right child of parent")

    @staticmethod
    def node_forward(node: Node,
                     feature,
                     activation,
                     BN: bool,
                     LN:bool,
                     DO_rate: float,
                     dtype=np.float64):
        """
        Forward propagation

        :param node:
        :param feature:
        :param activation:
        :param BN:

        The factors (rows) of truncation of U_t by tmm and mmm are concat in row major

        :param dtype:

        :return:
        """

        if node.is_inner() or node.is_root():
            U_l = node.left.get_cache("U_t")
            U_r = node.right.get_cache("U_t")
            B_t = node.get_cache("B_t")

            """ root """
            if node.is_root():
                A_predict = np.zeros((np.size(feature), 1), dtype=dtype)
                kron_prod = np.zeros((np.size(feature), B_t.shape[0]), dtype=dtype)

                A_predict, kron_prod = t_m_m_with_kron_prod_HTD(
                    feature.reshape((-1,)), B_t, U_l, U_r, A_predict, kron_prod)

                node.update_cache({"kron_prod": kron_prod.astype(dtype)})
                return A_predict.astype(dtype)

            """ inner """
            if node.is_inner():

                U_t = np.zeros((U_l.shape[0] * U_r.shape[0], B_t.shape[1]), dtype=dtype)
                index = np.linspace(0,
                                    U_l.shape[0] * U_r.shape[0] * B_t.shape[1] - 1,
                                    U_l.shape[0] * U_r.shape[0] * B_t.shape[1], dtype=int)
                kron_prod = np.zeros((U_l.shape[0] * U_r.shape[0], B_t.shape[0]), dtype=dtype)
                U_t, kron_prod = t_m_m_with_kron_prod_HTD(index, B_t, U_l, U_r, U_t, kron_prod)

                """ batch normalisation """
                if BN:
                    """BN paper:https://arxiv.org/abs/1502.03167v3
                    U_t_BN is the input U_t to the B_t, equivalent to papers x_i while
                    U_t equivalent to paper's x_i hat
                    Gamma and Beta are the trainable parameters of BN"""

                    node.update_cache({"U_t_BN": U_t.astype(dtype)})
                    Beta = node.get_cache("Beta")
                    Gamma = node.get_cache("Gamma")
                    mean = U_t.mean(dtype=dtype)
                    var = U_t.var(dtype=dtype)
                    U_t_mean = U_t - mean
                    U_t_var = U_t_mean / np.sqrt(var ** 2 + 1e-8)
                    U_t = Gamma * U_t_var + Beta
                    node.update_cache({"mean": mean})
                    node.update_cache({"var": var})

                # activation function
                if activation is not None:
                    U_t = activation(U_t.astype(dtype))

                # Dropout layer
                thres = np.random.rand(*U_t.shape).astype(dtype)
                U_t[thres <= DO_rate] = 0
                node.update_cache({"thres": thres})

                node.update_cache({"kron_prod": kron_prod, "U_t": U_t})

    @staticmethod
    def node_backward(node: Node,
                      feature,
                      activation,
                      BN,
                      LN,
                      DO_rate,
                      dtype=np.float64):
        ##################### root
        if node.is_root():
            """ 0: read cached C and W and initial gradient """
            cache = node.get_cache()
            kron_prod = cache["kron_prod"]
            B_t = cache["B_t"]
            grad_0 = cache["grad_0"]
            index = cache["index"]

            U_l = node.left.get_cache("U_t")
            U_r = node.right.get_cache("U_t")
            U_l_x, U_l_y = U_l.shape
            U_r_x, U_r_y = U_r.shape

            """ 1: calculate C and W gradient """
            grad_B_t = kron_prod.T @ grad_0
            grad_0 = grad_0 @ B_t.T

            """ 2: calculate left and right gradient """
            grad_kron_left, grad_kron_right = kron_grad_HTD(index=index,
                                                            U_l=U_l,
                                                            U_r=U_r,
                                                            grad_0=grad_0,
                                                            grad_l=np.zeros((U_l_x, U_l_y)),
                                                            grad_r=np.zeros((U_r_x, U_r_y)))

            node.update_cache({"B_t_grad": grad_B_t.astype(dtype)})
            node.left.update_cache({"grad_0": grad_kron_left.astype(dtype)})
            node.right.update_cache({"grad_0": grad_kron_right.astype(dtype)})

        ##################### inner
        if node.is_inner():
            cache = node.get_cache()
            B_t = cache["B_t"]
            kron_prod = cache["kron_prod"]
            grad_0 = cache["grad_0"]
            U_t = cache["U_t"]
            thres = cache["thres"]

            """ 0 DO layer """
            grad_0[thres < DO_rate] = 0

            """ 1 activation gradient """
            if activation is not None:
                grad_0 = activation.grad(U_t, grad_0)

            """ 2 Batch normalisation gradient """
            if BN:
                """ 
                U_t_BN is the input U_t to the B_t, equivalent to papers x_i while
                U_t equivalent to paper's x_i hat
                """
                U_t_BN = cache["U_t_BN"]
                mean = U_t_BN.mean()
                var = U_t_BN.var()
                Gamma = cache["Gamma"]

                node.update_cache({"Beta_grad": grad_0})
                node.update_cache({"Gamma_grad": grad_0 * U_t})
                grad_0 = grad_0 * Gamma
                grad_var = grad_0 * (U_t_BN - mean) * -0.5 * (var + 1e-8) ** -1.5
                grad_mean = grad_0 * -1 / np.sqrt(var + 1e-8) + grad_var * np.sum(-2 * (U_t_BN - mean)) / U_t_BN.size
                grad_0 = grad_0 * 1 / np.sqrt(var + 1e-8) + grad_var * 2 * (
                        U_t_BN - mean) / U_t_BN.size + grad_mean / U_t_BN.size

            """ 3 inner product to children """
            grad_B_t = kron_prod.T @ grad_0
            grad_0 = grad_0 @ B_t.T

            node.update_cache({"B_t_grad": grad_B_t.astype(dtype)})

            """ 4 gradient of child node"""
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
                index = np.linspace(0, U_t.shape[0] - 1, U_t.shape[0], dtype=int)
                grad_kron_left, grad_kron_right = kron_grad_HTD(index=index,
                                                                U_l=U_l,
                                                                U_r=U_r,
                                                                grad_0=grad_0,
                                                                grad_l=grad_kron_left,
                                                                grad_r=grad_kron_right)

                node.left.update_cache({"grad_0": grad_kron_left})
                node.right.update_cache({"grad_0": grad_kron_right})


@njit(parallel=True, nogil=True, fastmath=True)
def t_m_m_with_kron_prod_HTD(index: np.ndarray,
                             B: np.ndarray,
                             U_l: np.ndarray,
                             U_r: np.ndarray,
                             values: np.ndarray,
                             kron_prod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    The single entry method compute only one point in the initial tensor A

    The performance is not optimal since every outer product need to be calculated
    for each point occurs in A.

    !!! B is matricised

    :param B:
    :param U_l:
    :param U_r:
    :return:
    """
    assert index.size == values.size, "the size of index of values is different than the size of values"

    U_t_x, U_t_y = values.shape
    kron_prod_x, kron_prod_y = kron_prod.shape
    U_l_x, U_l_y = U_l.shape
    U_r_x, U_r_y = U_r.shape

    len_val = values.size

    for v in prange(len_val):
        for i in prange(B.shape[0]):

            a = (index[v] % (U_l_x * U_r_x)) // U_r_x
            b = i // U_r_y
            x = (index[v] % (U_l_x * U_r_x)) % U_r_x
            y = i % U_r_y

            if v < kron_prod_x:
                kron_prod[v, b * U_r_y + y] = U_l[a, b] * U_r[x, y]

            values[v % U_t_x, v // U_t_x] += (B[i, index[v] // (U_l_x * U_r_x)] * (U_l[a, b] * U_r[x, y]))
    return values, kron_prod


@njit(fastmath=True, cache=True, nogil=True, parallel=True)
def kron_grad_HTD(
        index: np.ndarray,
        U_l: np.ndarray,
        U_r: np.ndarray,
        grad_0: np.ndarray,
        grad_l: np.ndarray,
        grad_r: np.ndarray):
    U_l_x, U_l_y = U_l.shape
    U_r_x, U_r_y = U_r.shape
    grad_0_x, grad_0_y = grad_0.shape

    for i in prange(len(index)):
        for j in prange(grad_0_y):
            " U_l index "
            a = index[i] // U_r_x
            b = j // U_r_y
            " U_r index "
            x = index[i] % U_r_x
            y = j % U_r_y

            if grad_l is not None:
                grad_l[a, b] += grad_0[i, j] * U_r[x, y]

            if grad_r is not None:
                grad_r[x, y] += grad_0[i, j] * U_l[a, b]

    return grad_l, grad_r
