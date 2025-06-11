import numpy as np
from numba import njit, prange


class LayerNorm:
    def __init__(self):
        super(LayerNorm, self).__init__()

    def __call__(self, U_t, Gamma, Beta):
        return self.Layer_Norm(U_t, Gamma, Beta)

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def Layer_Norm(U_t, Gamma, Beta):
        """
        Layer normalization of U_t.

        :param U_t:
        :param Gamma:
        :param Beta:
        :return:
        """
        U_t_x, U_t_y = U_t.shape

        for i in prange(U_t_x):
            mean = U_t[i, :].mean()
            var = U_t[i, :].var()
            U_t[i, :] = Gamma * ((U_t[i, :] - mean) / np.sqrt(var ** 2 + 1e-8)) + Beta

        return U_t

    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True)
    def grad(U_t_0, Gamma, grad_0):
        """

        :param U_t_0: equal to "U_t_LN" stocked in Btree, the U_t before LN
        :param Gamma:
        :param Beta:
        :param grad_0:
        :return:
        """
        U_t_x, U_t_y = U_t_0.shape

        for i in prange(U_t_x):
            grad_0[i, :] = grad_0[i, :] * Gamma
            mean = U_t_0[i, :].mean()
            var = U_t_0[i, :].var()
            grad_var = grad_0[i, :] * (U_t_0[i, :] - mean) * -0.5 * (var + 1e-8) ** -1.5
            grad_mean = grad_0[i, :] * -1 / np.sqrt(var + 1e-8) + grad_var * np.sum(-2 * (U_t_0[i, :] - mean)) / U_t_y
            grad_0 = grad_0 * 1 / np.sqrt(var + 1e-8) + grad_var * 2 * (U_t_0[i, :] - mean) / U_t_y + grad_mean / U_t_y

        return grad_0


def layer_norm(U_t, Gamma, Beta):
    LN = LayerNorm()
    return LN.Layer_Norm(U_t, Gamma, Beta)


def layer_norm_grad(U_t_0, Gamma, Beta):
    LN = LayerNorm()
    return LN.grad(U_t_0, Gamma, Beta)
