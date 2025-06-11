import os
from typing_extensions import Union, Tuple, List, Any
import time
import psutil

import numpy as np
import scipy

from numba import njit, prange


HARTREE2ICM = 219474.63


########################################
#          decorator function          #
########################################
def monitor(func):
    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss

    def process_time():
        current = time.time()
        return current

    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        start = process_time()
        result = func(*args, **kwargs)
        finish = process_time()
        mem_after = process_memory()
        print("{}:consumed memory: {}".format(
            func.__name__, mem_after - mem_before)
        )
        return result

    return wrapper


########################################
#             optimisation             #
########################################

def power_law(x, C, alpha):
    return C * np.power(x, alpha)


def inverse_power_law(y, C, alpha):
    return (y / C) ** (1 / alpha)


@njit(fastmath=True, nogil=True, parallel=True, cache=True)
def matricise(X_tensor: np.ndarray, X_mat) -> np.ndarray:
    for i in prange(X_mat.shape[0]):
        for j in prange(X_mat.shape[1]):
            a = i // X_tensor.shape[2]
            b = i % X_tensor.shape[2]
            X_mat[i, j] = X_tensor[j, a, b]
    return X_mat


@njit(fastmath=True, nogil=True, parallel=True, cache=True)
def tensorise(X_mat: np.ndarray, X_tensor) -> np.ndarray:
    for k in prange(X_tensor.shape[0]):
        for i in prange(X_tensor.shape[1]):
            for j in prange(X_tensor.shape[2]):
                X_tensor[k, i, j] = X_mat[i * X_tensor.shape[2] + j, k]
    return X_tensor


########################################
#                  svd                 #
########################################

@njit
def svd(A_t: np.ndarray, full_matrices=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate SVD of matrix A_t

    :param A_t: np.ndarray
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray], the U, E, V of A
    """
    return np.linalg.svd(A_t, full_matrices=full_matrices)


@njit(fastmath=True)
def svd_random(A_t: np.ndarray, r: int, p: int, full_matrices: bool = True) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomized SVD

    :param A_t: np.ndarray, Matrix to be svd
    :param r: int, Target final rank of decomposition
    :param p: int, Decomposition factor
    :param full_matrices: bool, default: True. parameters of numpy sub svd
    :return: U, sigma, V, U \Sigma and V of matrix A_t
    """
    m, n = np.shape(A_t)
    omega = np.random.randn(n, r + p)
    Y = A_t @ omega
    Q, R = np.linalg.qr(Y)
    B = Q.T @ A_t
    U_B, sigma, V = svd(B, full_matrices=full_matrices)
    U = Q @ U_B[:, :r]
    sigma = sigma[:r]
    V = V[:, :r]
    return U, sigma, V


########################################
#             decomposition            #
########################################

@njit(nogil=True, parallel=True)
def m_m_m(U_t: np.ndarray, U_l: np.ndarray, U_r: np.ndarray) -> np.ndarray:
    """
    Calculate a matrix B inner product with the kronecker product of two other matrix U_l and U_r.
    It is used to calculate entire input tensor A point by point using B_M_M_single.
    Usage during decomposition: calculate B_t and reconstruction of A_H in root node


    :param A: np.ndarray, initial tensor
    :param U_l: np.ndarray, U_t of node 1 for reconstruction
    :param U_r: np.ndarray, U_t of node 2 for reconstruction
    :return: A: np.ndarray, vectorized product of inner kronecker.
    """

    U_t_x, U_t_y = U_t.shape
    U_l_x, U_l_y = U_l.shape
    U_r_x, U_r_y = U_r.shape

    B_t = np.zeros((U_t_y, U_l_y, U_r_y))

    for i in prange(0, U_l_y):
        for j in prange(0, U_r_y):
            product = np.zeros((U_l_x * U_r_x, 1))
            for l in prange(U_l_x):
                product[l * U_r_x: (l + 1) * U_r_x] = (U_l[l, i] * U_r[:, j]).reshape(-1, 1)
            B_t[:, i, j] = (U_t.T @ product).reshape(B_t[:, i, j].shape)
    return B_t


@njit(parallel=True, nogil=True, fastmath=True)
def t_m_m(index: np.ndarray,
          B: np.ndarray,
          U_l: np.ndarray,
          U_r: np.ndarray,
          values: np.ndarray) -> np.ndarray:
    """

    The single entry method compute only one point in the initial tensor A

    The performance is not optimal since every outer product need to be calculated
    for each point occurs in A.


    :param B:
    :param U_l:
    :param U_r:
    :return:
    """

    assert index.size == values.size, "the size of index of values is different than the size of values"

    U_t_x, U_t_y = values.shape
    U_l_x, U_l_y = U_l.shape
    U_r_x, U_r_y = U_r.shape

    len_val = values.size
    for v in prange(len_val):
        for i in prange(B.shape[1]):
            for j in prange(B.shape[2]):
                a = (index[v] % (U_l_x * U_r_x)) // U_r_x
                b = (index[v] % (U_l_x * U_r_x)) % U_r_x
                values[v % U_t_x, v // U_t_x] += (B[index[v] // (U_l_x * U_r_x), i, j] * (U_l[a, i] * U_r[b, j]))
    return values





########################################
#              polynomial              #
########################################

def construction_polynomial(U_t: np.ndarray, poly_fit, deg: int, coor: Union[np.ndarray, list]) -> np.ndarray:
    """
    Each column in the U_t will be interpolated by polynomial indicated
    by poly_fit with its degree deg.

    :param U_t: np.ndarray,
    :param poly_fit: function, numpy polynomial fit function
    :param deg: int, degree of polynomial
    :param coor: np.ndarray
    :return: U_c: np.ndarray,
    """
    U_c = np.zeros((deg + 1, U_t.shape[1]))
    for i in range(U_t.shape[1]):
        U_c[:, i] = poly_fit(coor, U_t[:, i], deg)
    return U_c


def reconstruction_polynomial(U_c: np.ndarray,
                              polyval,
                              coor: Union[np.ndarray, list]) -> np.ndarray:
    """
    U_t will be reconstructed from chebyshev basis U_c.

    :param U_c: np.ndarray
    :param polyval:
    :param coor:
    :param d: int
    :return: U_t: np.ndarray, full basis reconstructed by chebyshev
    """

    U_t = np.zeros((len(coor), U_c.shape[1]))

    for i in range(U_c.shape[1]):
        U_t[:, i] = polyval(coor, U_c[:, i]).reshape(-1, )

    return U_t

def print_logo():
    logo = r"""
   ╔═════════════════════════════════════════════════╗
   ║  ██╗  ██╗████████╗    ███████╗██████╗  ██████╗  ║
   ║  ██║  ██║╚══██╔══╝    ██╔════╝██╔══██╗ ██╔══██╗ ║
   ║  ███████║   ██║  ███╗ ███████╗██████╔╝ ██████╔╝ ║
   ║  ██╔══██║   ██║  ╚══╝ ██╔════╝██╔══██╗ ██╔═██║  ║
   ║  ██║  ██║   ██║       ██║     ██████╔╝ ██║ ║██╗ ║  
   ║  ╚═╝  ╚═╝   ╚═╝       ╚═╝      ╚════╝  ╚═╝ ╚══╝ ║  
   ║             Powered by Zixing QIU               ║   
   ╚═════════════════════════════════════════════════╝
    """
    print(logo)



