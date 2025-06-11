########################################
#                 RMSE                 #
########################################
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit


@njit
def RMSE(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calculate RMSE of X and Y

    :param X: np.ndarray
    :param Y: np.ndarray
    :return: np.float64, RMSE of X and Y
    """
    assert X.shape == Y.shape, "X and Y must have same shape"
    return np.sqrt(((X - Y) ** 2).mean())


@njit(nogil=True)
def cum_rmse(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    assert X.shape == Y.shape, "X and Y must have same shape"
    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    sorted_idx = np.argsort(X, axis=0)
    rmse = np.sqrt(np.cumsum((X[sorted_idx] - Y[sorted_idx]) ** 2) / Y.size)
    return rmse


@njit(fastmath=True, nogil=True, parallel=True, cache=True)
def MSE(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return ((X - Y) ** 2).mean()


@njit(fastmath=True, nogil=True, parallel=True, cache=True)
def MSE_grad(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return (2 / X.size) * (X - Y)


def RMSE_grad(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return (X - Y) / (X.size * np.sqrt(1 / X.size * (X - Y) ** 2) + 1)


########################################
#                 plot                 #
########################################


def plot_correlation_and_cumulate(A: np.ndarray, B: np.ndarray) -> None:
    """
    Evaluate the quality of A compare with B by plotting two figures:
    A correlation figure shows sorted elements' correlation between A and B.
    A cumulate RMSE figure shows sorted cumulated RMSE of all elements.

    :param A: np.ndarray
    :param B: np.ndarray
    :return: None
    """
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    plt.xlim([0.5, 2.7])
    plt.ylim([0.5, 2.7])
    x1, y1 = [0.5, 2.7], [0.5, 2.7]
    ax.plot(x1, y1, "black", ms=1, label="y = x base line", alpha=0.3)
    ax.scatter(np.sort(A.reshape(-1, 1)), np.sort(B.reshape(-1, 1)),
               c="red", label="ref - A_h", s=0.01)
    plt.xlabel("Energy of reference (cm-1)")
    plt.ylabel("Energy of decomposition (cm-1)")
    ax.legend()

    ax2 = fig.add_axes([0.2, 0.55, .25, .25])
    ax2.scatter(np.sort(A.reshape(-1, 1)[0: 100]), np.sort(B.reshape(-1, 1)[0: 100]),
                c="red", label="A_h", s=0.1)
    x1, y1 = [1.1, 1.5], [1.1, 1.5]
    ax2.plot(x1, y1, "black", ms=1, label="", alpha=0.3)
    plt.title("First 100")

    plt.show()

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111)
    rmse = cum_rmse(A, B)
    print("finish computing")
    plt.plot(rmse, c="tab:blue")
    plt.legend()

    plt.xlabel("Points number")
    plt.ylabel("Cumulate RMSE (cm-1)")
    plt.yscale("log")

    plt.show()


def plot_difference_of_matrix(A: np.ndarray, B: np.ndarray, vmax=1, vmin=0):
    """
    Plot two

    :param A:
    :param B:
    :param vmax:
    :param vmin:
    :return:
    """
    fig = plt.figure(figsize=(10, 50), dpi=300)
    ax = plt.subplot(131)
    img1 = plt.imshow(A, vmax=vmax, vmin=vmin)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="10%", pad=0.3)
    plt.colorbar(img1, cax=cax, orientation="horizontal")

    ax = plt.subplot(132)
    img1 = plt.imshow(B, vmax=vmax, vmin=vmin)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="10%", pad=0.3)
    plt.colorbar(img1, cax=cax, orientation="horizontal")

    ax = plt.subplot(133)
    img1 = plt.imshow(np.absolute(A - B), cmap="binary")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="10%", pad=0.3)
    plt.colorbar(img1, cax=cax, orientation="horizontal")

    plt.show()
