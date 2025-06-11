import sys
import os
import numpy as np
from numba import njit, prange

# sys.path.insert(0, "../../..")

# this is the file where the PES coordinates are
file_coors = os.path.join(os.path.dirname(__file__), '../../data/coeff_hono_pes')


try:
    with open(file_coors, encoding='utf-8') as file:
        coors = file.readlines()
        coors_data = np.zeros((len(coors), 7))
        for i, c in enumerate(coors):  # 389 coordinates

            # Rosmus ordering a,b,c,d,e,f,co,ii
            c = c.split()
            coors_data[i] = np.array(c).astype(np.float64)[0:7]

except:
    sys.exit("File " + file_coors + " could not be opened.")


@njit(nogil=True, parallel=True, cache=True, fastmath=True)
def energy_point(r2, r3, r1, t2, t1, t, coors_data=coors_data) -> float:
    """
    Calculate single point energy with param r2, r3, r1, t2, t1, t

    :param r2: r N=O, range(1.90, 2.60)
    :param r3: r ON, range(2.10, 3.25)
    :param r1: OH, range(1.30, 2.45)
    :param t2: u1(cos theta1), range(-0.65, -0.10)
    :param t1: u1(cos theta2), range(-0.65, 0.25)
    :param t: phi, range(0, pi)
    :param shift:

    :return: energy at one point
    """

    # value of the reference energy (hartrees)
    ene_ref = -0.200229252946

    # value of the energy
    ene = 0.

    n = coors_data.shape[0]

    # sum to all terms of the expression of the energy

    for i in prange(n):  # 389 coordinates
        # Rosmus ordering a,b,c,d,e,f,co,ii
        c = coors_data[i]

        ene += (1.0 - np.exp(-0.7 * (-2.213326 + r1))) ** c[0] \
               * (1.0 - np.exp(-0.7 * (-2.696732 + r2))) ** c[1] \
               * (1.0 - np.exp(-0.7 * (-1.82291 + r3))) ** c[2] \
               * (-1.931507 + t1) ** c[3] \
               * (-1.777642 + t2) ** c[4] \
               * np.cos(c[5] * (t - np.pi)) * c[6]

    ene *= np.exp(-(-2.2133326 + r1) ** 2) \
           * np.exp(-(-2.696732 + r2) ** 2 / 9.0) \
           * np.exp(-(-1.82291 + r3) ** 2 / 4.0) \
           * np.exp(-(-1.931507 + t1) ** 2 / 4.0) \
           * np.exp(-(-1.777642 + t2) ** 2 / 4.0)

    # energy value in hartrees (atomic units)
    energy = ene - ene_ref
    return energy


def PES_HONO(x_r2: np.ndarray,
             x_r3: np.ndarray,
             x_r1: np.ndarray,
             x_t2: np.ndarray,
             x_t1: np.ndarray,
             x_t: np.ndarray) -> np.ndarray:
    """
    Calculate PES value of HONO on each given coordinates grid

    the params of the PES are list of coordinates in each dimension

    Return: PES tensor of HONO

    :param x_r2: np.ndarray
    :param x_r3: np.ndarray
    :param x_r1: np.ndarray
    :param x_t2: np.ndarray
    :param x_t1: np.ndarray
    :param x_t: np.ndarray
    :return: np.ndarray
    """

    # single point energy

    # 1. create point coordinates in each dim

    grid = (x_r2, x_r3, x_r1, x_t2, x_t1, x_t)
    # 2. create mesh from coors
    mesh = np.meshgrid(*grid, indexing="ij")
    A = np.vectorize(energy_point)(*mesh)

    return A


def PES_HONO_4D(
        x_r3: np.ndarray,
        x_t2: np.ndarray,
        x_t1: np.ndarray,
        x_t: np.ndarray) -> np.ndarray:
    """
    Calculate 4D PES value of HONO on each given coordinates grid
    Not like PES_HONO(), it sets r1 and r2 = 2.2 .
    the params of the PES are list of coordinates in each dimension

    Return: 4D PES tensor of HONO

    :param x_r3: np.ndarray
    :param x_t2: np.ndarray
    :param x_t1: np.ndarray
    :param x_t: np.ndarray
    :return: np.ndarray
    """

    # single point energy
    def energy_point(r3, t2, t1, t) -> float:
        """
        Calculate single point energy with param r3, t2, t1, t

        :param r2: r N=O, range(1.90, 2.60)
        :param r3: r ON, range(2.10, 3.25)
        :param r1: OH, range(1.30, 2.45)
        :param t2: u1(cos theta1), range(-0.65, -0.10)
        :param t1: u1(cos theta2), range(-0.65, 0.25)
        :param t: phi, range(0, pi)
        :param shift:

        :return: energy at one point
        """
        r2 = 2.2
        r1 = 2.2
        # this is the file where the PES coordinates are
        file_coors = 'data/coeff_hono_pes'
        try:
            with open(file_coors, encoding='utf-8') as file:
                coors = file.readlines()
        except:
            sys.exit("File " + file_coors + " could not be opened.")

        # value of the reference energy (hartrees)
        ene_ref = -0.200229252946

        # value of the energy
        ene = 0.

        # sum to all terms of the expression of the energy
        for c in coors:  # 389 coordinates

            # Rosmus ordering a,b,c,d,e,f,co,ii
            c = c.split()

            ene += (1.0 - np.exp(-0.7 * (-2.213326 + r1))) ** int(c[0]) \
                   * (1.0 - np.exp(-0.7 * (-2.696732 + r2))) ** int(c[1]) \
                   * (1.0 - np.exp(-0.7 * (-1.82291 + r3))) ** int(c[2]) \
                   * (-1.931507 + t1) ** int(c[3]) \
                   * (-1.777642 + t2) ** int(c[4]) \
                   * np.cos(int(c[5]) * (t - np.pi)) * float(c[6])

        ene *= np.exp(-(-2.2133326 + r1) ** 2) \
               * np.exp(-(-2.696732 + r2) ** 2 / 9.0) \
               * np.exp(-(-1.82291 + r3) ** 2 / 4.0) \
               * np.exp(-(-1.931507 + t1) ** 2 / 4.0) \
               * np.exp(-(-1.777642 + t2) ** 2 / 4.0)

        # energy value in hartrees (atomic units)
        energy = ene - ene_ref
        return energy

    # 1. create point coordinates in each dim

    grid = (x_r3, x_t2, x_t1, x_t)
    # 2. create mesh from coors
    mesh = np.meshgrid(*grid, indexing="ij")
    a = np.vectorize(energy_point)(*mesh)
    return a


def Coulomb_PES(x_0, x_1, x_2, x_3, x_4, x_5, la=1e-1) -> np.ndarray:
    def energy_point(p_0, p_1, p_2, p_3, p_4, p_5, la=la):
        return 1 / (p_0 ** 2 + p_1 ** 2 + p_2 ** 2 + p_3 ** 2 + p_4 ** 2 + p_5 ** 2 + la) ** 0.5

    grid = (x_0, x_1, x_2, x_3, x_4, x_5)
    # 2. create mesh from coors
    mesh = np.meshgrid(*grid, indexing='ij')
    a = np.vectorize(energy_point)(*mesh)
    return a


def Quadrant(*args) -> np.ndarray:
    """
    Calculate Quadrant function

    :args: coordinates of quadrant function
    :return: np.ndarray
    """

    # single-point energy
    def energy_point(*args) -> float:
        result = 0
        for x in args:
            result += x ** 2
        return result

    # 1. create point coordinates in each dim

    grid = args
    # 2. create mesh from coors
    mesh = np.meshgrid(*grid)
    a = np.vectorize(energy_point)(*mesh)
    return a


def Quadrant_strong_coupling(*args, weight: float = 1) -> np.ndarray:
    """
    Calculate Quadrant function

    :args: coordinates of quadrant function
    :return: np.ndarray
    """

    # single-point energy
    def energy_point(*args) -> float:
        result = 0

        # quadrant function
        for x in args:
            result += x ** 2
        # add coupling
        result += weight * (args[0] * args[1] + args[0] * args[2] + args[0] * args[3] \
                            + args[1] * args[2] + args[1] * args[3] \
                            + args[2] * args[3])
        return result

    # 1. create point coordinates in each dim

    grid = args
    # 2. create mesh from coors
    mesh = np.meshgrid(*grid)
    a = np.vectorize(energy_point)(*mesh)
    return a


def Quadrant_weak_coupling(*args, weight: float = 1e-3) -> np.ndarray:
    """
    Calculate Quadrant function with normal coupling (each coupling has weight = 1)

    :args: coordinates of quadrant function
    :return: np.ndarray
    """

    # single-point energy
    def energy_point(*args) -> float:
        result = 0
        # quadrant function
        for x in args:
            result += x ** 2
        # add coupling
        result += weight * (args[0] * args[1] + args[0] * args[2] + args[0] * args[3]
                            + args[1] * args[2] + args[1] * args[3]
                            + args[2] * args[3])
        return result

    # 1. create point coordinates in each dim

    grid = args
    # 2. create mesh from coors
    mesh = np.meshgrid(*grid)
    a = np.vectorize(energy_point)(*mesh)
    return a


def Quadrant_combined_coupling(*args) -> np.ndarray:
    """
    Calculate Quadrant function with combined coupling

    :args: coordinates of quadrant function
    :return: np.ndarray
    """

    # single-point energy
    def energy_point(*args) -> float:
        result = 0
        # quadrant function
        for x in args:
            result += x ** 2
        # add coupling
        result += 1 * (args[0] * args[1]) + 0 * (
                args[0] * args[2] + args[0] * args[3] + args[1] * args[2] + args[1] * args[3] + args[2] * args[3])
        return result

    # 1. create point coordinates in each dim

    grid = args
    # 2. create mesh from coors
    mesh = np.meshgrid(*grid)
    a = np.vectorize(energy_point)(*mesh)
    return a
