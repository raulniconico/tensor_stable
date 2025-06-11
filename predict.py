import os
import sys

from tensor.optim import Adam, load


import numpy as np

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def predict(surface, btree_path, coef):
    if surface == 1:
        energy_12 = np.load(resource_path("data/12D_db/energy_1.npy"))

    elif surface == 2:
        energy_12 = np.load(resource_path("data/12D_db/energy_4.npy"))

    elif surface == 3:
        energy_12 = np.load(resource_path("data/12D_db/energy_6.npy"))
    else:
        raise Exception

    btree = load(btree_path)

    # normalisation of the input
    mean = energy_12.mean()
    std = energy_12.std()

    predict = btree.forward(coef) * std + mean
    return predict
