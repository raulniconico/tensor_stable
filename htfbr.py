import argparse
import os
import sys
from datetime import date
import numpy as np

from predict import predict


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("mode", type=str, choices=["train", "predict"], help="Choose the mode to the model. Options: 'train' or 'predict'")
    parser.add_argument("surface", type=int, help="The surface to be tested")
    parser.add_argument("pretrain", type=str, help="The path of pretrained model to be tested")
    parser.add_argument(
        "coef",
        type=float,
        nargs="+",
        help="Coefficient of the HT-FBR, must be 12D"
    )

    args = parser.parse_args()
    assert args.mode in ["train", "predict"], "mode should not be none and must be one of 'train' or 'predict'"
    assert args.surface in [0, 1, 2, 3], "surface should be 0, 1, or 2"
    assert len(args.coef) == 12, "len(coef) should be 12"

    mode = args.mode
    surface = args.surface
    btree_path = args.pretrain
    coef = np.array(args.coef).reshape((1, -1))

    if mode == "predict":
        prediction = predict(surface, btree_path, coef)
        print(float(prediction[0, 0]))
        sys.exit(0)

if __name__ == "__main__":

    # print("Warning: You are using HT-FBR software version 0.0.1 . This version is designed and distributed only for the BETA test"
    #       "inside the Lab ISMO. The usage permission ended by 01.07.2025")

    main()
