"""
Code snipped to generate all combinations of hyperparameters for GAN1 training.
saved as hyperparams.txt
"""

import itertools, random

ALL_LAMBDA_L1 = [50, 100, 150, 200, 250]
ALL_LAMBDA_L2 = [50, 100, 150, 200, 250]

combos = list(itertools.product(
    ALL_LAMBDA_L1,
    ALL_LAMBDA_L2
))
random.shuffle(combos)

with open("hyperparams.txt", "w") as f:
    for c in combos:
        f.write(" ".join(map(str, c)) + "\n")
