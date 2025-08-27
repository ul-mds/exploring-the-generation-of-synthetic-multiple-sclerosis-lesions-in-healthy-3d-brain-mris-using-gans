"""
In GAN2 l1 is computed for the hole brain and l2 only for the lesion mask.
Therefore Lesion-Areas conut twice in the loss function, if L2 is greater than zero.
"""
import itertools, random

ALL_LAMBDA_L1 = [50, 100, 150, 200]
ALL_LAMBDA_L2 = [0, 20, 50, 70]
FOLDS = [0, 1, 2, 3, 4]
combos = list(itertools.product(
    ALL_LAMBDA_L1,
    ALL_LAMBDA_L2
))
# Shuffle the combinations to ensure randomness
random.seed(42)
random.shuffle(combos)

with open("gan2_hyperparams.txt", "w") as f:
    for c in combos:
        # add every combination with each fold
        for fold in FOLDS:
            c_with_fold = list(c) + [fold]
            f.write(" ".join(map(str, c_with_fold)) + "\n")
