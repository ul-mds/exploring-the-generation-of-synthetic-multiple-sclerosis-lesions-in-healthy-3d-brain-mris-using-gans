import itertools, random

ALL_LAMBDA_L1_LESIONS = [50, 100, 150, 200, 250]
ALL_LAMBDA_L2_BACKGROUND = [0, 20, 50, 70, 100]
ALL_UPSAMPLING = ['deconvolution', 'linear']
ALL_NGF = [16, 32, 64]
FOLDS = [0, 1, 2, 3, 4]
combos = list(itertools.product(
    ALL_LAMBDA_L1_LESIONS,
    ALL_LAMBDA_L2_BACKGROUND,
    ALL_UPSAMPLING,
    ALL_NGF
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
