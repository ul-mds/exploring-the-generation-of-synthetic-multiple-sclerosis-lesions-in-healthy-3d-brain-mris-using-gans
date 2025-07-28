import itertools, random

ALL_LAMBDA_L1 = [50, 100, 150, 200]
ALL_LAMBDA_L2 = [0, 20, 50, 70, 100]

combos = list(itertools.product(
    ALL_LAMBDA_L1,
    ALL_LAMBDA_L2
))
random.shuffle(combos)

with open("hyperparams.txt", "w") as f:
    for c in combos:
        f.write(" ".join(map(str, c)) + "\n")