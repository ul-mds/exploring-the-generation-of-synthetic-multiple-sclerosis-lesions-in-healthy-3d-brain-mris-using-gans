import itertools, random

ALL_LAMBDA_L1_LESIONS = [50, 100, 200, 300, 400, 500]
ALL_LAMBDA_L2 = [0, 20, 50, 70, 100]
ALL_UPSAMPLING = ['deconvolution', 'linear']
ALL_NGF = [16, 32, 64]

combos = list(itertools.product(
    ALL_LAMBDA_L1_LESIONS,
    ALL_LAMBDA_L2,
    ALL_UPSAMPLING,
    ALL_NGF
))
random.shuffle(combos)

with open("hyperparams.txt", "w") as f:
    for c in combos:
        f.write(" ".join(map(str, c)) + "\n")