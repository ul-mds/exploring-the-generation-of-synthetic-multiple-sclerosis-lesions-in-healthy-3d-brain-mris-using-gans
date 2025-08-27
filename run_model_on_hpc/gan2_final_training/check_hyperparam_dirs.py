import os
import sys

def read_hyperparams(txt_file):
    with open(txt_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield tuple(stripped.split())

def main(base_path, hyperparam_txt):
    missing = []
    for l1, l2, fold in read_hyperparams(hyperparam_txt):
        dir_name = f"gan2_left_out_fold{fold}_L1_{l1}_L2_{l2}"
        full_path = os.path.join(base_path, dir_name)
        print(f"Checke: {full_path}")
        if not os.path.isdir(full_path):
            missing.append((l1, l2, fold))
    print("Fehlende Hyperparam-Kombinationen (L1 L2 Fold):")
    for l1, l2, fold in missing:
        print(f"{l1} {l2} {fold}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_hyperparam_dirs.py <base_path> <hyperparam_txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
