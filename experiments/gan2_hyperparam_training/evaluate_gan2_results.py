""" 
Scripts loads in the results of GAN2 testing, after running the GAN2 test script. 
It reconstructs the 3D images from the resulting slices and calculats the MSE as a
measure for the quality of the results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
import nibabel as nib
from pathlib import Path
import pickle
from PIL import Image
import argparse
import json
from experiment_utils import reconstruct_3d_from_slices, apply_gaussian_smoothing


def get_basename(center, patient):
    if center == "umcl" and isinstance(patient, str):
        num = ''.join(filter(str.isdigit, patient))
        return f"umcl_patient{int(num):02}"
    elif center == "msseg2" and isinstance(patient, str):
        return f"msseg2_patient{int(patient):03}"
    elif center == "msseg" and isinstance(patient, list) and len(patient) == 2:
        p, c = patient
        return f"msseg_patient{int(p):02}_center{int(c):02}"
    else:
        return f"{center}_patient{patient}"

def get_image_dict(training_path):
    """ 
    Loads the k_fold_patients.json file and creates a dictionary that maps each fold to a list of basenames.
    """
    with open((training_path / "k_fold_patients.json"), "r") as f:
        folds = json.load(f)

    # Dictionary: für jeden Fold eine Liste der Basenames
    basenames_per_fold = {}
    for fold, patients in folds.items():
        basenames_per_fold[fold] = [get_basename(center, patient) for center, patient in patients]
    return basenames_per_fold    

def calc_mse(output_path, reference_flair, ref_mask_path):
    """
    Calculates the MSE between the output image and the reference flair image.
    Also calculates the MSE for the area of the dilated lesion mask and the brain region.
    """
    # Load images
    output_img = nib.load(output_path).get_fdata()
    ref_flair_img = nib.load(reference_flair).get_fdata()
    ref_mask_img = nib.load(ref_mask_path).get_fdata()

    # Calculate MSE for the whole image
    mse_img = np.nanmean((ref_flair_img.flatten() - output_img.flatten()) ** 2)

    # Calculate MSE for the dilated lesion mask regions
    lesion_mask = ref_mask_img > 0
    mse_dialated_lsmask = np.nanmean((ref_flair_img[lesion_mask] - output_img[lesion_mask])**2)

    # Calculate MSE for the brain region
    brain_region = ref_flair_img > 0  # Assuming brain region is where the flair image has non-zero values
    mse_brain = np.nanmean((ref_flair_img[brain_region] - output_img[brain_region])**2)

    return mse_img, mse_dialated_lsmask, mse_brain


parser = argparse.ArgumentParser(description='GAN2 Eval Script.')
parser.add_argument('--min_line', type=int, required=True, help='minimum line of gan2_hyperparams to read in. Defining window of models to evaluate.')
parser.add_argument('--max_line', type=int, required=True, help='Max line of gan2_hyperparams to read in. Defining window of models to evaluate.')
parser.add_argument('--results_name', type=str, required=True, help='Name for eval_results pickle file.')
args = parser.parse_args()


test_results_dir = Path("/home/sc.uni-leipzig.de/iu71daji/thesis_gan2/testresults")
train_data_dir = Path("/home/sc.uni-leipzig.de/iu71daji/thesis_gan2/train_gan2")

gan2_hyperparams = (train_data_dir / "gan2_hyperparams.txt")
min_lines = args.min_line
max_lines = args.max_line
all_results = {}
basenames_imgs = get_image_dict(train_data_dir) # dict with img_basenames per fold. keys "fold_0" .. 
with open(gan2_hyperparams, "r") as f:
    for idx, line in enumerate(f):
        if idx < min_lines: 
            continue
        elif idx >= max_lines: 
            break
        parts = line.strip().split()
        lambda_L1, lambda_L2, fold = parts
        gan_name = f"gan2_left_out_fold{fold}_L1_{lambda_L1}_L2_{lambda_L2}"

        # add epochs 
        for epoch in [100, 200, 300, 400]:
            test_result_imgs = (test_results_dir / gan_name / f"test_{epoch}")
            key = f"gan2_L1_{lambda_L1}_L2_{lambda_L2}_epoch_{epoch}"
            for img_name in basenames_imgs[f"fold_{fold}"]:
                path_3d_img = (test_results_dir / gan_name / f"test_{epoch}" / "reconstructed_3d" / f"{img_name}_testresult_reconstructed.nii.gz")
                path_3d_img.parent.mkdir(parents=True, exist_ok=True)
                reference_flair = (train_data_dir / "interim_results" / f"{img_name}_flair_mni_registered.mgz") 
                ref_mask_path = (train_data_dir / "interim_results"/ f"{img_name}_ls_mask_dilated.nii.gz")
                postprocessed_path = Path(test_results_dir / gan_name / f"test_{epoch}" / "postprocessed")
                postprocessed_path.mkdir(parents=True, exist_ok=True)

                # Remove possible .nii file if it exists (without .gz) TODO delete this later
                nii_path = Path(str(path_3d_img).replace('.nii.gz', '.nii'))
                if nii_path.exists():
                    nii_path.unlink()
                if not path_3d_img.exists():
                    reconstruct_3d_from_slices(slice_dir=test_result_imgs,
                                            reference_flair_path=reference_flair, 
                                            output_path=path_3d_img, 
                                            prefix=img_name,
                                            ref_mask_path=ref_mask_path, 
                                            fold=fold,
                                            delete_slices=True)
                    
                    apply_gaussian_smoothing(reconstructed_img=path_3d_img,
                                             output_path=postprocessed_path, 
                                             basename=img_name,
                                             ls_mask_path= ref_mask_path
                                            )
                else:
                    print(f"Reconstructed 3d img {path_3d_img} already exists, skipping reconstruction.")
                
                # calc mse
                mse_img, mse_dialated_lsmask, mse_brain = calc_mse(path_3d_img, reference_flair, ref_mask_path)
                if key not in all_results.keys():
                    all_results[key] = {"mse_img": [mse_img],
                                        "mse_dialated_lsmask": [mse_dialated_lsmask],
                                        "mse_brain": [mse_brain]
                                        }
                else:
                    all_results[key]["mse_img"].append(mse_img)
                    all_results[key]["mse_dialated_lsmask"].append(mse_dialated_lsmask)
                    all_results[key]["mse_brain"].append(mse_brain)                                   
            if fold == "4": 
                # all 4 folds for current config were evaluated
                # calculate mean mse
                all_results[key]["mean_mse_(img,lsmsk,brain)"] = [
                    np.nanmean(all_results[key]["mse_img"]),
                    np.nanmean(all_results[key]["mse_dialated_lsmask"]),
                    np.nanmean(all_results[key]["mse_brain"])
                ]
    
            with open(f"{args.results_name}.pkl", "wb") as f:
                pickle.dump(all_results, f)
                print("mse results saved successfuly to: ", f)

    """
    test_results naming: 
    msseg_patient01_center01_flair_slice_002_truth.png

    ... fake.png

    ref_flair: 
    interim_results/msseg_patient02_center01_flair_mni_registered.mgz

    ref_lesion_mask: 
    interim_results/msseg_patient02_center01_ls_mask_dilated.nii

    base names: 
    name=f"msseg_patient{p:02}_center{c:02}"
    name=f"msseg2_patient{pid}
    name=f"umcl_patient{k:02}
    """
