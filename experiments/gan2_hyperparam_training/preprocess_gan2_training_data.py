"""
Preprocessing pipeline for GAN2 training data (FLAIR + lesion masks).

Author/Mod: Franziska Röpke (added/modified 2026-01-15)

FLAIR images are used as:
- target input to GAN2
- source input together with the lesion mask (mask overlaid/combined downstream)

After preprocessing, all images are:
- bias-field corrected
- skull stripped (brain extracted)
- registered to the MNI152 1mm template (incl. resampling to 256^3)

The ImgPreprocessing class applies a dataset-specific subset of steps depending on what
has already been done:

- MSSEG:
    already denoised, registered, brain-extracted, and bias-corrected
    -> apply MNI152 registration; bias correction is still applied
- UMCL:
    already co-registered and bias-corrected
    -> apply skull stripping, bias correction, and MNI152 registration
- MSSEG-2:
    already co-registered
    -> apply all steps

Lesion masks are dilated by 1 voxel to improve learning of lesion boundaries.
Finally, 3D volumes are sliced into 2D slices which are used to train GAN2.

Note: bias correction is applied even if the input is already bias-corrected.
"""

import sys
import os
from pathlib import Path
from itertools import product
import re
from itertools import combinations
import random
from sklearn.model_selection import KFold
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_preprocessing import ImgPreprocessing

def get_all_patients() -> list: 
    """ 
    Returns a list of all patients in the dataset. Kind of hard coded, but this is fine for now.
    Returns:
        list: List of all patients as tuples (center, patient_id)
    """
    umcl_patients = [f"patient{k:02}" for k in range(1, 31)]
    msseg_patients = [(p, c) for p in range(1, 6) for c in [1, 7]]

               
    all_patients = []
    for pid in umcl_patients:
        all_patients.append(('umcl', pid))
    for p, c in msseg_patients:
        all_patients.append(('msseg', (p, c)))

    return all_patients


def k_fold_patients(all_patients: list, results_dir: Path, k=5, random_state=42) -> dict:
    """ 
    Divides the patients into k folds for cross validation. Creates a 
    subdirectory for each fold.
    Parameters:
        all_patients (list): List of all patients as tuples (center, patient_id)
        results_dir (Path): Directory to create fold subdirectories in
        k (int): Number of folds
        random_state (int): Random seed for reproducibility
    Returns:
        dict: Dictionary mapping fold names to lists of patients
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_dirs = [Path(results_dir) / f"fold_{i}" for i in range(k)]

    for fold_dir in fold_dirs:
        fold_dir.mkdir(parents=True, exist_ok=True)
        ls_path = fold_dir / "mask_slices"
        ls_path.mkdir(parents=True, exist_ok=True)
        flair_path = fold_dir / "flair_slices"
        flair_path.mkdir(parents=True, exist_ok=True)

    all_folds = {"fold_0": [], 
                 "fold_1": [], 
                 "fold_2": [], 
                 "fold_3": [], 
                 "fold_4": []
                 }
    for i, (_, test_idxs) in enumerate(kf.split(all_patients)):
        for idx in test_idxs: 
            patient = all_patients[idx]
            all_folds[f"fold_{i}"].append(patient)
    
    all_patients_in_folds = [p for fold in all_folds.values() for p in fold]
    assert len(all_patients_in_folds) == len(set(all_patients_in_folds)), "Folds are not disjoint!"
    
    return all_folds


def main():
    """
    Main preprocessing pipeline for GAN2 training data.
    Processes UMCL and MSSEG datasets with k-fold cross-validation.
    """
    process_umcl = True
    process_msseg = True
    results_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/train_gan2_data/final_preprocessing_09102025")
    
    all_patients = get_all_patients()
    folds = k_fold_patients(all_patients, results_dir=results_dir, k=5, random_state=42)

    # save results to json file
    json_path = results_dir / "k_fold_patients.json"
    with open(json_path, "w") as f:
        json.dump(folds, f, indent=2)


    #-------------------------------------------------------------
    #                umcl data:                           -------
    #-------------------------------------------------------------
    if process_umcl:
        flairs_path = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/umcl/orig_flair_mris")
        lesions_path = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/umcl/orig_lesionmasks")
        t1s_path = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/umcl/orig_tw1_mris")
        brainmasks_path = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/umcl/orig_brainmasks")

        for k in range(1,13):
            # expected naming: patient12_consensus_gt.nii, patient12_FLAIR.nii
            flair_path=(flairs_path / f"patient{k:02}_FLAIR.nii.gz")
            ls_path=(lesions_path / f"patient{k:02}_consensus_gt.nii.gz")
            t1_path=(t1s_path / f"patient{k:02}_T1W.nii.gz")
            brain_path = (brainmasks_path / f"patient{k:02}_brainmask.nii.gz")
            
            umcl_patient = ImgPreprocessing(name=f"umcl_patient{k:02}",
                                            result_dir = results_dir,
                                            flair_path = flair_path,
                                            ls_path = ls_path,
                                            t1_path = t1_path,
                                            brain_path=brain_path
                                            )
            
            # find fold for cross validation  
            patient_id = ('umcl', umcl_patient.name.split('_')[1])
            fold_idx = next(idx for idx, patients in folds.items() if patient_id in patients)

            # bias correction and brain extraction
            umcl_patient.bias_correction(image_key='flair')
            umcl_patient.bias_correction(image_key='t1')

            # brain masks are given in orig dataset. We use these instead of extracting the brain.
            umcl_patient.apply_brainmask(image_key='flair')
            umcl_patient.apply_brainmask(image_key='t1')

            # register T1 and save the transform
            umcl_patient.calc_transform_mni152('t1')

            # apply the transform to flair and ls_mask
            umcl_patient.apply_mni_transform(image_key='flair')
            umcl_patient.apply_mni_transform(image_key='ls_mask')

            # dilate lesion_msk 
            umcl_patient.dilatate_lesionmasks()

            # slice the images and generate gan2 input 
            umcl_patient.create_gan2_source_target(fold=fold_idx)


    #-------------------------------------------------------------
    #                msseg data:                           -------
    #-------------------------------------------------------------
    if process_msseg:
        msseg_flairs_path = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/msseg/orig_flair_mris")
        msseg_t1s_path = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/msseg/orig_tw1_mris")
        msseg_lesions_path = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/msseg/orig_lesionmasks")

        msseg_patient_ids = range(1,6)  
        msseg_center_ids = [1,7] # # centers 1 and 7 are used for training, center 8 is not used (no siemens scanner)
        combinations = list(product(msseg_patient_ids, msseg_center_ids))
        for p, c in combinations:
            # expected naming: patient12_center_01_consensus_gt.nii.gz, patient12_center_01_FLAIR.nii.gz, patient12_center_08_T1W.nii.gz
        
            flair_path = (msseg_flairs_path / f"patient{p:02}_center_{c:02}_FLAIR.nii.gz").resolve()
            t1_path = (msseg_t1s_path / f"patient{p:02}_center_{c:02}_T1W.nii.gz").resolve()
            ls_path = (msseg_lesions_path / f"patient{p:02}_center_{c:02}_consensus_gt.nii.gz").resolve()

            msseg_patient = ImgPreprocessing(name=f"msseg_patient{p:02}_center{c:02}",
                                            result_dir=results_dir,
                                            flair_path=flair_path,
                                            t1_path=t1_path,
                                            ls_path=ls_path
                                            )
            
            # find fold for cross validation
            patient_id = ('msseg', (p, c))  
            fold_idx = next(idx for idx, patients in folds.items() if patient_id in patients)

            # register T1 and save the transform.
            msseg_patient.calc_transform_mni152('t1')
            
            # register ls_mask and flair to mni, using the transform from T1.
            msseg_patient.apply_mni_transform(image_key='ls_mask') 
            msseg_patient.apply_mni_transform(image_key='flair') 

            # dilate lesion_msk
            msseg_patient.dilatate_lesionmasks()

            # slice the images and generate gan2 input
            msseg_patient.create_gan2_source_target(fold=fold_idx)


if __name__ == "__main__":
    main()