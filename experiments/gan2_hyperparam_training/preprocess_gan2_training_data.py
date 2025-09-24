""" 
This script preprocessed the training images for Gan2. These are Flair images, which 
are used as the target_input to GAN2. Secondly they are also used with the additional lesion masks 
on top as source_input for GAN2.

After preprocessing all Images shall be: 
* bias corrected 
* scull stripped
* registered to mni152_1mm template (registration includes resampling to 256³)

The ImgPreprocessing class is used to apply these steps. Which steps are applied depends 
on the dataset and how much preprocessing has been already applied: 

MSSEG:  
    is already denoised, registered, brain extracted, bias field corrected
    -> only registration and to mni152 is applied bias correction is applied.

UMCL 
    is already coregistered and bias corrected.
    -> brain extraction, bias correction and registration is applied.

MSSEG-2 
    is already coregistered. 
    -> all steps are applied.

All lesion masks are dilated by 1 voxel to ensure that the GAN2 can learn the lesion boundaries better.
And the 3d images are sliced into 2D slices, which are then used as input to GAN2.
Note, bias correction is applied anyways, even if the images are already bias corrected.
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

process_umcl = False
process_msseg = False
process_msseg2 = True

# add toplevel directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_preprocessing import ImgPreprocessing

results_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/train_gan2_data/final_preprocessing_18092025")
#------------------------------------------------------------
#                k == 5 fold cross validation:        -------
#------------------------------------------------------------

def get_all_patients(): 
    """ 
    Returns a list of all patients in the dataset. Kind of hard coded, but this is fine for now.
    """
    msseg2_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/train_gan2_data/msseg2")
    umcl_patients = [f"patient{k:02}" for k in range(1, 31)]
    msseg_patients = [(p, c) for p in range(1, 6) for c in [1, 7]]
    msseg2_patients = [name for name in os.listdir(msseg2_dir)] 
               
    all_patients = []
    for pid in umcl_patients:
        all_patients.append(('umcl', pid))
    for p, c in msseg_patients:
        all_patients.append(('msseg', (p, c)))
    for pid in msseg2_patients:
        all_patients.append(('msseg2', pid))

    return all_patients

def k_fold_patients(all_patients, k=5, random_state=42):
    """ 
    Divides the patients into k folds for cross validation.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_dirs = [Path(results_dir) / f"fold_{i}" for i in range(k)]
    for fold_dir in fold_dirs:
        fold_dir.mkdir(parents=True, exist_ok=True)
        # create subdirectories for each fold
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
    
    # Assert: Folds müssen disjunkt sein
    all_patients_in_folds = [p for fold in all_folds.values() for p in fold]
    assert len(all_patients_in_folds) == len(set(all_patients_in_folds)), "Folds are not disjoint!"
    
    return all_folds

   
all_patients = get_all_patients()
folds = k_fold_patients(all_patients, k=5, random_state=42)

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

    for k in range(13,31):
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
        
        # egister ls_mask and flair to mni, using the transform from T1.
        msseg_patient.apply_mni_transform(image_key='ls_mask') 
        msseg_patient.apply_mni_transform(image_key='flair') 
        
        # dialate lesion_msk
        msseg_patient.dilatate_lesionmasks()

        # slice the images and generate gan2 input
        msseg_patient.create_gan2_source_target(fold=fold_idx)

#-------------------------------------------------------------
#                msseg-2 data:                           -------
#-------------------------------------------------------------
if process_msseg2:
    msseg2_flairs_path = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/msseg2/orig_flair_mris")
    msseg2_lesions_path = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/msseg2/orig_lesionmasks")

    # expected naming: msseg2_p019_flair.nii.gz, msseg2_p026_consensus.nii.gz
    for patient in list(msseg2_flairs_path.glob("*.nii.gz")):
        pid = re.search(r"msseg2_p(\d+)_flair", patient.name).group(1)
        patient_id = ('msseg2', pid)

        # find fold for cross validation, some imgs are not used for training gan2, scip them. 
        try:
            fold_idx = next(fold for fold, patients in folds.items() if patient_id in patients)
        except StopIteration:
            print(f"Patient {patient_id} not found in folds, skipping.")
            continue

        flair_path = msseg2_flairs_path / f"msseg2_p{pid}_flair.nii.gz"
        ls_path = msseg2_lesions_path / f"msseg2_p{pid}_consensus.nii.gz"

        msseg2_patient = ImgPreprocessing(name=f"msseg2_patient{pid}",
                                        result_dir=results_dir,
                                        flair_path=flair_path,
                                        ls_path=ls_path)
        
        # images are only registered, everything else needs to be applied.

        # bias correction
        msseg2_patient.bias_correction(image_key='flair')
        msseg2_patient.bias_correction(image_key='ls_mask')

        # brain extraction
        msseg2_patient.extract_brain(image_key='flair')

        # register and save the transform.
        # Note: msseg2 does not have T1 images, so we register flair to mni152_1mm template.
        msseg2_patient.calc_transform_mni152('flair')

        # apply the transform to flair and ls_mask  
        msseg2_patient.apply_mni_transform(image_key='flair')
        msseg2_patient.apply_mni_transform(image_key='ls_mask')

        # dialate lesion_msk
        msseg2_patient.dilatate_lesionmasks()

        # slice the images and generate gan2 input
        msseg2_patient.create_gan2_source_target(fold=fold_idx)