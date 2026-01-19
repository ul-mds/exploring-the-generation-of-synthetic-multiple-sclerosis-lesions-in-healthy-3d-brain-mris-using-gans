
""" 
Preprocessing-Script für GAN1 Evaluation dataset.

Author/Mod: Franziska Röpke (added/modified 2026-01-15)

Script converts .nhdr files to .nii.gz and registers lesion masks to MNI space.
to register the lesion masks, the corresponding T1 images are used.
T1 images are bias-corrected and brain-extracted before calculating the transform to MNI space.
transforms are then applied to the lesion masks.
In total 25 images are processed, 5 from ISBI and 20 from NITRC dataset. 

ISBI Images are already brain-extracted, so no brain extraction is performed here.
"""

import SimpleITK as sitk
from pathlib import Path
import sys
import os
import logging
import re

logging.basicConfig(
    filename="gan1_eval_masks_preprocessing.log",
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_preprocessing import ImgPreprocessing

results_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/gan1_eval_lesion_masks/preprocessing_results_02112025")

preprocess_isbi = True
preprocess_nitrc = False

######################################################################################
#                     NITRC data                                                     # 
######################################################################################

convert_to_nifti = False
register_masks = False
move_registered_masks = False

if preprocess_nitrc:
    base_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/gan1_eval_lesion_masks/DataMSLesion2008/Data_used")

    # 1. collect Patient-IDs
    patient_ids = set()
    for cohort in ["CHB_train", "UNC_train"]:
        cohort_dir = base_dir / cohort
        for patient_dir in cohort_dir.iterdir():
            if patient_dir.is_dir():
                patient_ids.add(patient_dir.name)

# 2. Conversion for all patients and modalities
if convert_to_nifti and preprocess_nitrc:
    for cohort in ["CHB_train", "UNC_train"]:
        cohort_dir = base_dir / cohort
        for patient_id in sorted(patient_ids):
            for modality in ["FLAIR", "T1", "T2", "lesion"]:
                nhdr_file = cohort_dir / f"{patient_id}_{modality}.nhdr"
                nii_file = cohort_dir / f"{patient_id}_{modality}.nii.gz"
                if nhdr_file.exists():
                    img = sitk.ReadImage(str(nhdr_file))
                    sitk.WriteImage(img, str(nii_file))
                    logging.info(f"Converted {nhdr_file} -> {nii_file}")
                else:
                    logging.warning(f"Missing: {nhdr_file}")


# preprocessing: register lesion masks to MNI space.
if register_masks and preprocess_nitrc:
    for cohort in ["CHB_train", "UNC_train"]:
        cohort_dir = base_dir / cohort
        
        for patient_id in sorted(patient_ids):
            t1_file = cohort_dir / f"{patient_id}_T1.nii.gz"
            lesion_file = cohort_dir / f"{patient_id}_lesion.nii.gz"
            registered_lesion_file = cohort_dir / f"{patient_id}_lesion_registered.nii.gz"

            if not t1_file.exists() or not lesion_file.exists():
                logging.warning(f"Missing T1 or lesion file for patient {patient_id} in cohort {cohort}")
                continue

            patient = ImgPreprocessing(name=f"nitrc_{patient_id}",
                                            result_dir=results_dir,
                                            t1_path=t1_file,
                                            ls_path=lesion_file
                                        )
            patient.bias_correction(image_key='t1')
            patient.extract_brain_ants(image_key='t1')
            patient.calc_transform_mni152('t1')
            patient.apply_mni_transform(image_key='ls_mask')


######################################################################
#                  ISBI data                                         #
######################################################################

if preprocess_isbi: 
    isbi_orig_imgs = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/gan1_eval_lesion_masks/isbi")


    for file in isbi_orig_imgs.iterdir():
        if "t2" in file.name: 
            patient_id = re.search(r'training(\d+_\d+)', file.name).group(1)
            print(patient_id)
            t2_path = file
            # get corresponding ls_mask
            pattern1 = isbi_orig_imgs / f'training{patient_id}_mask1.nii'
            pattern2 = isbi_orig_imgs / f'training{patient_id}_mask2.nii'
            if pattern1.exists():
                ls_path = pattern1
            elif pattern2.exists():
                ls_path = pattern2
            else: 
                print(f"No lesion mask found for patient {patient_id}.")
                continue
            img = ImgPreprocessing(name= f'isbi_{patient_id}', 
                                flair_path=t2_path, 
                                ls_path=ls_path, 
                                result_dir=results_dir
                                )
            # image key flair leads to t2 registration here
            img.bias_correction(image_key='flair')
            img.calc_transform_mni152(image_key='flair')
            img.apply_mni_transform(image_key='ls_mask')
