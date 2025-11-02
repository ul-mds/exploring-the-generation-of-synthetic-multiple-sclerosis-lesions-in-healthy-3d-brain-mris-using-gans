

import SimpleITK as sitk
from pathlib import Path
import sys
import os

# add toplevel directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_preprocessing import ImgPreprocessing

convert_to_nifti = False

base_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/gan1_eval_lesion_masks/DataMSLesion2008/Data_used")
results_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/gan1_eval_lesion_masks/preprocessing_results_02112025")


# 1. Patient-IDs sammeln
patient_ids = set()
for cohort in ["CHB_train", "UNC_train"]:
    cohort_dir = base_dir / cohort
    for file in cohort_dir.glob("*.nhdr"):
        name = file.stem
        # z.B. CHB_train_Case01_FLAIR -> CHB_train_Case01
        patient_id = "_".join(name.split("_")[:-1])
        patient_ids.add(patient_id)

# 2. Konvertierung für alle Patienten und Modalitäten
if convert_to_nifti:
    for cohort in ["CHB_train", "UNC_train"]:
        cohort_dir = base_dir / cohort
        for patient_id in sorted(patient_ids):
            for modality in ["FLAIR", "T1", "T2", "lesion"]:
                nhdr_file = cohort_dir / f"{patient_id}_{modality}.nhdr"
                nii_file = cohort_dir / f"{patient_id}_{modality}.nii.gz"
                if nhdr_file.exists():
                    img = sitk.ReadImage(str(nhdr_file))
                    sitk.WriteImage(img, str(nii_file))
                    print(f"Converted {nhdr_file} -> {nii_file}")
                else:
                    print(f"Missing: {nhdr_file}")


# preprocessing: register lesion masks to MNI space.
for cohort in ["CHB_train", "UNC_train"]:
    cohort_dir = base_dir / cohort
    
    for patient_id in sorted(patient_ids):
        t1_file = cohort_dir / f"{patient_id}_T1.nii.gz"
        lesion_file = cohort_dir / f"{patient_id}_lesion.nii.gz"
        registered_lesion_file = cohort_dir / f"{patient_id}_lesion_registered.nii.gz"

        if not t1_file.exists() or not lesion_file.exists():
            print(f"Missing T1 or lesion file for patient {patient_id} in cohort {cohort}")
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

    # move the registered lesion masks to a separate directory

    registered_ls_dir = results_dir / "registered_ls_masks"
    registered_ls_dir.mkdir(parents=True, exist_ok=True)

    for patient_id in sorted(patient_ids):
        registered_lesion_file = cohort_dir / f"{patient_id}_lesion_registered.nii.gz"
        if registered_lesion_file.exists():
            new_file = registered_ls_dir / f"{patient_id}_lesion_registered.nii.gz"
            registered_lesion_file.rename(new_file)
            print(f"Moved {registered_lesion_file} -> {new_file}")
        else:
            print(f"Missing: {registered_lesion_file}")