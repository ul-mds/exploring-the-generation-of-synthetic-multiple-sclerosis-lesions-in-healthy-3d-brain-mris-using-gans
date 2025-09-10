""" 
This script preprocesses the train / test images for Gan1. 
Things to do for train_images are: 
    * Coregister the images (Flair, T1 and lesionmasks) with each other (if they are'nt already).
    * Compute the transform for registering T1 on mni_template.
    * Run recon-all on the images.
    * Register the aseg.mgz (segmentation, outputed by recon-all) and the lesion_mask to mni_template.
        The registered lesion_mask is the target_gan1 image for training.
    * Generate source_gan1 from the registered aseg file.
Same pipeline for test_images, only difference: no lesion_mask to register and no target_gan1 image 
to generate. The chosen registration is to a mni152_template, resampled on 256^3 and a affine registration 
with 12 degrees of freedom is performed. 
"""

import sys
import os
from pathlib import Path
from itertools import product
import re
from itertools import combinations

# add toplevel directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_preprocessing import ImgPreprocessing

# general paths 

results_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/train_gan1_data/preprocessing_results_04092025")
if not results_dir.exists():
    results_dir.mkdir(parents=True, exist_ok=True)

#-------------------------------------------------------------
#                umcl data:                           -------
#------------------------------------------------------------
t1_paths = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/umcl/orig_tw1_mris")
flair_paths = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/umcl/orig_flair_mris")
ls_paths = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/umcl/orig_lesionmasks")
umcl_freesurfer = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/train_gan1_data/freesurfer_results_umcl")


for flair_path in flair_paths.iterdir():
    # expected naming: patient12_consensus_gt.nii, patient12_T1W.nii, patient12_FLAIR.nii
    patient = re.match(r"(patient\d+)", str(flair_path.name)).group(1)
    t1_path = t1_paths / f"{patient}_T1W.nii.gz"
    ls_path = ls_paths / f"{patient}_consensus_gt.nii.gz"
    aseg_path = umcl_freesurfer / f"umcl_{patient}" / "mri" / "aseg.mgz"

    umcl_patient = ImgPreprocessing(name= f"umcl_{patient}",
                                    result_dir=results_dir,
                                    mni_transform=None, 
                                    flair_path=flair_path, 
                                    t1_path=t1_path,
                                    ls_path=ls_path,
                                    aseg_path=aseg_path
                                   )
    
    # 1. Coregister: umcl images are already coregistered, we skip this step.

    # 2. register T1 and save the transform.
    umcl_patient.calc_transform_mni152('t1')
    
    # 3. register ls_mask to mni
    umcl_patient.apply_mni_transform(image_key='ls_mask') 

    # extract wm_mask from aseg and apply the mni transform (note: registering aseg first would destroy the labels)
    umcl_patient.extract_wm_mask()   
    umcl_patient.apply_mni_transform(image_key='wm_mask')  
    
    # 4. generate source_img and target_img
    umcl_patient.create_gan1_source()
    umcl_patient.create_gan1_target()
 

#-------------------------------------------------------------
#                msseg data:                           -------
#-------------------------------------------------------------

patient_ids = range(1,6)
center_ids = [1,7,8]
combinations = list(product(patient_ids, center_ids))

#msseg_freesurfer = Path(r"C:\Users\T14\Documents\Masterthesis\Data\train_gan1_data\freesurfer_results_msseg")
#t1_paths = Path(r"C:\Users\T14\Documents\Masterthesis\Data\original_data\msseg\orig_tw1_mris")
#flair_paths = Path(r"C:\Users\T14\Documents\Masterthesis\Data\original_data\msseg\orig_flair_mris")
#ls_paths = Path(r"C:\Users\T14\Documents\Masterthesis\Data\original_data\msseg\orig_lesionmasks")

msseg_freesurfer = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/train_gan1_data/freesurfer_results_msseg")
t1_paths = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/msseg/orig_tw1_mris")
flair_paths = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/msseg/orig_flair_mris")
ls_paths = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/msseg/orig_lesionmasks")


for p, c in combinations:
    # expected naming: patient03_center_01_T1W.nii, patient01_center_08_FLAIR.nii, patient02_center_01_consensus_gt.nii
    
    filename = f"patient{p:02}_center_{c:02}"
    t1_path = t1_paths / f"{filename}_T1W.nii.gz"
    flair_path = flair_paths / f"{filename}_FLAIR.nii.gz"
    ls_path = ls_paths / f"{filename}_consensus_gt.nii.gz"
    aseg_path = msseg_freesurfer / f"msseg_patient{p:02}_center{c:02}" / "mri" / "aseg.mgz"

    msseg_patient = ImgPreprocessing(name=f"msseg_patient{p:02}_center{c:02}",
                                    result_dir=results_dir,
                                    mni_transform=None,
                                    flair_path=flair_path,
                                    t1_path=t1_path,
                                    ls_path=ls_path,
                                    aseg_path=aseg_path
                                 )
    
    # 1. Coregister: msseg images are already coregistered, we skip this step.

    # 2. register T1 and save the transform.
    msseg_patient.calc_transform_mni152('t1')

    # 3. register ls_mask to mni
    msseg_patient.apply_mni_transform(image_key='ls_mask') 

    # extract wm_mask from aseg and apply the mni transform (note: registering aseg first would destroy the labels)
    msseg_patient.extract_wm_mask()   
    msseg_patient.apply_mni_transform(image_key='wm_mask')  
    
    # 4. generate source_img and target_img
    msseg_patient.create_gan1_source()
    msseg_patient.create_gan1_target()