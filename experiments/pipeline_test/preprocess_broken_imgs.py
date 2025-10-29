""" 
Preprocess test data for GAN1 model and the whole pipeline.
Script processes the test images, which are MRIs of healthy patients, without lesions.
The aseg.mgz file is expected to exist as a result of the recon-all preprocessing step.
Results are the gan1_source_imgs. The test data is split randomly into 2 subsets, one is
used for hyperparameter tuning of gan1 and the other for final testing of the pipeline.
"""

import sys
import os
from pathlib import Path
from itertools import product
import re
from itertools import combinations
import numpy as np
import shutil
import random

# add toplevel directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_preprocessing import ImgPreprocessing

results_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/test_pipeline_data/preprocessing_results_28102025")
freesurfer_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/test_pipeline_data/freesurfer_results06082025")
############################################################################
#                                 ICBM                                  #    
############################################################################

icbm_images = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/icbm")
icbm_freesurfer = freesurfer_dir / "freesurfer_results_icbm"

dir  = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/icbm/UTHC_2008")

t1_path = None
flair_path = None
for file in dir.iterdir():
    if file.suffix == '.nii' and 'T1' in file.name:
        t1_path = file
    elif file.suffix == '.nii' and 'FLAIR' in file.name:
        flair_path = file
    if t1_path and flair_path:
        break

aseg_path = icbm_freesurfer / f"icbm_patient{dir.name}" / "mri" / "aseg.mgz"
icbm_patient = ImgPreprocessing(name=f"icbm_patient{dir.name}",
                                    result_dir=results_dir,
                                    mni_transform=None,
                                    flair_path=flair_path, 
                                    t1_path=t1_path,
                                    aseg_path=aseg_path
                                )
# scull strip and bias correct
icbm_patient.bias_correction(image_key='t1')
icbm_patient.bias_correction(image_key='flair')

#icbm_patient.extract_brain(image_key='flair')
#icbm_patient.extract_brain(image_key='t1')
icbm_patient.extract_brain_ants(image_key='t1')
icbm_patient.extract_brain_ants(image_key='flair')

# register T1 and save the transform.
icbm_patient.calc_transform_mni152('t1')

# register flair to mni (used later to create gan2 input from synth_ls_mask and registered flair)
icbm_patient.apply_mni_transform(image_key='flair')

# extract wm from aseg and register to mni
icbm_patient.extract_wm_mask()
icbm_patient.apply_mni_transform(image_key='wm_mask')

icbm_patient.create_gan1_source()   

##############################################################################
##                                 PPMI                                  ##    
###############################################################################


ppmi_images2 = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/test_pipeline_data/ppmi/ppmi_siemens")
patient_id= "377547"



patient_name = f"ppmi_patient{patient_id}"

flair_path = None
t1_path = None
aseg_path = freesurfer_dir / "freesurfer_results_ppmi" / "siemens" / f"ppmi_patient{patient_id}" / "mri" / "aseg.mgz"
for file in ppmi_images2.iterdir():
    if file.suffix == '.nii' and 'T1' in file.name and patient_id in file.name:
        t1_path = file
    elif file.suffix == '.nii' and 'FLAIR' in file.name and patient_id in file.name:
        flair_path = file
    if t1_path and flair_path:
        break

ppmi_patient2 = ImgPreprocessing(name=patient_name,
                                result_dir=results_dir,
                                mni_transform=None,
                                flair_path=flair_path, 
                                t1_path=t1_path,
                                aseg_path=aseg_path
                            )
# scull strip and bias correct 
ppmi_patient2.bias_correction(image_key='t1')
ppmi_patient2.bias_correction(image_key='flair')

#ppmi_patient2.extract_brain(image_key='flair')
#ppmi_patient2.extract_brain(image_key='t1')
ppmi_patient2.extract_brain_ants(image_key='t1')
ppmi_patient2.extract_brain_ants(image_key='flair')

# register T1 and save the transform.
ppmi_patient2.calc_transform_mni152('t1')

# register flair to mni (used later to create gan2 input from synth_ls_mask and registered flair)
ppmi_patient2.apply_mni_transform(image_key='flair')

# extract wm from aseg and register to mni
ppmi_patient2.extract_wm_mask()
ppmi_patient2.apply_mni_transform(image_key='wm_mask')

ppmi_patient2.create_gan1_source()