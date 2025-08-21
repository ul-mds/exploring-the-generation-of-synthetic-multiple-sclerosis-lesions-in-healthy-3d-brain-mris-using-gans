"""" 
Script processes the evaluation data for gan1 testing. We use MS Data, which was not used during training ( because no
T1 imgs were available for these patients).
The script loads the images, computes the transform into mni space and applies the transform to the lesion_masks.
The output are transformed lesion_masks. Later on (not here) the distribution of some features will be computed on these masks 
& compared to the synthetic ones.
"""

from pathlib import Path
import shutil   
import json
import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_preprocessing import ImgPreprocessing

#########################################################################
#                                 MSSEG2                                #
#########################################################################
"""
msseg_orig_flairs = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/gan1_eval_lesion_masks/msseg2/orig_flair_mris")
msseg_orig_lsmsks = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/gan1_eval_lesion_masks/msseg2/orig_lesionmasks")
results_folder = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/gan1_eval_lesion_masks/msseg2/registered_lesionmasks")

# naming: 
# msseg2_p018_consensus.nii.gz
# msseg2_p016_flair.nii.gz
for file in msseg_orig_flairs.iterdir():
    if file.suffixes == ['.nii', '.gz']:
        flair_path = file
        patient_id = re.search(r'msseg2_p(\d+)', file.name).group(1)
        # find corresponding lesion mask
        lesion_mask = msseg_orig_lsmsks / f'msseg2_p{patient_id}_consensus.nii.gz'

        img = ImgPreprocessing(name= f'msseg2_p{patient_id}', 
                               flair_path=flair_path, 
                               ls_path=lesion_mask, 
                               result_dir=results_folder
                               )
        img.calc_transform_mni152(image_key='flair')
        img.apply_mni_transform(image_key='ls_mask')
        
         
"""         

#########################################################################
#                                 ISIBI                                 #
#########################################################################

#isbi
isbi_orig_imgs = msseg_orig_flairs = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/gan1_eval_lesion_masks/isbi")
isbi_result_folder = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/gan1_eval_lesion_masks/isbi/registered_lesionmasks")

# naming:
# training01_01_flair_pp.nii
# training01_01_mask1.nii
# training01_01_t2_pp.nii

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
                               result_dir=isbi_result_folder
                               )
        img.calc_transform_mni152(image_key='flair')
        img.apply_mni_transform(image_key='ls_mask')

        
        