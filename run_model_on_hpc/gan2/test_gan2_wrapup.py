""" 
Wrap up around the test.py script to test gan2. Slices a 3d Image, tests the gan2
model on it and merges the resulting slices to a 3d output image afterwards.
"""

import nibabel as nib
import numpy as np
import os
import shutil
from subprocess import run
from pathlib import Path

def split_nifti_to_slices(nifti_path, output_dir):
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()
    affine = nii.affine
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(volume.shape[2]):
        slice_i = volume[:, :, i]
        slice_nii = nib.Nifti1Image(slice_i, affine)
        nib.save(slice_nii, os.path.join(output_dir, f'slice_{i:03d}.nii.gz'))
    
    return volume.shape, affine

def merge_slices_to_nifti(slices_dir, output_path, original_shape, affine):
    slices = []
    for i in range(original_shape[2]):
        slice_path = os.path.join(slices_dir, f'slice_{i:03d}.nii.gz')
        if os.path.exists(slice_path):
            nii = nib.load(slice_path)
            slices.append(nii.get_fdata())
        else:
            print(f"Missing slice {i}")
            slices.append(np.zeros(original_shape[:2]))  # fallback
    
    volume = np.stack(slices, axis=2)
    new_nii = nib.Nifti1Image(volume, affine)
    nib.save(new_nii, output_path)

def run_test_script():
    run(['python', 'test.py', '--dataroot', 'temp_slices', '--name', 'my_model', '--model', 'pix2pix'])

if __name__ == '__main__':
    input_nifti = 'input_volume.nii.gz'
    temp_input_dir = 'temp_slices'
    temp_output_dir = 'results/my_model/test_latest/images'
    output_nifti = 'reconstructed_output.nii.gz'

    shape, affine = split_nifti_to_slices(input_nifti, temp_input_dir)

    run_test_script()

    # Assuming test.py saves each output slice as "slice_000_fake_B.nii.gz" or similar
    merge_slices_to_nifti(temp_output_dir, output_nifti, shape, affine)

    # Cleanup
    shutil.rmtree(temp_input_dir)
