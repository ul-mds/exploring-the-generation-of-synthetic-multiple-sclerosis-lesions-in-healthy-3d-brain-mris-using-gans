""" 
Script for intermediate processing in between GAN1 and GAN2. 
The GAN1 Results-Path is cleaned up: only the synthetic lesion masks are kept.

Then gan2 input gets created and the slices get saved.
"""

import numpy as np
import nibabel as nib
import argparse
from pathlib import Path
import pandas as pd

def create_gan2_source_target(basename, flair_path, lsmask_path, output_path):
        """
        Slices the 3D FLAIR and lesion mask images into 2D numpy arrays for GAN2 training.
        Only slices containing at least one lesion voxel are saved. This function is 
        codependent with the create_gan2_3d_output() function, which reconstructs the 3D images. 
        Changes made in this code should be applied to the reconstruction function as well.
        """

        flair_img = nib.load(str(flair_path)).get_fdata()
        lesion_img = nib.load(str(lsmask_path)).get_fdata()

        if flair_img.shape != lesion_img.shape:
            raise ValueError("Dimension mismatch between FLAIR and lesion mask.")

        if not np.all(np.isin(lesion_img, [0, 1])):
            raise ValueError("Lesion mask should be binary (0 or 1).")

        # save min and max for later denormalization
        csv_path = Path(output_path) / "flair_minmax.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=["name", "min", "max"])
        
        # add new row
        new_row = {"name": basename, "min": float(flair_img.min()), "max": float(flair_img.max())}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
        # Normalize flair_img to [0, 255] using min-max normalization
        flair_img = (flair_img - flair_img.min()) / (np.ptp(flair_img) + 1e-8) * 255

        # save min and max for later denormalization
        csv_path = Path(output_path) / "flair_minmax.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=["name", "min", "max"])
        
        # add new row
        new_row = {"name": basename, "min": float(flair_img.min()), "max": float(flair_img.max())}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        # create result_dirs if they don't exist yet.
        output_flairs = output_path / "flair_slices"
        output_masks = output_path / "mask_slices"
        output_flairs.mkdir(parents=True, exist_ok=True)
        output_masks.mkdir(parents=True, exist_ok=True)

        img_index = 0
        for i in range(flair_img.shape[0]): 
            flair_slice = flair_img[i, : , :]
            lesion_slice = lesion_img[i, : , :]

            # check if lesion voxels in this slice
            if np.any(lesion_slice): 
                
                # save imgs
                flair_slice_path = output_flairs / f"{basename}_flair_slice_{img_index:03d}.npy"
                lesion_slice_path = output_masks / f"{basename}_lsmsk_slice_{img_index:03d}.npy"

                np.save(flair_slice_path, flair_slice.astype(np.float32))  
                np.save(lesion_slice_path, lesion_slice.astype(np.uint8)) # save as uint8 for binary mask
                img_index += 1 

# get passed arguments
parser = argparse.ArgumentParser(description='GAN2 Preprocessing Script')
parser.add_argument('--flairs_path', type=str, required=True, help='Path to registered flairs')
parser.add_argument('--lsmasks_path', type=str, required=True, help='Path to synthetic lesion masks.')
parser.add_argument('--output_path', type=str, required=True, help='Output path for flair and mask slices.')
parser.add_argument('--gan1_epoch', type=str, required=True, help='Needed to complete path to GAN1 results.')
parser.add_argument('--gan1_name', type=str, required=True, help='Needed to complete path to GAN1 results.')
parser.add_argument('--gan1_phase', type=str, required=True, help='Needed to complete path to GAN1 results.')

args = parser.parse_args()

lsmask_path = Path(args.lsmasks_path) / args.gan1_name / f"{args.gan1_phase}_{args.gan1_epoch}" / "images"

for file in Path(args.flairs_path).glob("*.mgz"):
    basename = file.stem.replace('_flair_mni_registered', '')
    create_gan2_source_target(basename, 
                              flair_path= file, 
                              lsmask_path= lsmask_path / f"{basename}_fake_B.nii", 
                              output_path= Path(args.output_path)
                              )    
