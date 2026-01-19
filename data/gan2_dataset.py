""" 
Dataset class for GAN2 training/testing on 2D FLAIR slices with lesion masks.

Author/Mod: Franziska Röpke (added 2025-01-15)

Expected dataroot structure:
    registered_flairs/        (*.npy)
    registered_lesionmasks/   (*.npy)
Filenames must match (same basename for image and mask).

Data assumptions:
- Inputs are already 2D slices saved as numpy arrays (.npy).
- Only slices containing lesion voxels are used.
- Lesion masks are dilated, binary in {0,1}.
- FLAIR images are grayscale in [0,255] and normalized to [-1,1].

Model I/O:
- source: 2 channels
    (1) FLAIR with lesion region removed/zeroed
    (2) lesion mask
- target: original FLAIR (training: real lesions; testing: synthetic lesions)

If the GAN2 preprocessing script is used, the data should already be in the correct format (see experiments/gan2_hyperparameter_training).
"""

import os
import torch
import nibabel as nib
from data.base_dataset import BaseDataset
import numpy as np
from PIL import Image

class Gan2Dataset(BaseDataset):

    K = 5  # cross-validation folds

    @staticmethod
    def modify_commandline_options(parser):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """
        return parser.add_argument('--L1_mask_only', action='store_true', help='Computes L1 only for mask areas.')
         


    def __init__(self, opt) -> None:

        """
        Initializes the dataset class.  
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        Returns:
            a dataset class instance
        """
        super().__init__(opt)

        self.samples = []
        self.original_shape = None
        self.phase = opt.phase
        self.fold_idx = opt.k_fold if opt.k_fold >= 0 else -1

        if self.phase == "inference" or (self.fold_idx == -1 and self.phase == "train"):
            # no k-fold validation.
            flair_dir = os.path.join(opt.dataroot, "flair_slices")
            lsmsk_dir = os.path.join(opt.dataroot, "mask_slices")
            for fname in os.listdir(flair_dir):
                flair_path = os.path.join(flair_dir, fname)
                lsmsk_path = os.path.join(lsmsk_dir, fname.replace("flair", "lsmsk"))
                self.samples.append(
                            {
                            "fold" : None,
                            "basename": fname.replace(".npy", "").replace("_flair", ""),
                            "flair_path": flair_path,
                            "lsmsk_path": lsmsk_path
                            }) 
            return

        elif self.phase == 'test':
            folds = [self.fold_idx]
        elif self.phase == 'train' and self.fold_idx >= 0:
            folds = [i for i in range(self.K) if i != self.fold_idx]

        for fold in folds: 
            flair_dir = os.path.join(opt.dataroot, f"fold_{fold}", "flair_slices")
            lsmsk_dir = os.path.join(opt.dataroot, f"fold_{fold}", "mask_slices")

            # naming: umcl_patient01_flair_slice_051.npy umcl_patient25_mask_slice_112.npy
            for fname in os.listdir(flair_dir):
                if fname.endswith(".npy"):
                    flair_path = os.path.join(flair_dir, fname)
                    lsmsk_path = os.path.join(lsmsk_dir, fname.replace("flair", "lsmsk"))
                    if os.path.exists(lsmsk_path):
                        self.samples.append(
                            {
                            "fold" : fold,
                            "basename": fname.replace(".npy", "").replace("_flair", ""),
                            "flair_path": flair_path,
                            "lsmsk_path": lsmsk_path
                            })
    

    def __getitem__(self, index: int) -> dict:
        """
        Loads an image pair (source & target) and converts them into tensors.
        Parameters:
            index (int) -- index of the image pair to load
        Returns:
            a dictionary containing the source and target images as tensors,
            along with their file paths.
        """

        sample = self.samples[index]

        flair_img = np.load(sample["flair_path"])
        lsmsk_img = np.load(sample["lsmsk_path"])
        
        assert np.unique(lsmsk_img).tolist() == [0, 1], "Lesion mask should be binary with values 0 and 1."

        self.original_shape = lsmsk_img.shape

        # mask lesions in flair img with zeros
        masked_flair = flair_img.copy()
        masked_flair[lsmsk_img != 0] = 0

        #empty channel
        empty = np.zeros_like(flair_img)

        # scale flair to [-1, 1]
        masked_flair_scaled = masked_flair.astype(np.float32) / 127.5 - 1.0
        flair_img_scaled     = flair_img.astype(np.float32) / 127.5 - 1.0

        # Source tensor: 3-channel input [3, H, W]
        # Channel 0: FLAIR image with lesion regions set to zero (masked)
        # Channel 1: Binary lesion mask (0=background, 1=lesion)
        # Channel 2: Empty channel (zeros)
        
        source_array = np.stack([masked_flair_scaled, lsmsk_img, empty], axis=0)
        source_tensor = torch.tensor(source_array, dtype=torch.float32)

        if self.phase == "inference" or self.phase == "test": 
             return {"A": source_tensor, "A_paths": sample["flair_path"]}
        else: 
            # Target tensor: 3-channel ground truth [3, H, W]
            # Channel 0: Original FLAIR image (unmasked)
            # Channel 1: Binary lesion mask (0=background, 1=lesion)
            # Channel 2: Empty channel (zeros)
            target_array = np.stack([flair_img_scaled, lsmsk_img, empty], axis=0)
            target_tensor = torch.tensor(target_array, dtype=torch.float32)

            # Brain mask: binary tensor identifying non-background regions (non-zero FLAIR voxels)
            brainmask_tensor = torch.tensor(flair_img != 0, dtype=torch.float32)

            return {"A": source_tensor, "B": target_tensor, "mask": brainmask_tensor, 
                        "A_paths": sample["flair_path"], "B_paths": sample["flair_path"]}


    def __len__(self):
        return len(self.samples)