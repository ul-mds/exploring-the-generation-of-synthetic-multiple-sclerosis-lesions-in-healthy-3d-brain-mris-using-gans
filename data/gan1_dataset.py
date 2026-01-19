""" 
Dataset class for loading Nifti images for GAN1 training/testing.

Author/Mod: Franziska Röpke (added 2025-01-15)
"""

import os
import torch
import nibabel as nib
from data.base_dataset import BaseDataset
import numpy as np

class Gan1Dataset(BaseDataset):
    """
    Dataset class for loading Nifti images for GAN training/testing. This class is written to import data to train or test the GAN1 model, which 
    is a 3D U-Net model for brain lesion generation.
    """

    @staticmethod
    def modify_commandline_options(parser: object) -> object:
        """
        Adds command line options for the dataset
        Args:
            parser: the option parser
            is_train (bool): whether training phase or test phase
        Returns:
            the modified parser.
        """
        parser.add_argument('--L1_mask_only', action='store_true', help='Computes L1 only for mask area (e.g. Brain or lesion mask).')
        parser.add_argument('--crop_imgs', action='store_true', help='Croppes the input images to a 128³ cube. Use unet_128 if true.')
        return parser

    def __init__(self, opt: object) -> None:
        
        super().__init__(opt)
                
        self.L1_mask_only = opt.L1_mask_only 
        self.crop_imgs = opt.crop_imgs 
        self.phase = opt.phase

        self.basenames = []
        if opt.phase == 'train':
            self.source_dir = os.path.join(opt.dataroot, "train_gan1", "source")
            self.target_dir = os.path.join(opt.dataroot, "train_gan1", "target")

            if os.path.isdir(self.source_dir):
                source_files = [f for f in os.listdir(self.source_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
                self.basenames = sorted([f.replace("_source_gan1.nii.gz", "").replace("_source_gan1.nii", "") for f in source_files])

        # test imgs
        elif opt.phase == 'test':
            self.test_dir = os.path.join(opt.dataroot, "gan1_test_imgs")
            if os.path.isdir(self.test_dir):
                test_files = [f for f in os.listdir(self.test_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
                self.basenames = sorted([f.replace("_source_gan1.nii.gz", "").replace("_source_gan1.nii", "") for f in test_files])

        elif opt.phase == 'inference':
            self.test_dir = opt.dataroot
            source_files = [f for f in os.listdir(self.test_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
            self.basenames = sorted([f.replace("_source_gan1.nii.gz", "").replace("_source_gan1.nii", "") for f in source_files])

        print("Debug print, basenames:") 
        print(self.basenames)


    def __getitem__(self, index: int) -> dict: 
        """
        Loads a pair of images (source & target) and converts them into tensors
        Args:
            index (int): Index of the image pair to load.
        Returns:
            dict: A dictionary containing the source and target images as tensors,
                  along with their file paths.
        """

        if self.phase == 'test' or self.phase == 'inference':
            # Test or inference mode: only load source images
            test_filename = self.basenames[index] + "_source_gan1.nii.gz"
            test_path = os.path.join(self.test_dir, test_filename)
            test_img = nib.load(test_path).get_fdata()

            # crop 256^3 to 128³: 
            if self.crop_imgs: 
                test_img = self.crop_image(test_img)

            # Convert to PyTorch tensor
            test_tensor = torch.tensor(test_img, dtype=torch.float32).unsqueeze(0)

            return {"A": test_tensor, "A_paths": test_path}
        
        else: 
            # Training mode -> load image pairs
            basename = self.basenames[index]
            
            source_path = os.path.join(self.source_dir, basename + "_source_gan1.nii")
            target_path = os.path.join(self.target_dir, basename + "_target_gan1.nii")

            source_img = nib.load(source_path).get_fdata()
            target_img = nib.load(target_path).get_fdata()

            # crop images to 128³:
            if self.crop_imgs: 
                source_img = self.crop_image(source_img)
                target_img = self.crop_image(target_img)

            # Convert to PyTorch tensors
            source_tensor = torch.tensor(source_img, dtype=torch.float32).unsqueeze(0)  # [C, H, W, D]
            target_tensor = torch.tensor(target_img, dtype=torch.float32).unsqueeze(0)
            
            if self.L1_mask_only: 
                mask_tensor = target_tensor.clone()
                return {"A": source_tensor, "B": target_tensor, "mask": mask_tensor, 
                        "A_paths": source_path, "B_paths": target_path}

            else:     
                return {"A": source_tensor, "B": target_tensor, "A_paths": source_path, "B_paths": target_path}


    def __len__(self):
        """Returns the number of images in the dataset."""
         
        return len(self.basenames) 

        
    def crop_image(self, img: np.ndarray) -> np.ndarray:
        """
        Cuts out the center of the image (trying to avoid all the background.)
        we normaly expect to have 256 size and we crop to 128, taking the middle 
        of the cube. Expects x, y, and z coordinates to have same size. 
        Args:
            img (numpy array): The input image to be cropped.
        Returns:
            numpy array: The cropped image.
        """
        
        start = (img.shape[1] - 128) // 2  
        end = start + 128  
    
        cropped_img = img[start:end, start:end, start:end]
        return cropped_img
