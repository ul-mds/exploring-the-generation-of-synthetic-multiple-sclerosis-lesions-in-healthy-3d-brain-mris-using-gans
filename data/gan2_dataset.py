""" 
This class is meant to load in the data to train and test GAN2. Gan2 learns to create 
realistic looking synthetic lesions in a greyscale 3D MRI, from a lesion mask. 
source input: Brain MRI with lesion mask on top (areas labeld with one's where the lesions are 
supposed to be.) 
target_output: The same Brain MRI but the lesion mask areas replaced by synthetic lesions. 

The data class expects a dataroot which has two dirs: 
    * ./registered_flairs
    * ./registered_lesionmasks

names have to match and it's the same no matter if it is test or training images.

The given Images are already 2D slices and only slices which contain lesion voxels are 
 used. The Lesion masks are dialated and lesion_mask and mri have the same basename. 

 -> load them in, arrange source and target. 
        ( source: channel 1: greyscale mri, but deleted where lesion mask is + lesion_mask in channel 2)
        (traget: mri like it is in training.)

* Preprocessing:
    * slice the 3D image into 2D slices. 
    * only take slices which do contain eighter brain voxels or lesion mask voxels ? 
    * dilatation of the lesion mask to capture border artefacts. 
    * Channels: Channel1: MRI with zero where Lesions would be. 
                Channel2: Lesionmask. (0,255)
    
    Maybe the easyiest option is to slice the data and save the results in a temporary 
    directory. Than to load the pairs in again ? 

    TODO Channels and stuff doesn't matter for the moment, we care about when we load the data in. 
        TODO Same here: Then delete voxels where Lesions are in first channel..
"""



import os
import torch
import nibabel as nib
from data.base_dataset import BaseDataset
import numpy as np
from PIL import Image

class Gan2Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser


    def __init__(self, opt):
        """Initialisiert das Dataset"""
        super().__init__(opt)

        self.flair_dir = os.path.join(opt.dataroot, "flair_slices")
        self.lsmsk_dir = os.path.join(opt.dataroot, "lesion_slices")

        flair_files = [f for f in os.listdir(self.flair_dir) if f.endswith(".png")]

        # naming: umcl_patient01_flair_slice_051.png umcl_patient25_mask_slice_112.png
        self.flairnames = sorted([f.replace(".png", "") for f in flair_files])
        
        self.is_test = opt.phase == 'test'


        #print("flair_path: ", self.flair_dir)
        #print("ls_path: ", self.lsmsk_dir)
        #print("ls path exists: ", os.path.exists(self.lsmsk_dir))
        #print("flair path existsi? : ", os.path.exists(self.flair_dir))
        #print("no files; ", len(self.flairnames))

    def __getitem__(self, index):
        """
        Lädt ein Bildpaar (source & target) und konvertiert sie in Tensoren.
        """

        basename = self.flairnames[index]
        # load the flair img and it's corresponding lesion mask
        flair_path = os.path.join(self.flair_dir, basename + ".png")
        ls_msk_path = os.path.join(self.lsmsk_dir, basename.replace("flair", "mask") + ".png")

        
        flair_img = Image.open(flair_path).convert('L')  # 'L' für Graustufen
        flair_img = np.array(flair_img)


        lsmsk_img = Image.open(ls_msk_path).convert('L')
        lsmsk_img = np.array(lsmsk_img)

        masked_flair = flair_img.copy()
        masked_flair[lsmsk_img != 0] = 255

        # Source Tensor: orignal mri with lesionregions labeled as 255 [1, H, W]
        source_tensor = torch.tensor(masked_flair, dtype=torch.float32).unsqueeze(0)

        # Target Tensor: only original mri [1, H, W]
        target_tensor = torch.tensor(flair_img, dtype=torch.float32).unsqueeze(0)

        # Lesion-Mask: [1, H, W]
        mask_tensor = torch.tensor(lsmsk_img, dtype=torch.float32).unsqueeze(0)

        #print("Source Tensor shape: ", source_tensor.shape)
        #print("Target Tensor shape: ", target_tensor.shape)
        #print("Mask Tensor shape: ", mask_tensor.shape)


        if self.is_test: 
             return {"A": source_tensor, "A_paths": flair_path}
        else: 
             return {"A": source_tensor, "B": target_tensor, "mask": mask_tensor, 
                        "A_paths": flair_path, "B_paths": flair_path}


    def __len__(self):
        return len(self.flairnames)
