""" 
This class is meant to load in the data to train and test GAN2. Gan2 learns to create 
realistic looking synthetic lesions in a grayscale 3D MRI, from a lesion mask. 
source input: Brain MRI with lesion mask on top (areas labeld with one's where the lesions are 
supposed to be.) 
target_output: The same Brain MRI but the lesion mask areas replaced by synthetic lesions. 

The data class expects a dataroot which has two dirs: 
    * ./registered_flairs
    * ./registered_lesionmasks

names have to match and it's the same no matter if it is test or training images.

The given Images have to be already 2D slices and only slices, which contain lesion voxels are 
used. The Lesion masks are already dilated and lesion_mask and mri have the same basename. 
Images are expected to be 2d numpy arrays. saved as '.npy'. See preprocessing for more details.

 -> load them in, arrange source and target. 
        (source: channel 1: grayscale mri, but deleted where lesion mask is + lesion_mask in channel 2)
        (target: the original mri with real lesions in training / synthetic ones in testing.)

* Lesion mask is a binary mask with values [0,1].
* Flair images are grayscale images with values in [0,255], 
    they are normalized to [-1, 1] for the model.
    
"""



import os
import torch
import nibabel as nib
from data.base_dataset import BaseDataset
import numpy as np
from PIL import Image

class Gan2Dataset(BaseDataset):

    K = 5  # Number of folds for cross-validation

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--L1_mask_only', action='store_true', help='Computes L1 only for lesion areas.')
        #parser.add_argument('--k_fold', type=int, default=-1, help='K-fold cross-validation. -1: no k-fold, 0-4: use fold_x).')
        return parser


    def __init__(self, opt):
        """Initialisiert das Dataset"""
        super().__init__(opt)

        self.samples = []
        self.original_shape = None
        self.is_test = opt.phase == 'test'
        self.fold_idx = opt.k_fold if opt.k_fold >= 0 else -1

        # TODO implemment k=-1 for no k-fold

        if self.is_test: 
            folds = [self.fold_idx]        
        else: 
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
                            "basename": fname.replace(".npy", "").replace("flair", ""),
                            "flair_path": flair_path,
                            "lsmsk_path": lsmsk_path
                            })
    

    def __getitem__(self, index):
        """
        Loads an image pair (source & target) and converts them into tensors.
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

        # Source Tensor: orignal mri with lesionregions labeled as 0 [3, H, W]
        # 3 channels -> masked flair, lsmsk_img, empty
        
        source_array = np.stack([masked_flair_scaled, lsmsk_img, empty], axis=0)
        source_tensor = torch.tensor(source_array, dtype=torch.float32)

        if self.is_test: 
             return {"A": source_tensor, "A_paths": sample["flair_path"]}
        else: 
            # Target Tensor: only original mri [3, H, W]
            # 3 channels -> unmasked flair, lsmsk_img, empty
            target_array = np.stack([flair_img_scaled, lsmsk_img, empty], axis=0)
            target_tensor = torch.tensor(target_array, dtype=torch.float32)

            # get "BrainMask" / non-background area -> area where flair_img != 0. 
            brainmask_tensor = torch.tensor(flair_img != 0, dtype=torch.float32)

            return {"A": source_tensor, "B": target_tensor, "mask": brainmask_tensor, 
                        "A_paths": sample["flair_path"], "B_paths": sample["flair_path"]}


    def __len__(self):
        return len(self.samples)
