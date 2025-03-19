import os
import torch
import nibabel as nib
from data.base_dataset import BaseDataset

class MyNiftiDataset(BaseDataset):
    """Dataset für Pix2Pix mit NifTi-Bildern"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Fügt spezifische Optionen für das Dataset hinzu."""
        parser.add_argument('--nifti_normalize', type=bool, default=True, help='Normalisiert die NifTi-Dateien auf [0,1]')
        parser.add_argument('--L1_lesions_only', type=bool, default=False, help='Computes L1 only for lesion areas.')
        return parser

    def __init__(self, opt):
        """Initialisiert das Dataset"""
        super().__init__(opt)

        # Speicherorte der Bilder
        self.source_dir = os.path.join(opt.dataroot, "train_gan1", "source")
        self.target_dir = os.path.join(opt.dataroot, "train_gan1", "target")

        self.target_dir = os.path.join(opt.dataroot, "train_gan1", "target")
        source_files = [f for f in os.listdir(self.source_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
        self.basenames = sorted([f.replace("_source_gan1.nii", "").replace("_source_gan1.nii.gz", "") for f in source_files])
        
        # Lesion mask only for L1 
        self.L1_lesions_only = opt.L1_lesions_only

        # test imgs
        self.test_dir = os.path.join(opt.dataroot, "gan1_test_imgs")
        test_files = [f for f in os.listdir(self.test_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
        self.test_basenames = sorted([f.replace("_test_gan1.nii", "")for f in test_files])
        self.is_test = opt.phase == 'test'


    def __getitem__(self, index):
        """Lädt ein Bildpaar (source & target) und konvertiert sie in Tensoren"""

        if self.is_test: 
            test_filename = self.test_basenames[index] + "_test_gan1.nii"
            test_path = os.path.join(self.test_dir, test_filename)
            test_img = nib.load(test_path).get_fdata()

            # crop 256^3 to 128³: 
            test_img = self.crop_image(test_img)

            test_tensor = torch.tensor(test_img, dtype=torch.float32).unsqueeze(0)

            # return {"A": test_tensor, "B": None, "A_paths": test_path, "B_paths": None}
            return {"A": test_tensor, "A_paths": test_path}
        
        else: 
            #Training mode-> load image pairs
            basename = self.basenames[index]
            # Lade Dateipfade
            source_path = os.path.join(self.source_dir, basename + "_source_gan1.nii")
            target_path = os.path.join(self.target_dir, basename + "_target_gan1.nii")


            # Lade die NifTi-Bilder mit nibabel
            source_img = nib.load(source_path).get_fdata()
            target_img = nib.load(target_path).get_fdata()

            # crop images to 128³: 
            source_img = self.crop_image(source_img)
            target_img = self.crop_image(target_img)

            # Konvertiere zu PyTorch-Tensoren
            source_tensor = torch.tensor(source_img, dtype=torch.float32).unsqueeze(0)  # [C, H, W, D]
            target_tensor = torch.tensor(target_img, dtype=torch.float32).unsqueeze(0)
            
            # the mask for L1 == target_gan1, it's the same Input. 
            if self.L1_lesions_only: 
                mask_tensor = target_tensor.clone()
                return {"A": source_tensor, "B": target_tensor, "mask": mask_tensor, 
                        "A_paths": source_path, "B_paths": target_path}

            else:     
                return {"A": source_tensor, "B": target_tensor, "A_paths": source_path, "B_paths": target_path}

    def __len__(self):
        """Gibt die Anzahl der Bilder zurück"""
        if self.is_test:
            return len(self.test_basenames) 
        else: 
            return len(self.basenames)
        
    def crop_image(self, img):
        """
        Cuts out the center of the image (trying to avoid all the background.)
        we normaly expect to have 256 size and we crop to 128, taking the middle 
        of the cube. Expects x, y, and z coordinates to have same size ! 
        """
        # Berechne die Koordinaten für den mittleren Ausschnitt
        start = (img.shape[1] - 128) // 2  # Startindex in der Höhe
        end = start + 128  # Endindex in der Höhe
        
        # Ausschneiden des mittleren Bereichs (für alle Dimensionen)
        cropped_img = img[start:end, start:end, start:end]
        return cropped_img
