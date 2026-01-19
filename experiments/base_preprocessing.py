
"""
Preprocessing class for medical images, including registration to MNI space, skull stripping, and bias correction and other 
preprocessing steps. 

Author/Mod: Franziska Röpke (added/modified 2026-01-15)

Used for GAN1 and GAN2 preprocessing pipelines. Prduces interim results that can be used for 
further processing steps and source/target images ready to be fed into GAN1 / GAN2. 
(see data/gan1_dataset.py and data/gan2_dataset.py for details on model I/O formats).
"""

import ants
import shutil
from pathlib import Path
import numpy as np
import nibabel as nib
from nipype.interfaces.freesurfer import WatershedSkullStrip
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_dilation
from PIL import Image
import logging
import os
import time
import pandas as pd
import antspynet


def setup_logger(results_dir):
    log_path = Path(results_dir, "logging.txt")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True) 
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )

class ImgPreprocessing(): 
    T1_TEMPLATE = Path(r"./data/standard/icbm_avg_152_t1_tal_lin_resampled.nii.gz")
    T2_TEMPLATE = Path(r"./data/standard/icbm_avg_152_t2_tal_lin_resampled.nii.gz")

    def __init__(self, name:str, 
                 result_dir,
                  mni_transform = None, 
                  flair_path = None, 
                  t1_path = None, 
                  ls_path = None,
                  aseg_path = None,
                  brain_path = None): 
        
        self.name = name
        self.images = {'flair': flair_path, 
                       't1': t1_path, 
                       'ls_mask': ls_path,
                       'aseg': aseg_path,
                       'brain_mask': brain_path
                       }
        
        self.result_dir = result_dir
        setup_logger(self.result_dir)
        self.mni_transform = mni_transform

        interim_results = result_dir / "interim_results" 
        interim_results.mkdir(parents=True, exist_ok=True)

        for path in self.images.values(): 
            if path is not None and not Path(path).exists(): 
                raise ValueError(f"Path {path} doesn't exist!")
    

    def apply_mni_transform(self, image_key: str) -> None:
        """ 
        Function registers the image to the template and saves the resulting registered 
        image. A mni_transform has to be calculated beforehand using calc_transform_mni152().
        Parameters:
            image_key (str): Key of the image to be transformed ('flair', 't1', 'ls_mask', etc.) 
        """
        if self.mni_transform is None:
            raise ValueError(f"mni transformation for {self.name} not found, did you forget to compute it ?")
        
        if self.images[image_key] is None: 
            raise ValueError(f"No path to image {image_key} for {self.name} given.")

        # use linear interpolation for all images except the binary masks, which should be registered with nearest neighbor.
        if image_key == 'ls_mask' or image_key == 'wm_mask': 
            interpolator = 'nearestNeighbor'
        else:
            interpolator = 'linear'
        
        output_file = (self.result_dir / "interim_results" / f"{self.name}_{image_key}_mni_registered.nii.gz").resolve()
        img = ants.image_read(str(self.images[image_key])) 

        # we normally register to T1, only if no T1 is given, we register flair to T2 template.
        if self.images['t1'] is None:
            mni_template = ants.image_read(str(ImgPreprocessing.T2_TEMPLATE))
        else:
            mni_template = ants.image_read(str(ImgPreprocessing.T1_TEMPLATE))

        transformed_image = ants.apply_transforms(fixed=mni_template, 
                                                  moving= img, 
                                                  transformlist=[str(self.mni_transform)],
                                                  interpolator= interpolator)

        ants.image_write(transformed_image, str(output_file))

        self.images[image_key] = output_file

        logging.info(f"Completed MNI registration for img {self.name} {image_key}.")


    def calc_transform_mni152(self, image_key: str) -> None:
        """
        Function calculates the transform form the orig image to the mni template. 
        stores it.
        Parameters:
            image_key (str): Key of the image to be transformed ('flair', 't1', 'ls_mask', etc.)
        """

        if self.images[image_key] is None: 
            raise ValueError(f"No path to image {image_key} for {self.name} given.")
        
        # normaly we register to T1, but if no T1 is given, we register flair to T2 template.
        if image_key == 't1': 
            mni_template = ants.image_read(str(ImgPreprocessing.T1_TEMPLATE))
        elif image_key == 'flair':        
            mni_template = ants.image_read(str(ImgPreprocessing.T2_TEMPLATE))

        output_file = (self.result_dir / "interim_results" / f"{self.name}_{image_key}_mni152_transform.mat").resolve()

        # load imgs: 
        img = ants.image_read(str(self.images[image_key]))

        registration = ants.registration(fixed=mni_template, moving=img, type_of_transform="Affine") 

        # save transform in object and file
        self.mni_transform = output_file
        shutil.copy(registration['fwdtransforms'][0], output_file)

        logging.info(f"Calculated and saved mnni transformation for img {self.name} {self.name} succesfully.")


    def resample_to_256(self, image_key: str) -> None: 
        """
        Resamples a 3D image of arbitraty size to a 256^3 image. Voxels are not expected to be quadratic.
        Parameters:
            image_key (str): Key of the image to be transformed ('flair', 't1', 'ls_mask', etc.)
        """
        if self.images[image_key] is None: 
            raise ValueError(f"No path to image {image_key} for {self.name} given.")
        
        img = ants.image_read(str(self.images[image_key]))
        original_spacing = img.spacing  
        original_size = img.shape   

        new_size = (256, 256, 256)
        new_spacing = tuple((o_s * o_sz) / n_sz for o_s, o_sz, n_sz in zip(original_spacing, original_size, new_size))

        resampled_img = ants.resample_image(img, new_spacing, use_voxels=False)

        output_file = (self.result_dir / "interim_results" / f"{self.name}_{image_key}_mni_registered.nii.gz").resolve()
        ants.image_write(resampled_img, output_file)

        self.images[image_key] = output_file

        logging.info(f"Completed resampling to size 256^3 for img {self.name} {image_key}.")


    def coregister_to_t1(self, image_key: str) -> None:

        """ 
        Coregisters a given image to the fixed_img (t1).
        Saves the resulting image(s) in the results_dir. 
        Parameters:
            image_key (str): Key of the image to be transformed ('flair', 't1', 'ls_mask', etc.)
        """ 
        
        fixed_img = ants.image_read(str(self.images['t1']))
        moving_img = ants.image_read(str(self.images[image_key]))

        registration_output = ants.registration(fixed=fixed_img, 
                                    moving=moving_img, 
                                    type_of_transform="Affine")
        registered_img = registration_output['warpedmovout']

        # save result
        img_outpath = self.result_dir / "interim_results" / f"{self.name}_{image_key}_coregistered.nii.gz"
        ants.image_write(registered_img, str(img_outpath))
        self.images[image_key] = img_outpath

        logging.info(f"{image_key} registered to t1 successfully for {self.name} {image_key}")


    def bias_correction(self, image_key: str) -> None:
        """
        Performs bias field correction using ANTs N4BiasFieldCorrection.
        Parameters:
            image_key (str): Key of the image to be bias corrected ('flair', 't1', etc.)
        """
        input_path = self.images[image_key]
        img = ants.image_read(str(input_path))
        corrected = ants.n4_bias_field_correction(img)

        output_file = (self.result_dir / "interim_results" / f"{self.name}_{image_key}_bias_corrected.nii.gz").resolve()
        ants.image_write(corrected, str(output_file))

        self.images[image_key] = output_file
        logging.info(f"Completed bias correction for img {self.name} {image_key}.")
    

    def extract_brain(self, image_key: str, max_retries: int = 5) -> None:
        """
        Extracts the brain and cuts of the scull from the image. Uses FreeSurfer's WatershedSkullStrip.
        Parameters:
            image_key (str): Key of the image to be brain extracted ('flair', 't1', etc.)
        """

        output_file = (self.result_dir / "interim_results" / f"{self.name}_{image_key}_brain_extracted.nii.gz").resolve()

        for attempt in range(max_retries):
            try:
                skullstrip = WatershedSkullStrip()
                skullstrip.inputs.in_file = self.images[image_key]
                skullstrip.inputs.t1 = False
                skullstrip.inputs.out_file = output_file
                skullstrip.run()
                self.images[image_key] = output_file
                logging.info(f"Completed brain extraction for img {self.name} {image_key}.")
                return
            except Exception as e:
                logging.warning(f"Brain extraction failed for {self.name} {image_key} (attempt {attempt}): {e}")
                time.sleep(2)  
        logging.error(f"Brain extraction failed for {self.name} {image_key} after {max_retries} attempts.")
        raise RuntimeError(f"Brain extraction failed for {self.name} {image_key} after {max_retries} attempts.")


    def extract_brain_ants(self, image_key: str, modality: str = 'flair') -> None:
        """ 
        Extracts brain using ANTs instead of freesurfer. 
        modality can be 't1', 't2' or 'flair'.
        Parameters: 
            image_key (str): Key of the image to be brain extracted ('flair', 't1', etc.)
            modality (str): Modality of the image ('t1', 't2', 'flair')
        """
        output_file = (self.result_dir / "interim_results" / f"{self.name}_{image_key}_brain_extracted.nii.gz").resolve()
        img = ants.image_read(str(self.images[image_key]))
        brain_mask = antspynet.utilities.brain_extraction(img, modality=modality, verbose=True)
        brain = img * brain_mask
        ants.image_write(brain, str(output_file))   
        self.images[image_key] = output_file


    def apply_brainmask(self, image_key: str) -> None:
        """ 
        Applies a brain mask to the image, effectively removing non-brain areas.
        Parameters:
            image_key (str): Key of the image to be brain masked ('flair', 't1', etc.)  
        """
        if self.images['brain_mask'] is None: 
            raise ValueError(f"No brain mask set for {self.name}, can't apply brain mask to {image_key}.")
        
        
        img = nib.load(str(self.images.get(image_key)))
        brain_mask = nib.load(str(self.images['brain_mask']))
        img_brain_data = img.get_fdata() * brain_mask.get_fdata()
        img_brain = nib.Nifti1Image(img_brain_data, affine=img.affine, header=img.header)

        output_file = (self.result_dir / "interim_results" / f"{self.name}_{image_key}_brain_extracted.nii.gz").resolve()
        nib.save(img_brain, str(output_file))
        self.images[image_key] = output_file


    def dilatate_lesionmasks(self) -> None: 
        """ 
        Performs a dilatation on the image it gets, mainly meant to dilatate
        the lesion masks, which is necessary in preprocessing for gan2. 
        """
        # load img
        ls_mask_path = self.images.get('ls_mask')
        orig_img = nib.load(str(ls_mask_path))
        orig_data = orig_img.get_fdata()

        # perform closing
        struct_element = generate_binary_structure(orig_data.ndim, connectivity=3)

        dilated_data = binary_dilation(orig_data, structure=struct_element).astype(orig_data.dtype)
        dilated_img = nib.Nifti1Image(dilated_data, affine=orig_img.affine, header=orig_img.header)

        # save results
        out_dir = Path(self.result_dir) / "interim_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.name}_ls_mask_dilated.nii.gz"
        nib.save(dilated_img, out_path)

        self.images['ls_mask'] = out_path
        logging.info(f"Dilated lesion mask saved to: {out_path}")
    

    def create_gan2_source_target(self, fold: int, flair_key: str = 'flair', mask_key: str = 'ls_mask') -> None:
        """
        Slices the 3D FLAIR and lesion mask images into 2D numpy arrays for GAN2 training.
        Only slices containing at least one lesion voxel are saved. The FLAIR images are normalized to [0, 255]
        before slicing and original min and max values are saved for later denormalization.
        
        This function is codependent with the create_gan2_3d_output() function, which reconstructs the 3D images. 
        Changes made in this code should be applied to the reconstruction function as well.

        Parameters:
            fold (int): Fold index for k-fold cross-validation.
            flair_key (str): Key for the FLAIR image in self.images.
            mask_key (str): Key for the lesion mask image in self.images.
        """
        
        flair_path = self.images.get(flair_key)
        mask_path = self.images.get(mask_key)
        if flair_path is None or mask_path is None:
            raise ValueError("FLAIR or lesion mask path missing.")

        flair_img = nib.load(str(flair_path)).get_fdata()
        lesion_img = nib.load(str(mask_path)).get_fdata()

        if flair_img.shape != lesion_img.shape:
            raise ValueError("Dimension mismatch between FLAIR and lesion mask.")
        
        # save min and max for later denormalization
        csv_path = Path(self.result_dir) / "flair_minmax.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=["name", "min", "max"])

        # add new row
        new_row = {"name": self.name, "min": float(flair_img.min()), "max": float(flair_img.max())}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        # Normalize flair_img to [0, 255] using min-max normalization
        flair_img = (flair_img - flair_img.min()) / (np.ptp(flair_img) + 1e-8) * 255

        # Ensure lesion mask is binary
        assert np.all(np.isin(lesion_img, [0, 1])), "Lesion mask should be binary (0 or 1)."

        # create result_dirs if they don't exist yet.
        flair_dir = Path(self.result_dir) / f"{fold}" /  "flair_slices"
        mask_dir = Path(self.result_dir) / f"{fold}" / "mask_slices"
        flair_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        img_index = 0
        for i in range(flair_img.shape[0]): 
            flair_slice = flair_img[i, : , :]
            lesion_slice = lesion_img[i, : , :]

            # check if lesion voxels in this slice
            if np.any(lesion_slice): 
                
                # save imgs
                flair_path = flair_dir / f"{self.name}_flair_slice_{img_index:03d}.npy"
                lesion_path = mask_dir / f"{self.name}_lsmsk_slice_{img_index:03d}.npy"

                np.save(flair_path, flair_slice.astype(np.float32))  
                np.save(lesion_path, lesion_slice.astype(np.uint8)) 
                 # save as uint8 for binary mask
                img_index += 1 

        logging.info(f"Created {img_index} GAN2 source/target slice pairs for {self.name}.")


    def create_gan1_source(self) -> None: 
        """
        Creates and saves the GAN1 source image (white matter mask).
        Expects an aseg.mgz file to be in self.images['aseg'].
        """
        if self.images['wm_mask'] is None: 
            raise ValueError("No wm_mask file set, gan1_source can't be generated.")

        wm_img = nib.load(str(self.images['wm_mask']))
       
        # set output path and save source_img
        out_dir = Path(self.result_dir) / "gan1_source"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.name}_source_gan1.nii.gz"
        nib.save(wm_img, str(out_path))

        logging.info(f"GAN1 source saved to: {out_path}")
    

    def extract_wm_mask(self):
        """ 
        Extracts the white matter mask from the aseg image and saves it as a NIfTI file. Mostly used for
        gan1 input generation.
        """
        if self.images['aseg'] is None: 
            raise ValueError("No aseg file set, gan1_source can't be generated.")

        aseg_img = nib.load(str(self.images['aseg']))
        aseg_data = aseg_img.get_fdata()

        
        wm_mask_data = np.where(np.isin(aseg_data, [2, 41, 77]), 1, 0).astype(np.uint8)

        # create nii object
        wm_mask = nib.Nifti1Image(wm_mask_data, 
                                          affine=aseg_img.affine, 
                                          header=aseg_img.header)
        
        out_path = Path(self.result_dir) / "interim_results" / f"{self.name}_wm_mask.nii.gz"
        nib.save(wm_mask, str(out_path))
        self.images['wm_mask'] = out_path


    def create_gan1_target(self) -> None: 
        """ 
        saves the lesion mask as gan1_target image. Lesion mask should be registered before.
        """

        # load lesion_mask to save it as target_img.
        ls_mask_path = self.images['ls_mask']
        
        # set output path and save target_img
        target_dir = Path(self.result_dir) / "gan1_target"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{self.name}_target_gan1.nii.gz"

        # Load and save as NIfTI
        img = nib.load(str(ls_mask_path))
        data = img.get_fdata()

        # check if mask is binary
        data = data.astype(np.uint8)
        img = nib.Nifti1Image(data, affine=img.affine, header=img.header)
        
        if not np.array_equal(np.unique(data), [0, 1]):
            raise ValueError(f"Lesion mask for {self.name} is not binary. Please check the input data.")
        
        nib.save(img, str(target_path))

        logging.info(f"GAN1 target saved to: {target_path}")
