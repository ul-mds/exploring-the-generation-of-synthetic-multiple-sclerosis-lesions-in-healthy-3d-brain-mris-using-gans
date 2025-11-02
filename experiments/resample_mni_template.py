# Code snippet used to resample the MNI templates to 256x256x256 resolution. 

import ants
from pathlib import Path

def resample_template(output_name, brain_mask, template_path=None):
    if not template_path: 
        template_path = Path("/home/flydia/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz")
    print("Template exists:", template_path.exists())
    template = ants.image_read(str(template_path))
    brain_mask = ants.image_read(str(brain_mask))
    template_brain = ants.mask_image(template, brain_mask)
    original_spacing = template.spacing      # z.B. (1.0, 1.0, 1.5)
    original_size = template.shape           # z.B. (182, 218, 182)

    new_size = (256, 256, 256)
    # Berechne das neue Spacing so, dass das Field-of-View gleich bleibt
    new_spacing = tuple((o_s * o_sz) / n_sz for o_s, o_sz, n_sz in zip(original_spacing, original_size, new_size))

    print(f"Original size: {original_size}, spacing: {original_spacing}")
    print(f"New size: {new_size}, new spacing: {new_spacing}")

    resampled = ants.resample_image(template_brain, new_spacing, use_voxels=False)
    ants.image_write(resampled, str(output_name))
    print(f"Resampled image saved to {output_name}")

t1_template = Path("/home/flydia/Documents/mri_preprocessing_and_segmentation/data/orig_templates/mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin.nii")
t2_template = Path("/home/flydia/Documents/mri_preprocessing_and_segmentation/data/orig_templates/mni_icbm152_lin_nifti/icbm_avg_152_t2_tal_lin.nii")

t1_out = Path("/home/flydia/Documents/mri_preprocessing_and_segmentation/data/standard/icbm_avg_152_t1_tal_lin_resampled.nii.gz")
t2_out = Path("/home/flydia/Documents/mri_preprocessing_and_segmentation/data/standard/icbm_avg_152_t2_tal_lin_resampled.nii.gz")

brain_mask = Path("/home/flydia/Documents/mri_preprocessing_and_segmentation/data/standard/mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin_mask.nii")
 
resample_template(output_name=t1_out, brain_mask=brain_mask, template_path=t1_template)
resample_template(output_name=t2_out, brain_mask=brain_mask, template_path=t2_template)