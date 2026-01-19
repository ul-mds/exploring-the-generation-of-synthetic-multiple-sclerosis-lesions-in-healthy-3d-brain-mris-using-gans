"""
Code snippet used to resample the MNI templates to 256x256x256 resolution.

Author/Mod: Franziska Röpke (added/modified 2026-01-15)
"""


import ants
from pathlib import Path

def resample_template(output_name, brain_mask, template_path=None):
    """
    Resamples the given MNI template to 256x256x256 resolution, using the brain mask to extract only the brain region.
    Saves the resampled image to output_name.
    
    :param output_name: Path where the resampled image will be saved.
    :param brain_mask: Path to the brain mask image used to extract the brain region.
    :param template_path: Path to the original MNI template image to be resampled.
    """

    if not template_path: 
        template_path = Path("./data/standard/MNI152_T1_1mm_brain.nii.gz")
    print("Template exists:", template_path.exists())
    template = ants.image_read(str(template_path))
    brain_mask = ants.image_read(str(brain_mask))
    template_brain = ants.mask_image(template, brain_mask)
    original_spacing = template.spacing   
    original_size = template.shape          
    new_size = (256, 256, 256)

    # Calculate new spacing
    new_spacing = tuple((o_s * o_sz) / n_sz for o_s, o_sz, n_sz in zip(original_spacing, original_size, new_size))

    print(f"Original size: {original_size}, spacing: {original_spacing}")
    print(f"New size: {new_size}, new spacing: {new_spacing}")

    resampled = ants.resample_image(template_brain, new_spacing, use_voxels=False)
    ants.image_write(resampled, str(output_name))
    print(f"Resampled image saved to {output_name}")


def main():
    """
    Main function to resample MNI templates (T1 and T2) to 256x256x256 resolution.
    """

    t1_template = Path("./data/orig_templates/mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin.nii")
    t2_template = Path("./data/orig_templates/mni_icbm152_lin_nifti/icbm_avg_152_t2_tal_lin.nii")

    t1_out = Path("./data/standard/icbm_avg_152_t1_tal_lin_resampled.nii.gz")
    t2_out = Path("./data/standard/icbm_avg_152_t2_tal_lin_resampled.nii.gz")

    brain_mask = Path("./data/standard/mni_icbm152_lin_nifti/icbm_avg_152_t1_tal_lin_mask.nii")
    
    resample_template(output_name=t1_out, brain_mask=brain_mask, template_path=t1_template)
    resample_template(output_name=t2_out, brain_mask=brain_mask, template_path=t2_template)


if __name__ == "__main__":
    main()