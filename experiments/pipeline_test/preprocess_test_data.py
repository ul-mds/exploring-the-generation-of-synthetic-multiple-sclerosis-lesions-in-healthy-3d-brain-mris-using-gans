""" 
Preprocess test data for GAN1 model and pipeline inference.

Author/Mod: Franziska Röpke (added 2026-01-15)

Script processes the test images, which are MRIs of healthy patients, without lesions.
The aseg.mgz file is expected to exist as a result of the recon-all preprocessing step.
Results are the gan1_source_imgs. The test data is split randomly into 2 subsets, one is
used for hyperparameter tuning of gan1 and the other for final testing of the pipeline.
"""

import sys
import os
from pathlib import Path
import re
import shutil
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base_preprocessing import ImgPreprocessing


def main():
    """
    Main preprocessing pipeline for test data from multiple datasets.
    Processes CERMP, PPMI, OASIS, and ICBM datasets and splits them for GAN1 testing.
    """
    process_cermp = True
    process_ppmi = True
    process_oasis = True
    process_icbm = True

    results_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/test_pipeline_data/test_data_preprocessing240925ants_brain_extraction")
    freesurfer_dir = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/test_pipeline_data/freesurfer_results06082025")
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)

    all_basenames = []  # collect all processed images, for the 50 / 50 data split
    #-------------------------------------------------------------
    #                CERMP                           
    #-------------------------------------------------------------

    if process_cermp: 
        t1_cermp = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/cermp/t1")

        flair_cermp = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/cermp/flair")
        freesurfer_cermp = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/test_pipeline_data/cermp/freesurfer_results_cermp")

        #images to process
        cermp_images = [
                    'sub-0002', 'sub-0003', 'sub-0004', 'sub-0005', 'sub-0007', 'sub-0008',  'sub-0009', 'sub-0010', 'sub-0011',
                    'sub-0012', 'sub-0013', 'sub-0015', 'sub-0018', 'sub-0019', 'sub-0020', 'sub-0021', 'sub-0022', 'sub-0023', 
                    'sub-0024', 'sub-0028', 'sub-0029', 'sub-0030', 'sub-0031', 'sub-0032', 'sub-0034', 'sub-0035', 'sub-0037'
                    ]
        
        # naming: sub-0002_T1w.nii.gz
        for patient in cermp_images:
            patient_id = patient.split("-")[1]
            all_basenames.append(f"cermp_{patient_id}")
            t1_path = t1_cermp / f"{patient}_T1w.nii.gz"
            flair_path = flair_cermp / f"{patient}_FLAIR.nii.gz"
            aseg_path = freesurfer_cermp / f"cermp_patient{patient_id}" / "mri" / "aseg.mgz"
            
            cermp_patient = ImgPreprocessing(name=f"cermp_{patient_id}",
                                            result_dir=results_dir,
                                            mni_transform=None, 
                                            flair_path=flair_path, 
                                            t1_path=t1_path,
                                            ls_path=None,
                                            aseg_path=aseg_path
                                            )
            # scull strip and bias correct the imgs.
            cermp_patient.bias_correction(image_key='t1') 
            cermp_patient.bias_correction(image_key='flair')

            cermp_patient.extract_brain_ants(image_key='t1') 
            cermp_patient.extract_brain_ants(image_key='flair')

            cermp_patient.calc_transform_mni152('t1')
            
            # register flair to mni (used later to create gan2 input from synth_ls_mask and registered flair)
            cermp_patient.apply_mni_transform(image_key='flair')

            # extract wm from aseg and register to mni
            cermp_patient.extract_wm_mask()
            cermp_patient.apply_mni_transform(image_key='wm_mask')

            cermp_patient.create_gan1_source()

    ##############################################################################
    ##                                 PPMI                                  ##    
    ###############################################################################
    if process_ppmi:
        # 1: other scanners, 2: siemens
        ppmi_images1 = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/test_pipeline_data/ppmi/ppmi_other_scanners") 

        patient_ids1 = set()
        for file in ppmi_images1.iterdir():
            patient_id = file.stem.split('_')[0]
            patient_ids1.add(patient_id)

        for patient_id in patient_ids1:
            patient_name = f"ppmi_patient{patient_id}"
            all_basenames.append(patient_name)

            flair_path = None
            t1_path = None
            aseg_path = freesurfer_dir / "freesurfer_results_ppmi" / "other_scanners" / f"ppmi_patient{patient_id}" / "mri" / "aseg.mgz"
            for file in ppmi_images1.iterdir():
                if file.suffix == '.nii' and 'T1' in file.name and patient_id in file.name:
                    t1_path = file
                elif file.suffix == '.nii' and 'FLAIR' in file.name and patient_id in file.name:
                    flair_path = file
                if t1_path and flair_path:
                    break

            ppmi_patient1 = ImgPreprocessing(name=patient_name,
                                            result_dir=results_dir,
                                            mni_transform=None,
                                            flair_path=flair_path, 
                                            t1_path=t1_path,
                                            aseg_path=aseg_path
                                        )
            # scull strip and bias correct 
            ppmi_patient1.bias_correction(image_key='t1')
            ppmi_patient1.bias_correction(image_key='flair')

            ppmi_patient1.extract_brain_ants(image_key='t1')
            ppmi_patient1.extract_brain_ants(image_key='flair')

            # register T1 and save the transform.
            ppmi_patient1.calc_transform_mni152('t1')
            
            # register flair to mni (used later to create gan2 input from synth_ls_mask and registered flair)
            ppmi_patient1.apply_mni_transform(image_key='flair')
            
            # extract wm from aseg and register to mni
            ppmi_patient1.extract_wm_mask()
            ppmi_patient1.apply_mni_transform(image_key='wm_mask')

            ppmi_patient1.create_gan1_source()


        # 2: siemens
        ppmi_images2 = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/test_pipeline_data/ppmi/ppmi_siemens")
        patient_ids2 = set()
        for file in ppmi_images2.iterdir():
            patient_id = file.stem.split('_')[0]
            patient_ids2.add(patient_id)

        for patient_id in patient_ids2:
            patient_name = f"ppmi_patient{patient_id}"
            all_basenames.append(patient_name)
            flair_path = None
            t1_path = None
            aseg_path = freesurfer_dir / "freesurfer_results_ppmi" / "siemens" / f"ppmi_patient{patient_id}" / "mri" / "aseg.mgz"
            for file in ppmi_images2.iterdir():
                if file.suffix == '.nii' and 'T1' in file.name and patient_id in file.name:
                    t1_path = file
                elif file.suffix == '.nii' and 'FLAIR' in file.name and patient_id in file.name:
                    flair_path = file
                if t1_path and flair_path:
                    break

            ppmi_patient2 = ImgPreprocessing(name=patient_name,
                                            result_dir=results_dir,
                                            mni_transform=None,
                                            flair_path=flair_path, 
                                            t1_path=t1_path,
                                            aseg_path=aseg_path
                                        )
            # scull strip and bias correct 
            ppmi_patient2.bias_correction(image_key='t1')
            ppmi_patient2.bias_correction(image_key='flair')

            ppmi_patient2.extract_brain_ants(image_key='t1')
            ppmi_patient2.extract_brain_ants(image_key='flair')

            # register T1 and save the transform.
            ppmi_patient2.calc_transform_mni152('t1')
            
            # register flair to mni (used later to create gan2 input from synth_ls_mask and registered flair)
            ppmi_patient2.apply_mni_transform(image_key='flair')
            
            # extract wm from aseg and register to mni
            ppmi_patient2.extract_wm_mask()
            ppmi_patient2.apply_mni_transform(image_key='wm_mask')

            ppmi_patient2.create_gan1_source()

    ############################################################################
    #                                 OASIS                                 #    
    ############################################################################
    if process_oasis:
        #naming:
        # sub-OAS30411_sess-d5564_T1w.nii.gz
        oasis_images = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/oasis")
        oasis_freesurfer = freesurfer_dir /  "freesurfer_results_oasis"

        oasis_file_names = set()
        for file in oasis_images.iterdir():
            if file.suffixes == ['.nii', '.gz'] and 'T1w' in file.name:
                patient_file_name = re.sub(r'_(T1w|FLAIR).*$', '', file.stem)

                oasis_file_names.add(patient_file_name)

        for patient in oasis_file_names:
            patient_id = patient.split("-")[1].split("_")[0]
            patient_name = f"oasis_patient{patient_id}"
            all_basenames.append(patient_name)
            t1_path = oasis_images / f"{patient}_T1w.nii.gz"
            flair_path = oasis_images / f"{patient}_FLAIR.nii.gz"
            aseg_path = freesurfer_dir / "freesurfer_results_oasis" / f"oasis_patient{patient_id}" / "mri" / "aseg.mgz"

            oasis_patient = ImgPreprocessing(name=patient_name,
                                            result_dir=results_dir,
                                            mni_transform=None,
                                            flair_path=flair_path,
                                            t1_path=t1_path,
                                            aseg_path=aseg_path
                                        )
            # scull strip and bias correct
            oasis_patient.bias_correction(image_key='t1')
            oasis_patient.bias_correction(image_key='flair')

            oasis_patient.extract_brain_ants(image_key='t1')
            oasis_patient.extract_brain_ants(image_key='flair')

            # register T1 and save the transform.
            oasis_patient.calc_transform_mni152('t1')
            
            # register flair to mni (used later to create gan2 input from synth_ls_mask and registered flair)
            oasis_patient.apply_mni_transform(image_key='flair')
            
            # extract wm from aseg and register to mni
            oasis_patient.extract_wm_mask()
            oasis_patient.apply_mni_transform(image_key='wm_mask')      

            oasis_patient.create_gan1_source()

    ############################################################################
    #                                 ICBM                                  #    
    ############################################################################
    if process_icbm:
        icbm_images = Path("/mnt/c/Users/T14/Documents/Masterthesis/Data/original_data/icbm")
        icbm_freesurfer = freesurfer_dir / "freesurfer_results_icbm"

        for dir in icbm_images.iterdir():
            t1_path = None
            flair_path = None
            for file in dir.iterdir():
                if file.suffix == '.nii' and 'T1' in file.name:
                    t1_path = file
                elif file.suffix == '.nii' and 'FLAIR' in file.name:
                    flair_path = file
                if t1_path and flair_path:
                    break
            all_basenames.append(f"icbm_patient{dir.name}")
            aseg_path = icbm_freesurfer / f"icbm_patient{dir.name}" / "mri" / "aseg.mgz"
            icbm_patient = ImgPreprocessing(name=f"icbm_patient{dir.name}",
                                                result_dir=results_dir,
                                                mni_transform=None,
                                                flair_path=flair_path, 
                                                t1_path=t1_path,
                                                aseg_path=aseg_path
                                            )
            # scull strip and bias correct
            icbm_patient.bias_correction(image_key='t1')
            icbm_patient.bias_correction(image_key='flair')

            icbm_patient.extract_brain_ants(image_key='t1')
            icbm_patient.extract_brain_ants(image_key='flair')

            # register T1 and save the transform.
            icbm_patient.calc_transform_mni152('t1')
            
            # register flair to mni (used later to create gan2 input from synth_ls_mask and registered flair)
            icbm_patient.apply_mni_transform(image_key='flair')
            
            # extract wm from aseg and register to mni
            icbm_patient.extract_wm_mask()
            icbm_patient.apply_mni_transform(image_key='wm_mask')

            icbm_patient.create_gan1_source()   
        
    ################################################################
    #           split dataset in 50 - 50 split                     #
    #################################################################

    dataset1_gan1 = results_dir / "gan1_testdata" / "gan1_source"
    dataset1_regflairs = results_dir / "gan1_testdata" / "registered_flairs"
    dataset1_interim_results = results_dir / "gan1_testdata" / "interim_results"

    dataset1_gan1.mkdir(parents=True, exist_ok=True)
    dataset1_regflairs.mkdir(parents=True, exist_ok=True)
    dataset1_interim_results.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(all_basenames) # shuffle the basenames to get a random split

    split_index = len(all_basenames) // 2
    dataset1_basenames = all_basenames[:split_index]


    for basename in dataset1_basenames: 
        source = results_dir / "gan1_source" / f"{basename}_source_gan1.nii.gz"
        flair = results_dir / "interim_results" / f"{basename}_flair_mni_registered.nii.gz"
        shutil.copy(source, dataset1_gan1 / source.name)
        shutil.copy(flair, dataset1_regflairs / flair.name)
        os.remove(source)
        os.remove(flair)
        # and copy the interim results aswell
        for file in results_dir.glob(f"interim_results/{basename}*"):
            shutil.copy(file, dataset1_interim_results / file.name)
            os.remove(file)

    # move orig files (second half of dataset) into pipeline_test_data
    pipeline_test_data = results_dir / "pipeline_test_data"
    pipeline_test_data.mkdir(parents=True, exist_ok=True)

    dirs_to_keep = ["gan1_testdata", "pipeline_test_data"]
    for dir in results_dir.iterdir():
        if dir.is_dir() and dir.name not in dirs_to_keep: 
            shutil.move(str(dir), str(pipeline_test_data / dir.name))


if __name__ == "__main__":
    main()
