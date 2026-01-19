"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
The results will be saved at ./results/.
Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

Test a pix2pix model:
    python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix

    Originally from pix2pixnifti repository.
    Modified: Franziska Röpke (modified 2026-01-15)
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.evaluation import ExcelEvaluate
#from util.visualizer import save_images, save_nifti_images, save_web_nifti
from util import html_handler
from util.util import postprocess_images, print_timestamped
import time
import imageio
from util import util
import numpy as np


# Edited: Franziska Röpke (2026-01-15)
def save_png_images(visuals, query_name, opt, web_dir):
    """
   saves the 2D images as png for visual inspection.

    Parameter:
        np_dict: dict mit Bilddaten (z. B. {'real_A': np.array, ...})
        query_name: string – Basisname für die Dateien
        opt: Optionen, inkl. mapping_source, mapping_target, smoothing, postprocess
        web_dir: Pfad zum Zielverzeichnis
    """

    new_names = {
        'real_A': f"{query_name}_real_A",
        'real_B': f"{query_name}_real_B",
        'fake_B': f"{query_name}_fake_B",
        'truth': f"{query_name}_truth"
        }

    os.makedirs(web_dir, exist_ok=True)

    import matplotlib.pyplot as plt

    for label, image in visuals.items():
        img_path = os.path.join(web_dir, f"{query_name}_{label}.png")


        # Trainingslogik: nur Kanal 0
        if image.shape[1] == 3:
            image_numpy = image[0, 0].detach().cpu().numpy()
        else:
            image_numpy = image[0].detach().cpu().numpy()

        image_numpy = image_numpy.astype(np.uint8)

        # Speichern mit imageio
        imageio.imwrite(img_path, image_numpy)


def save_npy_array(visuals, query_name, opt, web_dir):
    """
    Saves the 2D images as numpy arrays for further processing.

    Parameters:
        visuals: dict with image data (e.g. {'real_A': np.array, ...})
        query_name: string - base name for the files
        opt: options object
        web_dir: path to the target directory
    """
    os.makedirs(web_dir, exist_ok=True)

    for label, image in visuals.items():
        np_path = os.path.join(web_dir, f"{query_name}_{label}.npy")
        
        # Extract only channel 0
        if image.shape[1] == 3:
            image_numpy = image[0, 0].detach().cpu().numpy()
        else:
            image_numpy = image[0].detach().cpu().numpy()

        np.save(np_path, image_numpy)
       
if __name__ == '__main__':
    opt = TestOptions().parse()
    
    # Hard-code some parameters for test
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    # Determine whether we are using nifti images
    nifti = True if opt.dataset_mode == "nifti" else False
    
    # Define web directory
    if nifti:
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}_{}'.format(opt.phase, opt.epoch, opt.postprocess))
        excel_filename = os.path.join(web_dir, (opt.phase + "_" + str(opt.name) + ".csv"))
        exc_eval = ExcelEvaluate(excel_filename, opt.excel)
    else:
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
        exc_eval = None
        
    if opt.load_iter > 0:
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    
    print('Creating web directory:', web_dir)
    webpage = html_handler.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    if opt.eval:
        model.eval()
    
    init_time = time.time()
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        
        if i % 5 == 0:
            print('Processing (%04d)-th image... %s' % (i, img_path))
        
        query_name = os.path.basename(img_path[0]).split(".")[0]
        save_png_images(visuals, query_name, opt, web_dir)
        save_npy_array(visuals, query_name, opt, web_dir)
    
    webpage.save()
    end_time = round(time.time() - init_time, 3)
    print_timestamped("Testing process completed in " + str(end_time) + "s.")
    
    if nifti:
        exc_eval.close()
