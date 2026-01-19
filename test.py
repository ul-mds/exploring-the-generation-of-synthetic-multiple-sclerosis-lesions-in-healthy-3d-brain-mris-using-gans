"""
General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

The script creates model and dataset given the options, then runs inference and saves 
results to an HTML file in the results directory.

Example usage:
    python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix

    Originally from pix2pixnifti repository.
    Modified: Franziska Röpke (modified 2026-01-15)
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html_handler
from util.util import print_timestamped
import time


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
    
    # Define website directory for results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    if opt.load_iter > 0:
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html_handler.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    if opt.eval:
        model.eval()
    
    init_time = time.time()
    for i, data in enumerate(dataset):
        dataset_mode = model.opt.dataset_mode
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals(dataset_mode)
        img_path = model.get_image_paths()
        
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_images(
            opt=opt,
            webpage=webpage,
            visuals=visuals,
            dataset_mode=dataset_mode,
            image_path=img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize
        )

    webpage.save()
    end_time = round(time.time() - init_time, 3)
    print_timestamped("testing process completed in " + str(end_time) + "s.")