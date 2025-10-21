"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
The results will be saved at ./results/.
Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

Test a pix2pix model:
    python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.evaluation import ExcelEvaluate
from util.visualizer import save_images, save_nifti_images, save_web_nifti
from util import html_handler
from util.util import postprocess_images, print_timestamped
import time

# Fimos trying to debug the container:
#import debugpy

# Starte den Debugging-Server
#debugpy.listen(("0.0.0.0", 5678))
#print("🚀 Warten auf Debugger Attach auf Port 5678...")

# Optional: hier anhalten bis sich dein Debugger verbunden hat
#debugpy.wait_for_client()
#print("✅ Debugger verbunden!")

# end of Fimos stuff.

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling;
    # comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a website
    exc_eval = None
    
    web_dir = os.path.join(opt.results_dir, opt.name,
                        '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html_handler.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix.
    # You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    init_time = time.time()
    for i, data in enumerate(dataset):
        dataset_mode = model.opt.dataset_mode
        #if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #    break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals(dataset_mode)  # get image results
        img_path = model.get_image_paths()  # get image paths
        
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_images(opt=opt,    
            webpage=webpage,
            visuals=visuals,
            dataset_mode=dataset_mode,
            image_path=img_path, 
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize)

    webpage.save()  # save the HTML
    end_time = round(time.time() - init_time, 3)
    print_timestamped("The testing process took " + str(end_time) + "s.")
