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
#from util.visualizer import save_images, save_nifti_images, save_web_nifti
from util import html_handler
from util.util import postprocess_images, print_timestamped
import time
import imageio
from util import util


# FIMOS: 
def save_png_images(visuals, query_name, opt, web_dir):
    """
    Speichert 2D-Bilder als PNG-Dateien statt als NIfTI.

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

    if len(image.shape) > 4:
        continue  # Volumen überspringen
    else:
        # Trainingslogik: nur Kanal 0
        if image.shape[1] == 3:
            image_numpy = image[0, 0].detach().cpu().numpy()
        else:
            image_numpy = image[0].detach().cpu().numpy()

        image_numpy = image_numpy.astype(np.uint8)

        # Speichern mit imageio
        imageio.imwrite(img_path, image_numpy)




if __name__ == '__main__':
    
    # Fimos debugpy
    import debugpy
    # Starte den Debugging-Server
    debugpy.listen(("0.0.0.0", 5678))
    print("🚀 Warten auf Debugger Attach auf Port 5678...")

    # Optional: hier anhalten bis sich dein Debugger verbunden hat
    debugpy.wait_for_client()
    print("✅ Debugger verbunden!")


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
    # Determine whether we are using nifti images
    nifti = True if opt.dataset_mode == "nifti" else False
    # create a website
    exc_eval = None
    if nifti:
        web_dir = os.path.join(opt.results_dir, opt.name,
                               '{}_{}_{}'.format(opt.phase, opt.epoch, opt.postprocess))  # define the website directory
        excel_filename = os.path.join(web_dir, (opt.phase + "_" + str(opt.name) + ".csv"))
        exc_eval = ExcelEvaluate(excel_filename, opt.excel)
    else:
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
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        
        query_name = os.path.basename(img_path[0])
        query_name = query_name.split(".")[0]

        save_png_images(visuals, query_name, opt, web_dir)
        # Postprocess the results to numpy arrays
        # np_post = postprocess_images(visuals, opt, dataset.dataset.original_shape)
        #if 'real_B' in np_post:
        #    # Evaluate the results
        #    exc_eval.evaluate(np_post, query_name_next, opt.smoothing)
        
    
    webpage.save()  # save the HTML
    end_time = round(time.time() - init_time, 3)
    print_timestamped("The testing process took " + str(end_time) + "s.")
    if nifti:
        exc_eval.close()
