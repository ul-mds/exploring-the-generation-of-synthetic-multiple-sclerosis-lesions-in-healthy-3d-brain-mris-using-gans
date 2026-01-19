"""
General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

The script creates model, dataset, and visualizer given the options, then performs standard network training.
During training, it visualizes/saves images, prints/saves loss plots, and saves models.
The script supports continue/resume training. Use '--continue_train' to resume previous training.

Example usage:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

Author/Mod: Franziska Röpke (modified 2026-01-15)

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md

    Originally from pix2pixnifti repository.
    Modified: Franziska Röpke (modified 2026-01-15)
"""
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import print_timestamped
import time
import torch 

if __name__ == '__main__':
    opt = TrainOptions().parse()
    
    # Disable display for 3D training (very memory intensive)
    if opt.model == "pix2pix3d":
        opt.display_id = -1
    
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    
    model = create_model(opt)
    model.setup(opt)
    
    # Print network information if requested
    if opt.print_model_info and "pix2pix" in opt.model:
        from torchsummary import summary
        final_size = opt.crop_size if "crop" in opt.preprocess else opt.load_size
        if "3d" in opt.model:
            summary(model.netG, (opt.input_nc, final_size, final_size, final_size))
        else:
            summary(model.netG, (opt.input_nc, final_size, final_size))
    
    visualizer = Visualizer(opt)
    total_iters = 0
    init_time = time.time()

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        visualizer.reset()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            if hasattr(opt, 'L1_mask_only'):
                model.set_input(data, opt.L1_mask_only)
            else:
                model.set_input(data)

            model.optimize_parameters(i)

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, save_result)
                
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('Saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            
        model.update_learning_rate()
        
        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
    end_time = round(time.time() - init_time, 3)
    print_timestamped("Training process completed in " + str(end_time) + "s.")
