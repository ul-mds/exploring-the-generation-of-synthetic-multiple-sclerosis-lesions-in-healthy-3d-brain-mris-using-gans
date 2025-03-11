#!/bin/bash

python train.py \
  --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
  --model pix2pix3d \
  --name gan1_model \
  --gpu_ids 0 \
  --norm instance \
  --input_nc 1 \
  --output_nc 1 \
  --netG unet_256 \
  --ngf 16 \
  --ndf 32 \
  --upsampling deconvolution \
  --preprocess none \
  --no_flip \
  --verbose \
  --display_freq 100 \
  --print_freq 50 \
  --update_html_freq 200 \
  --save_epoch_freq 10 \
  --print_model_info \
  --lambda_L1 0.0 \
  --gamma_TMSE 0.0 \
  --n_epochs 100 \
  --dataset_mode my_nifti \
  --phase train
