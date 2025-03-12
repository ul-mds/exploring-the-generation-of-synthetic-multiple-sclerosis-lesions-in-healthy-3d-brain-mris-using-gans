#!/bin/bash

#SBATCH --job-name="train_gan1"
#SBATCH --time=02-00:00:00
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=a30
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath

    # start training
    python train.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --phase train \

    # print options
    --display_freq 100 \
    --print_freq 50 \
    --update_html_freq 200 \
    --save_epoch_freq 10 \
    --print_model_info \

    # options similar to test_gan1_slurm.sh
    --name gan1_WmInTarget_3 \
    --model pix2pix3d \
    --dataset_mode my_nifti \
    --norm none \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_128 \
    --ngf 32 \
    --ndf 32 \
    --gan_mode 'lsgan' \
    --upsampling deconvolution \
    --preprocess none \
    --no_flip \
    --verbose 
EOF
