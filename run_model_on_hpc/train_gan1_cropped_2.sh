#!/bin/bash

#SBATCH --job-name="train_gan1"
#SBATCH --time=02-00:00:00
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=rtx2080ti
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath

    # Start monitoring GPU usage in the background
    # Logs the GPU usage to the slurm.out file every 10 seconds
    #nohup watch -n 300 nvidia-smi >> gpu_usage.log &

    # start training
    python train.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --phase train \
    --display_freq 100 \
    --print_freq 50 \
    --update_html_freq 200 \
    --save_epoch_freq 10 \
    --print_model_info \
    --name gan1_cropped_2 \
    --model pix2pix3d \
    --dataset_mode my_nifti \
    --norm instance \
    --lambda_L1 300 \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_128 \
    --netD n_layers \
    --n_layers_D 5 \
    --ngf 32 \
    --ndf 16 \
    --gan_mode 'lsgan' \
    --upsampling deconvolution \
    --n_epochs_decay 0 \
    --n_epochs 100 \
    --preprocess none \
    --no_flip \
    --verbose 
EOF
