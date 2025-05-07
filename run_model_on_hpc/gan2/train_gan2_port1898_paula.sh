#!/bin/bash

#SBATCH --job-name="ga2_train"
#SBATCH --time=02-00:00:00
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=a30
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
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan2/train_gan2 \
    --phase train \
    --display_freq 100 \
    --print_freq 50 \
    --update_html_freq 200 \
    --save_epoch_freq 50 \
    --print_model_info \
    --name gan2_1_port1898 \
    --model pix2pix \
    --dataset_mode gan2 \
    --norm instance \
    --lambda_L1 300 \
    --gamma_TMSE 0 \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_256 \
    --ngf 32 \
    --ndf 32 \
    --gan_mode 'lsgan' \
    --upsampling deconvolution \
    --preprocess none \
    --no_flip \
    --verbose \
    --display_port 1898 \
    --num_threads 0
EOF
