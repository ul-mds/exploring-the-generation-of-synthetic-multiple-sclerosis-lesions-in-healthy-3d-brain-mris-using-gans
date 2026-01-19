#!/bin/bash

#SBATCH --job-name="gan1_hyperparam_training"
#SBATCH --time=02-00:00:00
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=a30
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x-%j_%a.out
#SBATCH --error=slurm-%x-%j_%a.err 
#SBATCH --array=1-25

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

read LAMBDA_LS LAMBDA_BG < <(sed -n "${SLURM_ARRAY_TASK_ID}p" gan1_hyperparameters.txt)


apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath


    # start training
    python train.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --phase train \
    --save_epoch_freq 150 \
    --print_model_info \
    --name gan1_hyptr_ls_${LAMBDA_LS}_bg_${LAMBDA_BG} \
    --model pix2pix3d \
    --dataset_mode gan1 \
    --norm instance \
    --lambda_ls "$LAMBDA_LS" \
    --lambda_bg "$LAMBDA_BG" \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_256 \
    --ngf 32 \
    --ndf 32 \
    --gan_mode 'lsgan' \
    --upsampling linear \
    --n_epochs_decay 0 \
    --n_epochs 600 \
    --preprocess none \
    --no_flip \
    --verbose \
    --L1_mask_only
EOF
