#!/bin/bash

#SBATCH --job-name="gan1_hyperparam_training"
#SBATCH --time=02-00:00:00
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=a30
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

#read LAMBDA_L1 LAMBDA_L2 UPSAMPLING NGF < <(sed -n "${SLURM_ARRAY_TASK_ID}p" gan1_hyperparams.txt)

LAMBDA_L2=100
LAMBDA_L1=100

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath


    # start training
    python train.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --phase train \
    --save_epoch_freq 25 \
    --print_model_info \
    --name gan1_L1_${LAMBDA_L1}_L2_${LAMBDA_L2}_3107 \
    --model pix2pix3d \
    --dataset_mode gan1 \
    --norm instance \
    --lambda_L1 "$LAMBDA_L1" \
    --lambda_L2 "$LAMBDA_L2" \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_256 \
    --ngf 32 \
    --ndf 16 \
    --gan_mode 'lsgan' \
    --upsampling linear \
    --n_epochs_decay 0 \
    --n_epochs 100 \
    --preprocess none \
    --no_flip \
    --verbose \
    --L1_mask_only
EOF
