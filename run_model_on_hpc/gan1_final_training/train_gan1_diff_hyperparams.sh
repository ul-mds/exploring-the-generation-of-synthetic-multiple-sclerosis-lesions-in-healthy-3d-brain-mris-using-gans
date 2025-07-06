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
#SBATCH --array=1-10

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

read LAMBDA_LESION LAMBDA_BG UPSAMPLING NGF < <(sed -n "${SLURM_ARRAY_TASK_ID}p" hyperparams.txt)


apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath


    # start training
    python train.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --phase train \
    --save_epoch_freq 100 \
    --print_model_info \
    --name gan1_${SLURM_ARRAY_TASK_ID}_L1{LAMBDA_LESION}_L2${LAMBDA_BG}_${UPSAMPLING}_ngf${NGF}_060725 \
    --model pix2pix3d \
    --dataset_mode gan1 \
    --norm instance \
    --lambda_L1 "$LAMBDA_LESION" \
    --lambda_L2_background "$LAMBDA_BG" \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_256 \
    --ngf ${NGF} \
    --ndf 16 \
    --gan_mode 'lsgan' \
    --upsampling ${UPSAMPLING} \
    --n_epochs_decay 0 \
    --n_epochs 800 \
    --preprocess none \
    --no_flip \
    --verbose \
    --L1_lesions_only
EOF
