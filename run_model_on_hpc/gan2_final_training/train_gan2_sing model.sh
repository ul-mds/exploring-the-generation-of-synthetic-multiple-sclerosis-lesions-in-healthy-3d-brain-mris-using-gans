#!/bin/bash

#SBATCH --job-name="gan2_train"
#SBATCH --time=02-00:00:00
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=rtx2080ti
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

LAMBDA_LS=0
LAMBDA_BRAIN=100
UPSAMPLING="linear"
NGF=32
FOLD=1
apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath


    # start training
    python train.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan2/train_gan2 \
    --phase train \
    --save_epoch_freq 50 \
    --print_model_info \
    --name gan2_left_out_fold${FOLD}_L1${LAMBDA_BRAIN}_L2${LAMBDA_LS}_${UPSAMPLING}_ngf${NGF}_noMask1207 \
    --model pix2pix \
    --dataset_mode gan2 \
    --norm instance \
    --lambda_L1 ${LAMBDA_BRAIN} \
    --lambda_L2 ${LAMBDA_LS} \
    --netG unet_256 \
    --ngf ${NGF} \
    --ndf 16 \
    --gan_mode 'lsgan' \
    --upsampling ${UPSAMPLING} \
    --n_epochs_decay 0 \
    --n_epochs 50 \
    --preprocess none \
    --no_flip \
    --display_id 0 \
    --verbose \
    --L1_mask_only \
    --k_fold ${FOLD}
EOF
