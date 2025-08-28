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

LAMBDA_L1=70
LAMBDA_L2=150
EPOCHS=100

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath


    # start training
    python train.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan2/train_gan2/all_traindata \
    --phase train \
    --save_epoch_freq 100\
    --print_model_info \
    --name gan2_final_model_L1_${LAMBDA_L1}_L2_${LAMBDA_L2} \
    --model pix2pix \
    --dataset_mode gan2 \
    --norm instance \
    --lambda_L1 ${LAMBDA_L1} \
    --lambda_L2 ${LAMBDA_L2} \
    --netG unet_256 \
    --ngf 32 \
    --ndf 32 \
    --gan_mode 'lsgan' \
    --upsampling linear \
    --n_epochs_decay 0 \
    --n_epochs ${EPOCHS} \
    --preprocess none \
    --no_flip \
    --display_id 0 \
    --verbose \
    --L1_mask_only
EOF
