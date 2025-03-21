#!/bin/bash

#SBATCH --job-name="gan1_cropped_L1_masks_2"
#SBATCH --time=02-00:00:00
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=v100
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err 
#SBATCH --array=1-5  

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

ALL_GAMMA_T=(1 10 20 50 100)

INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Die Werte für diesen Job holen
GAMMA_T=${ALL_GAMMA_T[$INDEX]}

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath


    # start training
    python train.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --phase train \
    --save_epoch_freq 20 \
    --print_model_info \
    --name gan1_cropped_L1_masks_2 \
    --model pix2pix3d \
    --dataset_mode my_nifti \
    --norm instance \
    --lambda_L1 300 \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_128 \
    --ngf 32 \
    --ndf 16 \
    --gan_mode 'lsgan' \
    --upsampling deconvolution \
    --n_epochs_decay 0 \
    --n_epochs 800 \
    --preprocess none \
    --no_flip \
    --verbose \
    --L1_lesions_only \
    --gamma_TMSE "$GAMMA_T" \
    --crop_imgs
EOF
