#!/bin/bash

#SBATCH --job-name="gan1_L1_masks_3"
#SBATCH --time=02-00:00:00
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=v100
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err 
#SBATCH --array=1-4

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

ALL_LAMBDA_L1=(200 300 400 500)

INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Die Werte für diesen Job holen
LAMBDA=${ALL_LAMBDA_L1[$INDEX]}

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
    --name gan1_L1_masks_3 \
    --model pix2pix3d \
    --dataset_mode my_nifti \
    --norm instance \
    --lambda_L1 "$LAMBDA" \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_256 \
    --ngf 32 \
    --ndf 16 \
    --gan_mode 'lsgan' \
    --upsampling deconvolution \
    --n_epochs_decay 0 \
    --n_epochs 800 \
    --preprocess none \
    --no_flip \
    --verbose \
    --L1_lesions_only
EOF
