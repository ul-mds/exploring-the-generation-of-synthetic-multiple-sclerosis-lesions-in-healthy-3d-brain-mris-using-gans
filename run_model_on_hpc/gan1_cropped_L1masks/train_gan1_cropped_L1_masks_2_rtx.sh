#!/bin/bash

#SBATCH --job-name="gan1_cropped_L1_masks_2"
#SBATCH --time=02-00:00:00
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=rtx2080ti
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x-%j-%a.out
#SBATCH --error=slurm-%x-%j-%a.err 
#SBATCH --array=1-20  # 5 Gamma-Werte × 4 Epochen = 20 Kombinationen

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

ALL_GAMMA_T=(1 10 20 50 100)
ALL_EPOCHS=(200 400 600 800)

# SLURM-Array-Index berechnen
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Berechnen, welcher Gamma-Wert und welche Epoche benutzt werden
GAMMA_INDEX=$((INDEX / 4))  # Ganzzahldivision, um Gamma-Wert zu bestimmen
EPOCH_INDEX=$((INDEX % 4))  # Modulo, um Epoche zu bestimmen

GAMMA_T=${ALL_GAMMA_T[$GAMMA_INDEX]}
EPOCH=${ALL_EPOCHS[$EPOCH_INDEX]}

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath

    # start testing
    python test.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --phase test \
    --save_epoch_freq 20 \
    --print_model_info \
    --lambda_L1 300 \
    --input_nc 1 \
    --name gan1_cropped_L1_masks_2_Tgamma_${GAMMA_T} \
    --model pix2pix3d \
    --dataset_mode my_nifti \
    --norm instance \
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
    --crop_imgs \
    --epoch "$EPOCH"
EOF
