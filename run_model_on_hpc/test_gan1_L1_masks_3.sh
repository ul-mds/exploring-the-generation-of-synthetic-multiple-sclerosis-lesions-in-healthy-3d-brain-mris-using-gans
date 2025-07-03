#!/bin/bash

#SBATCH --job-name="test_gan1"
#SBATCH --time=02-00:00:00
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=rtx2080ti
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=1-8  # <== Stelle hier die Anzahl der Jobs ein!


IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

ALL_LAMBDA_L1=(200 300 400 500)
ALL_EPOCHS=(200 400 600 800)

# Slurm-Array beginnt bei 1, also Index um 1 reduzieren
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

LAMBDA_INDEX=$((INDEX / 4))  # Ganzzahldivision, um Gamma-Wert zu bestimmen
EPOCH_INDEX=$((INDEX % 4))  # Modulo, um Epoche zu bestimmen

LAMBDA=${ALL_LAMBDA_L1[$LAMBDA_INDEX]}
EPOCH=${ALL_EPOCHS[$EPOCH_INDEX]}

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath

    # Start testing
    python test.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --results_dir /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --phase test \
    --name gan1_L1_masks_3_L${LAMBDA} \
    --model pix2pix3d \
    --dataset_mode my_nifti \
    --norm instance \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_256 \
    --ngf 32 \
    --ndf 16 \
    --upsampling deconvolution \
    --preprocess none \
    --no_flip \
    --verbose \
    --epoch "$EPOCH"
EOF
