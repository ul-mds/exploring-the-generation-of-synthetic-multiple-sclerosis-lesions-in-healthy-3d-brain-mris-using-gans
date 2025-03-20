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
#SBATCH --array=1-3  # <== Stelle hier die Anzahl der Jobs ein!


IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

EPOCHS=(500 600 700 800)  # Liste der Epochen
RESULTS_DIRS=(
    "/home/sc.uni-leipzig.de/iu71daji/thesis_gan1/gan1_cropped_1_500eps"
    "/home/sc.uni-leipzig.de/iu71daji/thesis_gan1/gan1_cropped_1_600eps"
    "/home/sc.uni-leipzig.de/iu71daji/thesis_gan1/gan1_cropped_1_700eps"
    "/home/sc.uni-leipzig.de/iu71daji/thesis_gan1/gan1_cropped_1_800eps"
)

# Slurm-Array beginnt bei 1, also Index um 1 reduzieren
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Die Werte für diesen Job holen
EPOCH=${EPOCHS[$INDEX]}
RESULTS=${RESULTS_DIRS[$INDEX]}

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath

    # Start testing
    python test.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --results_dir "$RESULTS" \
    --phase test \
    --name gan1_cropped_1_100eps \
    --model pix2pix3d \
    --dataset_mode my_nifti \
    --norm instance \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_128 \
    --ngf 32 \
    --ndf 16 \
    --upsampling deconvolution \
    --preprocess none \
    --no_flip \
    --verbose \
    --epoch "$EPOCH"
EOF
