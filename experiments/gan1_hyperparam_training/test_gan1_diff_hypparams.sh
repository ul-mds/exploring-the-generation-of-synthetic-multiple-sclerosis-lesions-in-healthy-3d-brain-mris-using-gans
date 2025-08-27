#!/bin/bash

#SBATCH --job-name="test_gan1"
#SBATCH --time=02-00:00:00
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=a30
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=1-100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

read LAMBDA_L1 LAMBDA_L2 < <(sed -n "${SLURM_ARRAY_TASK_ID}p" gan1_hyperparameters.txt)

MODEL_NAME="gan1_hyptr_L1_${LAMBDA_L1}_L2_${LAMBDA_L2}"

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath

    # Start testing
    for EPOCH in 150 300 450 600; do
    python test.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --results_dir /home/sc.uni-leipzig.de/iu71daji/thesis_gan1/testresults \
    --phase test \
    --name ${MODEL_NAME} \
    --model pix2pix3d \
    --dataset_mode gan1 \
    --norm instance \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_256 \
    --ngf 32 \
    --ndf 16 \
    --upsampling linear \
    --preprocess none \
    --no_flip \
    --verbose \
    --epoch "\$EPOCH"
    done
EOF

