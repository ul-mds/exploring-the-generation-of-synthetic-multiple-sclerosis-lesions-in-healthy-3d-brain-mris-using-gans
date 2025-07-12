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


IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

MODEL_NAME="gan1_L1_100_L2_0"

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath

    # Start testing
    python test.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1/gan1_test_imgs \
    --results_dir /home/sc.uni-leipzig.de/iu71daji/thesis_gan1/testresults \
    --phase test \
    --name ${MODEL_NAME} \
    --model pix2pix3d \
    --dataset_mode gan_1 \
    --norm instance \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_256 \
    --ngf 32 \
    --ndf 32 \
    --upsampling linear \
    --preprocess none \
    --no_flip \
    --verbose
EOF
