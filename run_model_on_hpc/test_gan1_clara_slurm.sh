#!/bin/bash

#SBATCH --job-name="train_gan1"
#SBATCH --time=02-00:00:00
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=v100
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath

    # Start testing
    python test.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --results_dir /home/sc.uni-leipzig.de/iu71daji/thesis_gan1/test_results_gan1_WmInTarget_5 \
    --phase test \
    --name gan1_WmInTarget_5 \
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
    --verbose 
EOF
    
