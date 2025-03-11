#!/bin/bash

#SBATCH --job-name="train_gan1"
#SBATCH --time=02-00:00:00
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=a30
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


IMAGE="nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

apptainer exec --nv $IMAGE bash -c "
    cd $WORKDIR
    pip install --upgrade pip
    pip install mpmath -r requirements.txt

    # Starte das Training
    python test.py --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan1 \
    --results_dir /home/sc.uni-leipzig.de/iu71daji/thesis_gan1/test_results \
    --name gan1_model \
    --model pix2pix3d \
    --dataset_mode my_nifti \
    --phase test \
    --eval \
    --gpu_ids 0 \
    --norm instance \
    --input_nc 1 \
    --output_nc 1 \
    --netG unet_256 \
    --ngf 16 \
    --ndf 32 \
    --upsampling deconvolution \
    --preprocess none \
    --no_flip \
    --verbose \
    --lambda_L1 0.0 \
    --gamma_TMSE 0.0 
"
