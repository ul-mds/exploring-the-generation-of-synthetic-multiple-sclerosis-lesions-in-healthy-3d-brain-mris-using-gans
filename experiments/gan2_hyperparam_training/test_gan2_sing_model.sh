#!/bin/bash

#SBATCH --job-name="gan2_test"
#SBATCH --time=02-00:00:00
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=a30
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x-%j-%a.out
#SBATCH --error=slurm-%x-%j-%a.err 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

gan_name="gan2_final_model_L1_bg_100_L2_ls_20_3"
LAMBDA_L1=100
LAMBDA_L2=20

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath

    # start testing
    for EPOCH in 100 200 300 400; do
    python test_gan2.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan2/train_gan2/fold_4 \
    --results_dir /home/sc.uni-leipzig.de/iu71daji/thesis_gan2/testresults \
    --phase inference \
    --lambda_bg ${LAMBDA_L1} \
    --lambda_ls ${LAMBDA_L2} \
    --name ${gan_name} \
    --model pix2pix \
    --dataset_mode gan2 \
    --norm instance \
    --netG unet_256 \
    --ngf 32 \
    --ndf 32 \
    --upsampling linear \
    --preprocess none \
    --no_flip \
    --num_test 100 \
    --output_nc 3 \
    --input_nc 3 \
    --verbose \
    --epoch 100
    done
EOF
