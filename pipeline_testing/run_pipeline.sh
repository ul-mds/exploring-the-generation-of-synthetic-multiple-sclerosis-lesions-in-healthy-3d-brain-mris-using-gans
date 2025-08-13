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

# Variables to set: 
GAN1_SOURCE="/home/sc.uni-leipzig.de/iu71daji/test_pipeline/test_imgs/gan1_source"
GAN1_NAME="gan1_hyptr_L1_100_L2_250"
GAN1_EPOCH=450
GAN1_RESULTS_DIR="/home/sc.uni-leipzig.de/iu71daji/test_pipeline/gan1results_synthetic_masks"


REGISTERED_FLAIRS="/home/sc.uni-leipzig.de/iu71daji/test_pipeline/test_imgs/registered_flairs"
GAN2_SOURCE="/home/sc.uni-leipzig.de/iu71daji/test_pipeline/test_imgs/gan2_source"
GAN2_RESULTS_DIR="/home/sc.uni-leipzig.de/iu71daji/test_pipeline/gan2results"
GAN2_NAME="gan2_left_out_fold3_L1_150_L2_70"
GAN2_EPOCH=200

PIPELINE_RESULTS_DIR="/home/sc.uni-leipzig.de/iu71daji/test_pipeline/final_pipeline_results"


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR

    # run gan1
    python test.py \
    --dataroot $GAN1_SOURCE \
    --results_dir $GAN1_RESULTS_DIR \
    --phase test \
    --name ${GAN1_NAME} \
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
    --epoch "\$GAN1_EPOCH"
EOF

# gan2 preprocessing with new synthetic lesionmask 
# call: 
apptainer exec --nv $IMAGE bash <<EOF 
	cd $WORKDIR	
	python pipeline_testing/gan2_preprocess.py \
	--flairs_path $REGISTERED_FLAIRS \
	--lsmasks_path $GAN1_RESULTS_DIR \
	--output_path $GAN2_SOURCE
EOF

# run gan2
apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR

    # start testing
    python test_gan2.py \
    --dataroot $GAN2_SOURCE \
    --results_dir $GAN2_RESULTS_DIR \
    --phase inference \
    --name $GAN2_NAME \
    --model pix2pix \
    --dataset_mode gan2 \
    --norm instance \
    --netG unet_256 \
    --ngf 32 \
    --ndf 32 \
    --upsampling linear \
    --preprocess none \
    --no_flip \
    --output_nc 3 \
    --input_nc 3 \
    --verbose \
    --epoch $GAN2_EPOCH
EOF

# postprocess gan2 results
apptainer exec --nv $IMAGE bash <<EOF
	cd $WORKDIR	
 	python pipeline_testing/postprocessing.py \
	--slices_dir $GAN2_RESULTS_DIR \
	--reference_flair_path $REGISTERED_FLAIRS \
	--lsmasks_path $GAN1_RESULTS_DIR \
	--output_path $PIPELINE_RESULTS_DIR
EOF 
