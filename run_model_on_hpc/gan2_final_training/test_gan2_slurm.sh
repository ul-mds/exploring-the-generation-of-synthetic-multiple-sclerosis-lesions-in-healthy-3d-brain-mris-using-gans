#!/bin/bash

#SBATCH --job-name="gan2_test"
#SBATCH --time=02-00:00:00
#SBATCH --partition=clara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=rtx2080ti
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x-%j-%a.out
#SBATCH --error=slurm-%x-%j-%a.err 
#SBATCH --array=1-10

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/Pix2PixNIfTI"

read LAMBDA_LS LAMBDA_BRAIN UPSAMPLING NGF FOLD < <(sed -n "${SLURM_ARRAY_TASK_ID}p" gan2_hyperparams.txt)

gan_name=gan2_left_out_fold${FOLD}_L1${LAMBDA_BRAIN}_L2${LAMBDA_LS}

apptainer exec --nv $IMAGE bash <<EOF
    cd $WORKDIR
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install mpmath

    # start testing
    python test_gan2.py \
    --dataroot /home/sc.uni-leipzig.de/iu71daji/thesis_gan2/train_gan2 \
    --results_dir /home/sc.uni-leipzig.de/iu71daji/thesis_gan2/testresults \
    --phase test \
    --lambda_L1 ${LAMBDA_BRAIN} \
    --lambda_L2 ${LAMBDA_LS} \
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
    --k_fold ${FOLD}
EOF
