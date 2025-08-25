#!/bin/bash

#SBATCH --job-name="gan1_eval"
#SBATCH --time=09:00:00
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=a30
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE="/home/sc.uni-leipzig.de/iu71daji/masterthesis/containers/nifti_img.sif"
WORKDIR="/home/sc.uni-leipzig.de/iu71daji/masterthesis/mri_preprocessing_and_segmentation/gan1_evaluation"

cd $WORKDIR

apptainer exec --nv $IMAGE python eval_gan1_hyperparam_training.py
