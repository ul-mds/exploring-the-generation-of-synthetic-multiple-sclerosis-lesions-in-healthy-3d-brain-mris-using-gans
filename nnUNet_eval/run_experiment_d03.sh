#!/bin/bash

#SBATCH --job-name="train_nnUNet"
#SBATCH --time=14:00:00
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=a30
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

DATASET_ID="003"
DATASET_NAME="Dataset003_RealSynthetic"
TEST_RESULTS_DIR="test_results_003_RealSynthetic"

# load myenv environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate myenv

# load CUDA for PyTorch
module load CUDA/12.6.0

# set environment variables
export nnUNet_raw="/work/iu71daji-pipeline_eval/nnUNet_raw"
export nnUNet_preprocessed="/work/iu71daji-pipeline_eval/nnUNet_preprocessed"
export nnUNet_results="/work/iu71daji-pipeline_eval/nnUNet_results"

nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity

nnUNetv2_train ${DATASET_ID} 3d_fullres 0

nnUNetv2_predict -i ${nnUNet_raw}/${DATASET_NAME}/imagesTs \
			-o ${nnUNet_raw}/${TEST_RESULTS_DIR} \
			-d ${DATASET_ID} \
			-c 3d_fullres \
			-f 0 
