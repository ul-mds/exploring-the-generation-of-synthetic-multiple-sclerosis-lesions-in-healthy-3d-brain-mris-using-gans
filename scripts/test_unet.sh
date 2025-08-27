#!/bin/bash

python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_unet --gpu_id 0 --fp16 \
--preprocess pad --load_size 256 --netG unet_128 --upsampling linear --postprocess 1 --excel \
--lambda_L1 100 --gamma_TMSE 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testBraTS \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/testbrats_unet_out_100_0.txt" & \
python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_unet --gpu_id 1 --fp16 \
--preprocess pad --load_size 256 --netG unet_128 --upsampling linear --postprocess 1 --excel \
--lambda_L1 500 --gamma_TMSE 1000 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testBraTS \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/testbrats_unet_out_500_1000.txt"

wait

python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_unet --gpu_id 2 --fp16 \
--preprocess pad --load_size 256 --netG unet_128 --upsampling linear --postprocess 1 --excel \
--lambda_L1 100 --gamma_TMSE 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testUK \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/testuk_unet_out_100_0.txt" & \
python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_unet --gpu_id 3 --fp16 \
--preprocess pad --load_size 256 --netG unet_128 --upsampling linear --postprocess 1 --excel \
--lambda_L1 500 --gamma_TMSE 1000 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testUK \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/testuk_unet_out_500_1000.txt"

wait

python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_unet --gpu_id 0 --fp16 \
--preprocess pad --load_size 256 --netG unet_128 --upsampling linear --postprocess 1 --excel \
--lambda_L1 100 --gamma_TMSE 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testNoTruth \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/notruth_unet_out_100_0.txt" & \
python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_unet --gpu_id 1 --fp16 \
--preprocess pad --load_size 256 --netG unet_128 --upsampling linear --postprocess 1 --excel \
--lambda_L1 500 --gamma_TMSE 1000 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testNoTruth \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/notruth_unet_out_500_1000.txt"