#!/bin/bash


python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_resnet --gpu_id 0 --fp16 \
--preprocess none --netG resnet_6blocks --upsampling linear --postprocess 1 --excel \
--lambda_L1 100 --gamma_TMSE 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testBraTS \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/testbrats_resnet_100_0.txt" & \
python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_resnet --gpu_id 1 --fp16 \
--preprocess none --netG resnet_6blocks --upsampling linear --postprocess 1 --excel \
--lambda_L1 500 --gamma_TMSE 500 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testBraTS \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/testbrats_resnet_500_500.txt"

wait

python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_resnet --gpu_id 0 --fp16 \
--preprocess none --netG resnet_6blocks --upsampling linear --postprocess 1 --excel \
--lambda_L1 100 --gamma_TMSE 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testUK \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/testuk_resnet_100_0.txt" & \
python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_resnet --gpu_id 1 --fp16 \
--preprocess none --netG resnet_6blocks --upsampling linear --postprocess 1 --excel \
--lambda_L1 500 --gamma_TMSE 500 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testUK \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/testuk_resnet_500_500.txt" & \

wait

python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_resnet --gpu_id 0 --fp16 \
--preprocess none --netG resnet_6blocks --upsampling linear --postprocess 1 --excel \
--lambda_L1 100 --gamma_TMSE 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testNoTruth \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/notruth_resnet_100_0.txt" & \
python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_resnet --gpu_id 1 --fp16 \
--preprocess none --netG resnet_6blocks --upsampling linear --postprocess 1 --excel \
--lambda_L1 500 --gamma_TMSE 500 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase testNoTruth \
--dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/notruth_resnet_500_500.txt"