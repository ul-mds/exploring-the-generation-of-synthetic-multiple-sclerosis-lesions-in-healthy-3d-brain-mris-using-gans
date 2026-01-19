#!/bin/bash
lone=(0 100 500 1000)
ltwo=(0 100 500 1000)

# python3 ./datasets/make_nifti_dataset.py --folder /raid/gbaldini/braindata_all --save_folder /raid/gbaldini/pix_data --phase validation

for l in ${lone[@]}; do
  for i in ${!ltwo[@]}; do
    python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_resnet --gpu_id ${i} --fp16 \
    --preprocess none --netG resnet_6blocks --upsampling linear --postprocess 1 --excel \
    --lambda_L1 $l --gamma_TMSE ${ltwo[$i]} --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase validation \
    --dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/resnet_${l}_${ltwo[$i]}.txt" & 
  done
  wait
done

# python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_unet_load240crop --gpu_id 1 --fp16 \
# --preprocess pad --load_size 256 --netG unet_128 --upsampling linear --postprocess 1 --excel \
# --lambda_L1 100 --gamma_TMSE 100 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase validation \
# --dataroot /input --checkpoints_dir /checkpoints --results_dir /results >/results/unet_load240crop.txt

# nvidia-docker run --rm --mount type=bind,source=/raid/gbaldini/pix_data,target=/input --mount type=bind,source=/raid/gbaldini/pix_checkpoints,target=/checkpoints --mount type=bind,source=/raid/gbaldini/pix_output,target=/results pix_grid_val
