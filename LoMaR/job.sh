#!/bin/bash
#SBATCH --mem=250G # memory pool for all cores`
#SBATCH --time 24:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=faizan.khan@kaust.edu.sa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --job-name=te_mae_encoderonly_mask_0.8_wind_7_100epoches
#SBATCH --output=logs/mae_encoderonly_mask_0.8_wind_7_100epoches
#SBATCH --error=test_lomar_mae.err #The .error file name
#SBATCH --output=test_lomar_mae.out #The .output file name
#SBATCH --constraint=[v100]
#SBATCH --account conf-cvpr-2022.11.18-elhosemh

source /home/khanff/miniconda3/envs/lomar
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--master_addr=127.0.0.1 --master_port=29516 main_pretrain_lomar.py \
    --batch_size 256 \
    --accum_iter 4 \
    --output_dir /ibex/ai/project/c2090/lomar_plus_save/checkpoint/linear_mae_encoderonly_mask_0.8_wind_7_num_2_epochs_100 \
    --log_dir /ibex/ai/project/c2090/lomar_plus_save/logs/linear_mae_encoderonly_mask_0.8_wind_7_num_2_epochs_100 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --epochs 100 \
    --input_size 224 \
    --warmup_epochs 5 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --window_size 7 \
    --num_window 2 \
    --mask_ratio 0.8 \
    --data_path /ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg
