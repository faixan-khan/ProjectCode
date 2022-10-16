#!/bin/bash
#SBATCH --mem=250G # memory pool for all cores`
#SBATCH --time 84:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=faizan.khan@kaust.edu.sa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --job-name=mae_encoderonly_mask_0.8_neigh_0.05_wind_7_8100epoches_384
#SBATCH --output=logs/mae_encoderonly_mask_0.8_neigh_0.05_wind_7_8100epoches_384
#SBATCH --error=lomar_mae384.err #The .error file name
#SBATCH --output=lomar_mae384.out #The .output file name
#SBATCH --account conf-cvpr-2022.11.18-elhosemh

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--master_addr=127.0.0.1 --master_port=47144 main_pretrain_lomar.py \
    --model mae_vit_base_patch16_384 \
    --num_window 4 \
    --batch_size 256 \
    --input_size 384 \
    --accum_iter 4 \
    --output_dir /ibex/ai/project/c2090/lomar_plus_save/checkpoint/mae_encoderonly_mask_0.8_neigh_0.05_wind_2_100epoches_384 \
    --log_dir /ibex/ai/project/c2090/lomar_plus_save/logs/mae_encoderonly_mask_0.8_neigh_0.05_wind_2_100epoches_384 \
    --norm_pix_loss \
    --distributed \
    --amp_autocast True \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --window_size 7 \
    --neigh_ratio 0.05 \
    --mask_ratio 0.8 \
    --data_path /ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg/
