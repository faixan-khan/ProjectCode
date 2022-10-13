#!/bin/bash
#SBATCH --mem=250G # memory pool for all cores`
#SBATCH --time 24:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=faizan.khan@kaust.edu.sa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --job-name=mae_encoder_with_decoder_mask_0.85_wind_7_100epoches
#SBATCH --output=logs/mae_encoder_with_decoder_mask_0.85_wind_7_100epoches
#SBATCH --error=lomar_mae85.err #The .error file name
#SBATCH --output=lomar_mae85.out #The .output file name


source /home/khanff/miniconda3/envs/lomar
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--master_addr=127.0.0.1 --master_port=29516 main_pretrain.py \
    --batch_size 256 \
    --accum_iter 4 \
    --output_dir /ibex/ai/project/c2090/lomar_plus_save/checkpoint/mae_encoder_with_decoder_mask_0.85_wind_7_100epoches \
    --log_dir /ibex/ai/project/c2090/lomar_plus_save/logs/mae_encoder_with_decoder_mask_0.85_wind_7_100epoches \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --window_size 7 \
    --num_window 2 \
    --mask_ratio 0.85 \
    --data_path /ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg
