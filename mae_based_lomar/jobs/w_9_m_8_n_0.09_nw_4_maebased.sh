#!/bin/bash
#SBATCH --mem=250G # memory pool for all cores`
#SBATCH --time 44:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=faizan.khan@kaust.edu.sa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --job-name=w_9_m_8_n_15_nw_4_maebased
#SBATCH --output=logs/w_9_m_8_n_15_nw_4_maebased
#SBATCH --error=w_9_m_8_n_15_nw_4_maebased.err #The .error file name
#SBATCH --output=w_9_m_8_n_15_nw_4_maebased.out #The .output file name
#SBATCH --account conf-cvpr-2022.11.18-elhosemh


source /home/khanff/miniconda3/envs/lomar
cd ..
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--master_addr=127.0.0.1 --master_port=29516 main_pretrain.py \
    --batch_size 256 \
    --accum_iter 4 \
    --output_dir /ibex/ai/project/c2090/lomar_plus_save/checkpoint/raven/mae_based_mask_0.8_neigh_0.09_wind_9_num_4_epochs_100 \
    --log_dir /ibex/ai/project/c2090/lomar_plus_save/logs/raven/mae_based_mask_0.8_neigh_0.09_wind_9_num_4_epochs_100 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --window_size 9 \
    --num_window 4 \
    --mask_ratio 0.8 \
    --neigh_ratio 0.09 \
    --data_path /ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg
