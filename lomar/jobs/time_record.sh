#!/bin/bash
#SBATCH --mem=250G # memory pool for all cores`
#SBATCH --time 24:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=faizan.khan@kaust.edu.sa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --job-name=w_7_n_15_nw_4
#SBATCH --output=logs/w_7_n_15_nw_4
#SBATCH --error=w_7_n_15_nw_4_time.err #The .error file name
#SBATCH --output=w_7_n_15_nw_4_time.out #The .output file name
#SBATCH --account conf-cvpr-2022.11.18-elhosemh


source /home/khanff/miniconda3/envs/lomar
cd ..
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--master_addr=127.0.0.1 --master_port=29517 main_pretrain_lomar.py \
    --batch_size 256 \
    --accum_iter 4 \
    --output_dir /ibex/ai/project/c2090/lomar_plus_save/checkpoint/time_w_7_n_15_nw_4 \
    --log_dir /ibex/ai/project/c2090/lomar_plus_save/logs/w_7_n_15_nw_4 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --distributed \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --window_size 7 \
    --num_window 4 \
    --amp_autocast True \
    --neigh_ratio 0.15 \
    --mask_ratio 0.8 \
    --data_path /ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg
