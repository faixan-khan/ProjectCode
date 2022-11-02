#!/bin/bash
#SBATCH --mem=250G # memory pool for all cores`
#SBATCH --time 124:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=faizan.khan@kaust.edu.sa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --job-name=finetune_original_mae_m_75_r_224
#SBATCH --output=logs/finetune_original_mae_m_75_r_224
#SBATCH --error=finetune_mae_m_75_r_224.err #The .error file name
#SBATCH --output=finetune_mae_m_75_r_224.out #The .output file name
#SBATCH --account conf-cvpr-2022.11.18-elhosemh

cd ..
source /home/khanff/miniconda3/envs/lomar
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--master_addr=127.0.0.1 --master_port=29517 main_finetune.py \
    --batch_size 256 \
    --accum_iter 1 \
    --output_dir /ibex/ai/project/c2090/lomar_plus_save/checkpoint/raven_mae/finetune_original_mae_m_75_r_224 \
    --log_dir /ibex/ai/project/c2090/lomar_plus_save/logs/raven_mae/finetune_original_mae_m_75_r_224 \
    --model vit_base_patch16 \
    --finetune /ibex/ai/project/c2090/lomar_plus_save/checkpoint/raven_mae/original_mae_m_75_r_224/checkpoint-99.pth \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --cutmix 1.0 \
    --data_path /ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg