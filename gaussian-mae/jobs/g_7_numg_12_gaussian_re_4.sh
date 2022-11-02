#!/bin/bash
#SBATCH --mem=250G # memory pool for all cores`
#SBATCH --time 44:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=faizan.khan@kaust.edu.sa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
#SBATCH --job-name=g_7_numg_12_gaussian_re_4
#SBATCH --output=logs/g_7_numg_12_gaussian_re_4
#SBATCH --error=g_7_numg_12_gaussian_re_4.err #The .error file name
#SBATCH --output=g_7_numg_12_gaussian_re_4.out #The .output file name
#SBATCH --account conf-cvpr-2022.11.18-elhosemh


source /home/khanff/miniconda3/envs/lomar
cd ..
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--master_addr=127.0.0.1 --master_port=29517 main_pretrain.py \
    --batch_size 256 \
    --accum_iter 4 \
    --num_gaussians 12 \
    --reconstruction_per_gaussian 4 \
    --output_dir /ibex/ai/project/c2090/lomar_plus_save/checkpoint/raven/gaussian_mae_encoderonly_numg_12_re_4_epochs_100 \
    --log_dir /ibex/ai/project/c2090/lomar_plus_save/logs/raven/gaussian_mae_encoderonly_numg_12_re_4_epochs_100 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg