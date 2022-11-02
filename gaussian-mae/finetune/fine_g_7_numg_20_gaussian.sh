#!/bin/bash
#SBATCH --mem=250G # memory pool for all cores`
#SBATCH --time 120:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=faizan.khan@kaust.edu.sa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --job-name=fine_g_7_numg_20_gaussian
#SBATCH --output=logs/fine_g_7_numg_20_gaussian
#SBATCH --error=fine_g_7_numg_20_gaussian.err #The .error file name
#SBATCH --output=fine_g_7_numg_20_gaussian.out #The .output file name

cd ..
source /home/khanff/miniconda3/envs/lomar
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--master_addr=127.0.0.1 --master_port=29514 main_finetune.py \
    --batch_size 256 \
    --accum_iter 1 \
    --output_dir /ibex/ai/project/c2090/lomar_plus_save/checkpoint/raven/finetuned_gaussian_mae_encoderonly_numg_20_epochs_100 \
    --log_dir /ibex/ai/project/c2090/lomar_plus_save/logs/raven/finetuned_gaussian_mae_encoderonly_numg_20_epochs_100 \
    --model vit_base_patch16 \
    --finetune /ibex/ai/project/c2090/lomar_plus_save/checkpoint/raven/gaussian_mae_encoderonly_numg_20_epochs_100/checkpoint-99.pth \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --cutmix 1.0 \
    --dist_eval \
    --data_path /ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg
