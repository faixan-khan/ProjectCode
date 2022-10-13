#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --mail-user=faizan.khan@kaust.edu.sa #Your Email address assigned for your job
#SBATCH --mail-type=ALL #Receive an email for ALL Job Statuses
#SBATCH -J mae
#SBATCH --error=mae.err #The .error file name
#SBATCH --output=mae.out #The .output file name
#SBATCH --time=72:00:00 #Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --nodes=1 #Number of Nodes desired e.g 1 node
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=[v100]
#SBATCH --mem=60G

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--master_addr=127.0.0.1 --master_port=47144 main_pretrain.py \
    --output_dir ./logger_0.1_neigh \
    --log_dir ./logger_0.1_neigh \
    --batch_size 256 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg/
    --distributed \