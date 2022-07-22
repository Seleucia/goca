#!/bin/bash

# Parameters
#SBATCH --job-name=kinetics_swav
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=200G
#SBATCH --output=logs/job_logs/slrm_knn_stdout.%j
#SBATCH --error=logs/job_logs/slrm_knn_stderr.%j
#MASTER_ADDR
#MASTER_PORT=40000
#master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
#dist_url="tcp://"
#dist_url+=$master_node
#dist_url+=:40000

master_node=${SLURM_NODELIST:0:17}${SLURM_NODELIST:18:1}
master_node=${SLURM_NODELIST}
srun --label python3 -u knn_eval.py --root_dir='path_to_features' \
