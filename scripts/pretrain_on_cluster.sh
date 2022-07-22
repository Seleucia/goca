#!/bin/bash

# Parameters
#SBATCH --job-name=kinetics-training
#SBATCH --nodes=8
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job_logs/slrm_stdout.%j
#SBATCH --error=logs/job_logs/slrm_stderr.%j
#SBATCH --gres=gpu:8
#SBATCH --mem-per-gpu=42G
#MASTER_ADDR
#MASTER_PORT=40000
#master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
#dist_url="tcp://"
#dist_url+=$master_node

#dist_url+=:40000

master_node=${SLURM_NODELIST:0:17}${SLURM_NODELIST:18:1}
#master_node=${SLURM_NODELIST}
srun --label python3 -u ssv_main.py --master_node $master_node --master_port 4000 --bash='true' \
--hidden_mlp=2048 --emedding_dim=128 --nmb_prototypes=1000 \
--vid_base_arch='s3d' --pool_type='s3d_old' --resume_training='False' --resume_model_path='' --use_fp16='false' \
--stride=2  --num_frames_lst='32,16' --nmb_temporal_samples='2,4'  --evaluation_type=0 --train_clip_consecutive_start_distance='16,-1|16,-1' \
--batch_size=10  --batch_size_val=10 --epochs=500 --curr_final_epochs=500 --queue_length=0 \
--share_proj_head='true' --video_modal_merge_start=-1 --video_modal_merge_algo=2 --video_modal_merge=2 --video_channels='rgb' --ds_name='ucf101' --split_idx=1 \
--use_protreg='False' --use_precomp_prot='False' --nmb_prototypes=1000 --freeze_prototypes_epoch=-1 --epsilon2=0.03 --epsilon=0.02 \
--warmup_epochs=10 --start_warmup=0.3  --final_lr=0.0006 --use_vicc_opt='false' --fix_lr='false' --pre_protype_normalize='true'