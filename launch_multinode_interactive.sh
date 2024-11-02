#!/bin/bash

# If you are running a job within a single node, you can just run accelerate launch.
# If you want to run a job across multiple nodes, run the following: 
# salloc --nodes=2 --ntasks-per-node=1 -c 8 --mem=100G --time=23:55:00 --gres=gpu:2 --partition=pli-c
# conda activate halos_3
# HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) MASTER_PORT=29501 srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 ./launch_interactive.sh

echo "Running on node: $(hostname)"
echo "Machine Rank: $SLURM_PROCID"

# Load necessary modules (adjust as needed for your system)
module load anaconda3/2024.2

# Activate your conda environment
conda init
conda activate halos

echo "Machine Rank: $SLURM_PROCID"

accelerate launch \
    --config_file accelerate_config/fsdp_2x2gpu.yaml \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    launch.py loss=kto model=llama datasets=[ultrafeedback_hybrid] exp_name=llama-test \
	++cache_dir=/scratch/gpfs/ke7953/models \
	++model.name_or_path=meta-llama/Meta-Llama-3-8B \
	++lr=1e-6 \
	++loss.beta=0.1
