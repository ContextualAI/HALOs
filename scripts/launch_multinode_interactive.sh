#!/bin/bash

# If you want to run a job across multiple nodes, run the following to get 4 gpus for each of 2 nodes (total of 8):
# salloc --nodes=2 -c 8 --mem=40G --time=23:55:00 --gres=gpu:4 --partition=pli-c
# Then run . launch_multinode_interactive.sh

# Function to initialize the environment and print diagnostic information
# very important that this is run within srun for training to work!!!
init_env() {
    # Load necessary modules (adjust as needed for your system)
    module load anaconda3/2024.2

    # Activate your conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate halos_6

    echo "Running on node: $(hostname)"
    echo "Machine Rank: $SLURM_PROCID"
    
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=29500
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    
    echo "Master node: $MASTER_ADDR"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
    echo "GPUs per node: $SLURM_GPUS_PER_NODE"
}

export -f init_env

# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
export MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct

accelerate launch \
    --config_file accelerate_config/fsdp_2x2gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=simpo model=llama datasets=[ultrabin] exp_name=llama3-8B-instruct-simpo_100 \
    ++cache_dir=/scratch/gpfs/ke7953/models \
    ++model.name_or_path=\$MODEL_PATH \
    ++lr=1e-6 \
    ++model.batch_size=2 ++model.gradient_accumulation_steps=16 ++model.eval_batch_size=2 \
    ++loss.beta=10 ++loss.gamma_beta_ratio=0.3 \
    ++model.max_grad_norm=100
"