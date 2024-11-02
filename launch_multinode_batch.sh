#!/bin/bash
#SBATCH --job-name=multinode-test
#SBATCH --nodes=2
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --partition=pli-c
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Function to find an available port
find_free_port() {
    local port
    while true; do
        # Generate a random port number between 20000 and 65000
        port=$(shuf -i 29500-29510 -n 1)
        # Check if the port is in use
        if ! netstat -tuln | grep -q ":$port "; then
            echo "$port"
            break
        fi
    done
}

# Function to initialize the environment and print diagnostic information
# very important that this is run within srun for training to work!!!
init_env() {
    # Load necessary modules (adjust as needed for your system)
    module load anaconda3/2024.2

    # Activate your conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate halos

    echo "Running on node: $(hostname)"
    echo "Machine Rank: $SLURM_PROCID"
    
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=$(find_free_port | tr -d '\n')
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    
    echo "Master node: $MASTER_ADDR"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
    echo "GPUs per node: $SLURM_GPUS_PER_NODE"
}

export -f find_free_port
export -f init_env

# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '
init_env
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
'
