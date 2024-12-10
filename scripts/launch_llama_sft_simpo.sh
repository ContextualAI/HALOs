#!/bin/bash
#SBATCH --job-name=llama-sft-simpo
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --partition=pli-c
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

LR=$1

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
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
export MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=simpo model=llama datasets=[ultrafeedback_armorm] exp_name=llama3-8B-instruct-simpo_100 \
    ++cache_dir=/scratch/gpfs/ke7953/models \
    ++model.name_or_path=\$MODEL_PATH \
    ++lr=1e-6 \
    ++model.batch_size=2 ++model.gradient_accumulation_steps=16 ++model.eval_batch_size=2 \
    ++model.max_length=2048 ++model.max_prompt_length=1800 \
    ++loss.beta=10 ++loss.gamma_beta_ratio=0.3 \
    ++model.max_grad_norm=100

python -m train.sample \$CKPT --gpu_count 2 --output_file outputs/llama3-8b-sft-simpo-${LR}.json

lm_eval --model hf \
  --model_args pretrained=\$CKPT,tokenizer=\$CKPT,parallelize=True \
  --tasks arc_easy,arc_challenge,winogrande,bbh_cot_fewshot,gsm8k_cot \
  --batch_size 4
"
