#!/bin/bash
#SBATCH --job-name=llama-instruct-grpo
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --partition=pli-c
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --exclude=della-j14g1
#SBATCH --constraint=rh9|rh8

BETA=$1
LR=$2
EPS=$3
EPOCHS=$4
GRADACC=$5

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
export EXP_NAME=llama3-8B-instruct-grpo-${BETA}-${EPS}-${LR}-${EPOCHS}-${GRADACC}
export CKPT=/scratch/gpfs/ke7953/models/\$EXP_NAME/FINAL

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=grpo model=llama train_datasets=[ultrafeedback_armorm] test_datasets=[ultrafeedback_armorm] exp_name=\$EXP_NAME \
    ++cache_dir=/scratch/gpfs/ke7953/models \
    ++model.name_or_path=\$MODEL_PATH \
    ++lr=${LR} \
    ++loss.beta=${BETA} ++loss.epsilon=${EPS} \
    ++humanline=false ++n_epochs=${EPOCHS} ++sync_reference=true ++model.max_grad_norm=0.1 \
    ++model.batch_size=32 ++model.gradient_accumulation_steps=${GRADACC} ++model.eval_batch_size=32

# lm_eval --model hf \
#   --model_args pretrained=\$CKPT,tokenizer=\$CKPT,parallelize=True \
#   --tasks arc_easy,arc_challenge,winogrande,bbh_cot_fewshot,gsm8k_cot \
#   --batch_size 4

python -m train.sample \$CKPT --gpu_count 2 --output_file outputs/\$EXP_NAME.json

ssh della-gpu \"source ~/.bashrc && \
            conda activate halos && \
            cd /home/ke7953/HALOs && \
            alpaca_eval evaluate --annotators_config /home/ke7953/HALOs/alpaca_eval_gpt-4.1.yaml --model_outputs=outputs/\$EXP_NAME.json \"
"