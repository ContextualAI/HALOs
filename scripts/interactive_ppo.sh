#!/bin/bash

# example usage:
# bash scripts/interactive_ppo.sh 0.05 1e-6 0.2

KL_COEF=$1 # 0.05
LR=$2 # 1e-6
CLIP_RANGE=$3 # 0.2

# Function to find an available port
find_free_port() {
    local port
    while true; do
        port=$(shuf -i 29500-29510 -n 1)
        if ! netstat -tuln | grep -q ":$port "; then
            echo "$port"
            break
        fi
    done
}

# Function to initialize the environment
init_env() {
    module load anaconda3/2024.2
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate halos

    echo "Running on node: $(hostname)"
    echo "Machine Rank: $SLURM_PROCID"
    
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=$(find_free_port | tr -d '\n')
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export HF_ALLOW_CODE_EVAL=1
}

export -f find_free_port
export -f init_env

# Run the training script
init_env
export MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export HALO_DIR=/scratch/gpfs/sl2998/workspace/HALOs
export CACHE_DIR=/scratch/gpfs/sl2998/models/llama-3-8b-instruct-offline-ppo-kl-${KL_COEF}-lr-${LR}
export EXP_NAME=llama-3-8b-instruct-offline-ppo-kl-${KL_COEF}-lr-${LR}

if [ ! -d "${CACHE_DIR}" ]; then
    mkdir -p "${CACHE_DIR}"
fi

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    launch.py loss=ppo model=llama train_datasets=[ultrafeedback_armorm] test_datasets=[ultrabin] exp_name=${EXP_NAME} \
    ++cache_dir=${CACHE_DIR} \
    ++model.name_or_path=$MODEL_PATH \
    ++lr=${LR} \
    ++intermediate_checkpoints=false \
    ++online=false \
    ++eval_every=256 \
    ++weight_decay=0.0 \
    ++beta1=0.9 \
    ++beta2=0.95 \
    ++eps=1e-5 \
    ++warmup=0.10 \
    ++loss.ppo_epochs=1 \
    ++loss.cliprange=${CLIP_RANGE} \
    ++loss.lam=0.95 \
    ++loss.gamma=1.0 \
    ++loss.critic_coef=0.1 \
    ++loss.KL_coef=${KL_COEF} \
    ++n_epochs=1 \
    ++model.batch_size=8 \
    ++model.max_grad_norm=1 \
    ++model.max_length=2048 \
    ++model.max_prompt_length=1024 \
    ++model.gradient_accumulation_steps=8 \
    ++model.eval_batch_size=32 \
    $MODEL_LOAD_ARG 2>&1 | tee $HALO_DIR/logs/$EXP_NAME.log

