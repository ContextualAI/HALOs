#!/bin/bash
#SBATCH --job-name=llama-odpo-v1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=23:55:00
#SBATCH --partition=pli-c
#SBATCH --exclude=della-j14g1
#SBATCH --constraint=rh9|rh8
#SBATCH --output=logs/llama-odpo-v1-%j.out
#SBATCH --error=logs/llama-odpo-v1-%j.err


BETA=$1
LR=$2
TOTAL_PROMPTS=11264
PROMPTS_PER_ROUND=1024

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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
}

export -f find_free_port
export -f init_env

# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
pip show vllm

export MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export HALO_DIR=/home/sl2998/workspace/HALOs
export CACHE_DIR=/scratch/gpfs/sl2998/models/llama-instruct-dpo-online-prompt-v1

if [ ! -d \$CACHE_DIR ]; then
    mkdir -p \$CACHE_DIR
fi

CURRENT_CKPT=\$MODEL_PATH
CUMULATIVE_PROMPTS=0

while [ \$CUMULATIVE_PROMPTS -lt ${TOTAL_PROMPTS} ]; do
    END_CUMULATIVE_PROMPTS=\$((CUMULATIVE_PROMPTS + ${PROMPTS_PER_ROUND}))
    SAMPLES_FILE=\${CACHE_DIR}/\${CUMULATIVE_PROMPTS}_\${END_CUMULATIVE_PROMPTS}_samples.json
    DATA_FILE=\${CACHE_DIR}/\${CUMULATIVE_PROMPTS}_\${END_CUMULATIVE_PROMPTS}_data.json

    # if creating a pairwise dataset, you need to oversample, since ~half of pairs will be discarded (e.g., both outputs have the same score)
    python -m train.sample \$CURRENT_CKPT \
        --output_file \$SAMPLES_FILE \
        --gpu_count 4 \
        --datasets ultrafeedback_armorm \
        --mode train \
        --split train \
        --num_samples_per_prompt 4 \
        --num_prompts ${PROMPTS_PER_ROUND} \
        --num_skip \$CUMULATIVE_PROMPTS \
        --num_epochs 1
    PYTHON_EXIT_CODE=\$?

    if [ \$PYTHON_EXIT_CODE -eq 0 ]; then
        echo \"Training on \$CUMULATIVE_PROMPTS through \$END_CUMULATIVE_PROMPTS prompts ... \"
        
        EXP_NAME=llama3-8B-instruct-dpo-prompt-v1-\${CUMULATIVE_PROMPTS}

        # Label samples with API (must ssh into the login node for internet access)
        if [ ! -f \$DATA_FILE ]; then
            echo \"Labeling samples with API...\"
            ssh della-gpu \"source ~/.bashrc && \
                conda activate halos && \
                cd /home/sl2998/workspace/HALOs && \
                accelerate launch -m train.label_v1 \$SAMPLES_FILE \$DATA_FILE \
                --api_type openai --api_key \$OPENAI_API_KEY --feedback_type pairwise --batch_size 16 \"
        else
            echo \"\$DATA_FILE already exists, skipping labeling...\"
        fi

        if [ \$CUMULATIVE_PROMPTS -eq 0 ]; then
            MODEL_LOAD_ARG=\"++model.load_from=\$CURRENT_CKPT ++model.from_checkpoint=null \"
        else
            MODEL_LOAD_ARG=\"++model.load_from=\$CURRENT_CKPT ++model.from_checkpoint=\$CURRENT_CKPT \"
        fi
        
        # Train DPO model on the newly sampled and labeled data
        accelerate launch \
            --config_file accelerate_config/fsdp_4gpu.yaml \
            --machine_rank \$SLURM_PROCID \
            --main_process_ip \$MASTER_ADDR \
            --main_process_port \$MASTER_PORT \
            launch.py loss=dpo ++loss.beta=${BETA} model=llama exp_name=\$EXP_NAME \
            train_datasets=[\$DATA_FILE] test_datasets=[ultrafeedback_armorm] ++lr=${LR} \
            ++cache_dir=\$CACHE_DIR \
            ++model.name_or_path=\$MODEL_PATH \$MODEL_LOAD_ARG \
            ++model.batch_size=4 ++model.gradient_accumulation_steps=8 ++model.eval_batch_size=4 ++model.max_grad_norm=0.1 \
            ++online=true ++sync_reference=true

        NEW_CKPT=\${CACHE_DIR}/\${EXP_NAME}/FINAL

        if [ \$CURRENT_CKPT != \$MODEL_PATH ] && [ \$SLURM_PROCID -eq 0 ]; then
            rm -rf \$CURRENT_CKPT
        fi

        CURRENT_CKPT=\$NEW_CKPT
        CUMULATIVE_PROMPTS=\$END_CUMULATIVE_PROMPTS
    else
        echo \"Ending training early due to zero new samples ...\"
        break
    fi
done

"