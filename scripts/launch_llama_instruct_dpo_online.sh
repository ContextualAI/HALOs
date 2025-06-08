#!/bin/bash

BETA=$1
LR=$2
TOTAL_PROMPTS=$3
PROMPTS_PER_ROUND=$4

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
}

export -f find_free_port
export -f init_env

# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
export MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export CACHE_DIR=/scratch/gpfs/ke7953/models/llama-instruct-dpo-online

if [ ! -d \$CACHE_DIR ]; then
    mkdir -p \$CACHE_DIR
fi

CUMULATIVE_PROMPTS=0

if [ \$CUMULATIVE_PROMPTS -eq 0 ]; then
    CURRENT_CKPT=\$MODEL_PATH
else
    CKPT_PROMPTS=\$((CUMULATIVE_PROMPTS - ${PROMPTS_PER_ROUND}))
    CURRENT_CKPT=\${CACHE_DIR}/llama3-8B-instruct-dpo-\${CKPT_PROMPTS}/FINAL
fi

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
        --num_samples_per_prompt 8 \
        --num_prompts ${PROMPTS_PER_ROUND} \
        --num_skip \$CUMULATIVE_PROMPTS \
        --num_epochs 1
    PYTHON_EXIT_CODE=\$?

    if [ \$PYTHON_EXIT_CODE -eq 0 ]; then
        echo \"Training on \$CUMULATIVE_PROMPTS through \$END_CUMULATIVE_PROMPTS prompts ... \"
        
        EXP_NAME=llama3-8B-instruct-dpo-\${CUMULATIVE_PROMPTS}

        accelerate launch \
            --config_file accelerate_config/fsdp_4gpu.yaml \
            --machine_rank \$SLURM_PROCID \
            --main_process_ip \$MASTER_ADDR \
            --main_process_port \$MASTER_PORT \
            -m train.label \$SAMPLES_FILE \$DATA_FILE --reward_model_path RLHFlow/ArmoRM-Llama3-8B-v0.1 \
                --feedback_type pairwise --batch_size 16 --threshold 0.01

        # # Label samples with API (must ssh into the login node for internet access)
        # ssh della-gpu \"source ~/.bashrc && \
        #     conda activate halos && \
        #     cd /home/ke7953/HALOs && \
        #     accelerate launch -m train.label \$SAMPLES_FILE \$DATA_FILE \
        #     --api_type openai --api_key \$OPENAI_API_KEY --feedback_type pairwise --batch_size 16 --threshold 0 \"

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
            ++model.name_or_path=\$MODEL_PATH \$MODEL_LOAD_ARG ++online=true \
            ++model.batch_size=32 ++model.gradient_accumulation_steps=1 ++model.eval_batch_size=32 ++model.max_grad_norm=0.5

        NEW_CKPT=\${CACHE_DIR}/\${EXP_NAME}/FINAL

        CURRENT_CKPT=\$NEW_CKPT
        CUMULATIVE_PROMPTS=\$END_CUMULATIVE_PROMPTS
    else
        echo \"Ending training early due to zero new samples ...\"
        break
    fi
done

python -m train.sample \$CURRENT_CKPT --gpu_count 2 --output_file outputs/\$EXP_NAME.json

ssh della-gpu \"source ~/.bashrc && \
            conda activate halos && \
            cd /home/ke7953/HALOs && \
            alpaca_eval evaluate --annotators_config /home/ke7953/HALOs/alpaca_eval_gpt-4.1.yaml --model_outputs=outputs/\$EXP_NAME.json \"
"