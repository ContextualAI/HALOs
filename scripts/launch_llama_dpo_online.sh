#!/bin/bash

BETA=$1
LR=$2
TOTAL_PROMPTS=2048
PROMPTS_PER_ROUND=512  # Train on fewer examples before sampling
NUM_ROUNDS=$(($TOTAL_PROMPTS / $PROMPTS_PER_ROUND))  # Calculate this once at the start

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
    conda activate halos_6

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
export MODEL_PATH=meta-llama/Meta-Llama-3-8B
export SFT_CKPT=/scratch/gpfs/ke7953/models/llama3-8B-sft/FINAL
export REWARD_CKPT=/scratch/gpfs/ke7953/models/llama3-8B-reward/FINAL

# Start iterative training from SFT checkpoint
CURRENT_CKPT=\$SFT_CKPT
ROUND=1
CUMULATIVE_PROMPTS=0

while [ \$ROUND -le ${NUM_ROUNDS} ]; do
    echo \"Starting round \$ROUND of ${NUM_ROUNDS} \"

    # Sample from current model (first round uses SFT checkpoint)
    python -m train.sample \$CURRENT_CKPT \
        --output_file R\${ROUND}_samples.json \
        --gpu_count 4 \
        --datasets alpacaeval \
        --mode train \
        --num_samples_per_prompt 4 \
        --n_examples ${PROMPTS_PER_ROUND} \
        --num_skip \$CUMULATIVE_PROMPTS \
        --batch_size ${PROMPTS_PER_ROUND}

    # if no more prompts left, end training early
    NUM_SAMPLES=\$(jq '. | length' R\${ROUND}_samples.json)

    if [[ \$NUM_SAMPLES -gt 0 ]]; then
        CUMULATIVE_PROMPTS=\$((CUMULATIVE_PROMPTS + ${PROMPTS_PER_ROUND}))

        # Label samples with reward model
        accelerate launch \
            --config_file accelerate_config/fsdp_4gpu.yaml \
            --machine_rank \$SLURM_PROCID \
            --main_process_ip \$MASTER_ADDR \
            --main_process_port \$MASTER_PORT \
            label.py \$REWARD_CKPT R\${ROUND}_samples.json R\${ROUND}_data.json \
            --feedback_type pairwise --batch_size 16

        NUM_EXAMPLES=\$(jq '. | length' R\${ROUND}_data.json)

        if [ \$ROUND -eq 1 ]; then
            # First round: load from SFT checkpoint
            MODEL_LOAD_ARG=\"++model.load_from=\$CURRENT_CKPT\"
        else
            # Subsequent rounds: policy resumes from previous checkpoint; reference model is also updated via load_from
            MODEL_LOAD_ARG=\"++model.from_checkpoint=\$CURRENT_CKPT ++model.load_from=\$CURRENT_CKPT\"
        fi
        
        # Train DPO model on the newly sampled and labeled data
        accelerate launch \
            --config_file accelerate_config/fsdp_4gpu.yaml \
            --machine_rank \$SLURM_PROCID \
            --main_process_ip \$MASTER_ADDR \
            --main_process_port \$MASTER_PORT \
            launch.py loss=dpo model=llama exp_name=llama3-8B-sft-dpo-R\${ROUND} \
            train_datasets=[R\${ROUND}_data.json] test_datasets=[ultrafeedback_armorm] \
            ++cache_dir=/scratch/gpfs/ke7953/models \
            ++model.name_or_path=\$MODEL_PATH \
            ++lr=${LR} \
            ++loss.beta=${BETA} \
            ++model.batch_size=16 ++model.gradient_accumulation_steps=1 ++model.eval_batch_size=16 ++online=true \
            \$MODEL_LOAD_ARG

        # Get new checkpoint path
        NEW_CKPT=/scratch/gpfs/ke7953/models/llama3-8B-sft-dpo-R\${ROUND}/FINAL

        # Clean up old checkpoint directory if it's not the SFT checkpoint
        if [ \$CURRENT_CKPT != \$SFT_CKPT ] && [ \$SLURM_PROCID -eq 0 ]; then
            OLD_EXP_DIR=\$(dirname \$CURRENT_CKPT)
            echo \"Cleaning up \$OLD_EXP_DIR\"
            rm -rf \$OLD_EXP_DIR
        fi

        # Update checkpoint for next round
        CURRENT_CKPT=\$NEW_CKPT
        ROUND=\$((ROUND + 1))
    else
        echo \"Ending training early due to zero new samples ...\"
        break
    fi
done

# Final evaluation
lm_eval --model hf \
    --model_args pretrained=\$CURRENT_CKPT,tokenizer=\$CURRENT_CKPT,parallelize=True \
    --tasks arc_easy,arc_challenge,winogrande,bbh_cot_fewshot,gsm8k_cot \
    --batch_size 4
"