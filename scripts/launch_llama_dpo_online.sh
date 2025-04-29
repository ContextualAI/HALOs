#!/bin/bash
#SBATCH --job-name=test-job
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --partition=pli-c
#SBATCH --output=%x.out
#SBATCH --error=%x.err

BETA=$1
LR=$2
N_EPOCHS=$3 # 3
SPLIT=$4
START_ROUND=${5:-1}  # Default to 1 if not provided

TOTAL_PROMPTS=2048
PROMPTS_PER_ROUND=64  # Train on fewer examples before sampling
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

# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
export MODEL_PATH=allenai/tulu-2-13b
export SFT_CKPT=allenai/tulu-2-13b
# export REWARD_CKPT=/scratch/gpfs/sl2998/models/llama3-8B-reward/FINAL
export REWARD_CKPT=openbmb/UltraRM-13b
export CACHE_DIR=/scratch/gpfs/sl2998/models/test-tulu-2-13b-online-dpo-split-${SPLIT}-beta-${BETA}-lr-${LR}-eps-${N_EPOCHS}
export HALO_DIR=/home/sl2998/workspace/HALOs

mkdir -p \$CACHE_DIR

# Start iterative training from SFT checkpoint or continue from a previous round
if [ ${START_ROUND} -eq 1 ]; then
    CURRENT_CKPT=\$SFT_CKPT
    ROUND=1
    CUMULATIVE_PROMPTS=0
else
    PREV_ROUND=$((${START_ROUND} - 1))
    CURRENT_CKPT=\${CACHE_DIR}/test-tulu-2-13b-online-dpo-split-${SPLIT}-beta-${BETA}-lr-${LR}-eps-${N_EPOCHS}-R\${PREV_ROUND}/FINAL
    ROUND=${START_ROUND}
    CUMULATIVE_PROMPTS=$(((${START_ROUND} - 1) * ${PROMPTS_PER_ROUND}))
    echo \"Continuing from round \${ROUND} with checkpoint \${CURRENT_CKPT}\"
fi

while [ \$ROUND -le ${NUM_ROUNDS} ]; do
    echo \"Starting round \$ROUND of ${NUM_ROUNDS} \"
    SAMPLES_FILE=\${CACHE_DIR}/R\${ROUND}_samples.json
    DATA_FILE=\${CACHE_DIR}/R\${ROUND}_data.json
    EXP_NAME=test-tulu-2-13b-online-dpo-split-${SPLIT}-beta-${BETA}-lr-${LR}-eps-${N_EPOCHS}-R\${ROUND}

    # Check if samples file already exists
    if [ ! -f \$SAMPLES_FILE ]; then
        echo \"Sampling for round \$ROUND...\"
        # Sample from current model (first round uses SFT checkpoint)
        python -m train.sample \$CURRENT_CKPT \
            --output_file \$SAMPLES_FILE \
            --gpu_count 4 \
            --datasets tulu_v25_arena23 \
            --mode train \
            --num_samples_per_prompt 4 \
            --num_prompts ${PROMPTS_PER_ROUND} \
            --num_skip \$CUMULATIVE_PROMPTS \
            --batch_size ${PROMPTS_PER_ROUND}
    else
        echo \"Samples file \$SAMPLES_FILE already exists, skipping sampling step.\"
    fi

    # if no more prompts left, end training early
    NUM_SAMPLES=\$(jq '. | length' \$SAMPLES_FILE)

    if [[ \$NUM_SAMPLES -gt 0 ]]; then
        CUMULATIVE_PROMPTS=\$((CUMULATIVE_PROMPTS + ${PROMPTS_PER_ROUND}))
        
        # Check if data file already exists
        if [ ! -f \$DATA_FILE ]; then
            echo \"Labeling samples for round \$ROUND...\"
            # Label samples with API (must ssh into the login node for internet access)
            ssh della-gpu \"source ~/.bashrc && \
                conda activate halos && \
                cd /home/sl2998/workspace/HALOs && \
                accelerate launch -m train.label \$SAMPLES_FILE \$DATA_FILE \
                --reward_model_path \$REWARD_CKPT \
                --feedback_type binary --batch_size 16 \"
        else
            echo \"Data file \$DATA_FILE already exists, skipping labeling step.\"
        fi

        # First round: load from SFT checkpoint
        # Subsequent rounds: policy resumes from previous checkpoint; reference model stays at SFT checkpoint
        if [ \$ROUND -eq 1 ]; then
            MODEL_LOAD_ARG=\"++model.load_from=\$CURRENT_CKPT\"
        else
            MODEL_LOAD_ARG=\"++model.from_checkpoint=\$CURRENT_CKPT ++model.load_from=\$SFT_CKPT\"
        fi
        
        # Train DPO model on the newly sampled and labeled data
        accelerate launch \
            --config_file accelerate_config/fsdp_4gpu.yaml \
            --machine_rank \$SLURM_PROCID \
            --main_process_ip \$MASTER_ADDR \
            --main_process_port \$MASTER_PORT \
            launch.py loss=dpo model=llama exp_name=\$EXP_NAME \
            train_datasets=[\$DATA_FILE] test_datasets=[ultrabin] \
            ++cache_dir=\$CACHE_DIR \
            ++model.name_or_path=\$MODEL_PATH \
            ++lr=${LR} \
            ++loss.beta=${BETA} \
            ++model.batch_size=8 \
            ++model.gradient_accumulation_steps=4 \
            ++model.eval_batch_size=32 \
            ++online=true \
            \$MODEL_LOAD_ARG

        NEW_CKPT=\${CACHE_DIR}/\${EXP_NAME}/FINAL

        # Clean up old checkpoint directory if it's not the SFT checkpoint
        # if [ \$CURRENT_CKPT != \$SFT_CKPT ] && [ \$SLURM_PROCID -eq 0 ]; then
        #     OLD_EXP_DIR=\$(dirname \$CURRENT_CKPT)
        #     echo \"Cleaning up \$OLD_EXP_DIR\"
        #     rm -rf \$OLD_EXP_DIR
        # fi

        CURRENT_CKPT=\$NEW_CKPT
        ROUND=\$((ROUND + 1))
    else
        echo \"Ending training early due to zero new samples ...\"
        break
    fi
done

# Evaluate the final model
python -m train.sample \$CURRENT_CKPT --gpu_count 4 --output_file \${HALO_DIR}/evals/outputs/\$EXP_NAME.json

ssh della-pli \"source ~/.bashrc && \
            conda activate halos && \
            cd \$HALO_DIR/evals && \
            python -c \\\"from datasets import load_dataset; ds = load_dataset('openai/openai_humaneval'); ds = load_dataset('TIGER-Lab/MMLU-Pro'); ds = load_dataset('truthful_qa', 'multiple_choice'); ds = load_dataset('evalplus/humanevalplus'); ds = load_dataset('google-research-datasets/mbpp'); ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')\\\" \
            alpaca_eval evaluate --is_overwrite_leaderboard=True --model_outputs=\${HALO_DIR}/outputs/\$EXP_NAME.json \"

# For GPQA, this dataset is gated, so you will have to accept the terms of use at https://huggingface.co/datasets/Idavidrein/gpqa and login via huggingface-cli login using your HF Hub token before running this task.

lm_eval --model hf \
    --model_args pretrained=\$CURRENT_CKPT,tokenizer=\$CURRENT_CKPT,parallelize=True \
    --tasks arc_easy,arc_challenge,winogrande,bbh_cot_fewshot,gsm8k_cot,ifeval,mmlu,mmlu_pro,gpqa_diamond_zeroshot,gpqa_diamond_n_shot,gpqa_diamond_cot_zeroshot,gpqa_diamond_cot_n_shot,humaneval,humaneval_plus,mbpp,mbpp_plus,truthfulqa_mc1,toxigen \
    --confirm_run_unsafe_code \
    --batch_size 32 2>&1 | tee \$HALO_DIR/evals/outputs/\${EXP_NAME}_lm_eval.log

python -m evals.scripts.summarize_metrics outputs -o evals/metrics_summary

for TASK in hh safe_rlhf wildbench; do
    export RLHF_SAMPLES=\$HALO_DIR/evals/outputs/\${EXP_NAME}_\${TASK}.json
    export BASE_SAMPLES=\$HALO_DIR/evals/outputs/\${EXP_NAME}_\${TASK}_baseline.json
    export PAIR_REWARDS=\$HALO_DIR/evals/outputs/\${EXP_NAME}_\${TASK}_pair_rewards.json
    
    python -m train.sample \$CURRENT_CKPT --datasets \$TASK \
        --num_samples_per_prompt 1 \
        --mode test \
        --gpu_count 4 \
        --output_file \$RLHF_SAMPLES
    
    # Using SFT ckpt as baseline by default
    python -m train.sample \$SFT_CKPT --datasets \$TASK \
        --num_samples_per_prompt 1 \
        --mode test \
        --gpu_count 4 \
        --output_file \$BASE_SAMPLES
    
    python -m train.label \
        --second_samples_path \$BASE_SAMPLES \
        --api_type openai --api_key \$OPENAI \
        \$RLHF_SAMPLES \$PAIR_REWARDS --feedback_type pairwise
done
"