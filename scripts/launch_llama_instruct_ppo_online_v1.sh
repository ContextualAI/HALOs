#!/bin/bash
#SBATCH --job-name=llama-oppo-binary-v1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --partition=pli-c
#SBATCH --exclude=della-j14g1
#SBATCH --constraint=rh9|rh8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# example usage: 
# sbatch launch_llama_ppo_online_binary.sh 0.05 1e-6 0.2

KL_COEF=$1 # 0.05
LR=$2 # 1e-6
CLIP_RANGE=$3 # 0.2

TOTAL_PROMPTS=11264
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
pip show vllm

export MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export HALO_DIR=/scratch/gpfs/sl2998/workspace/HALOs
export REWARD_CKPT=RLHFlow/ArmoRM-Llama3-8B-v0.1
export CACHE_DIR=/scratch/gpfs/sl2998/models/llama-3-8b-instruct-online-ppo-kl-${KL_COEF}-lr-${LR}-clip-${CLIP_RANGE}-binary-v1

if [ ! -d \"\$CACHE_DIR\" ]; then
    mkdir -p \$CACHE_DIR
fi

# Start iterative training from SFT checkpoint or last saved checkpoint
CURRENT_CKPT=\$MODEL_PATH
ROUND=1
CUMULATIVE_PROMPTS=0

# Check if there are previous checkpoints to resume from
for r in \$(seq ${NUM_ROUNDS} -1 1); do
    LAST_EXP_NAME=llama-3-8b-instruct-online-ppo-kl-${KL_COEF}-lr-${LR}-clip-${CLIP_RANGE}-binary-v1-R\${r}
    LAST_CKPT=\${CACHE_DIR}/\${LAST_EXP_NAME}/FINAL
    if [ -d \"\$LAST_CKPT\" ]; then
        CURRENT_CKPT=\$LAST_CKPT
        ROUND=\$((r + 1))
        CUMULATIVE_PROMPTS=\$((r * PROMPTS_PER_ROUND))
        echo \"Resuming from round \$r checkpoint: \$CURRENT_CKPT\"
        echo \"Starting at round \$ROUND with \$CUMULATIVE_PROMPTS prompts already processed\"
        break
    fi
done

while [ \$ROUND -le ${NUM_ROUNDS} ]; do
    echo \"Starting round \$ROUND of ${NUM_ROUNDS} \"
    SAMPLES_FILE=\${CACHE_DIR}/R\${ROUND}_samples.json

    echo \"Sampling from current model...\"
    # Sample from current model (first round uses SFT checkpoint)
    python -m train.sample \$CURRENT_CKPT \
        --output_file \$SAMPLES_FILE \
        --gpu_count 4 \
        --datasets ultrafeedback_armorm \
        --split train \
        --mode train \
        --num_samples_per_prompt 4 \
        --num_prompts ${PROMPTS_PER_ROUND} \
        --num_skip \$CUMULATIVE_PROMPTS \
        --batch_size ${PROMPTS_PER_ROUND}
    echo \"Sampling complete. Labeling samples...\"

    # if no more prompts left, end training early
    NUM_SAMPLES=\$(jq '. | length' \$SAMPLES_FILE)

    if [[ \$NUM_SAMPLES -gt 0 ]]; then
        EXP_NAME=llama-3-8b-instruct-online-ppo-kl-${KL_COEF}-lr-${LR}-clip-${CLIP_RANGE}-binary-v1-R\${ROUND}
        CUMULATIVE_PROMPTS=\$((CUMULATIVE_PROMPTS + ${PROMPTS_PER_ROUND}))
        DATA_FILE=\${CACHE_DIR}/R\${ROUND}_data.json

        # Label samples with API (must ssh into the login node for internet access)
        ssh della-gpu \"source ~/.bashrc && \
            conda activate halos && \
            cd /home/sl2998/workspace/HALOs && \
            accelerate launch -m train.label_v1 \$SAMPLES_FILE \$DATA_FILE \
            --api_type openai --api_key \$OPENAI_API_KEY --feedback_type pairwise --batch_size 16 \"

        # accelerate launch --config_file accelerate_config/fsdp_4gpu.yaml \
        #     -m train.label \$SAMPLES_FILE \$DATA_FILE \
        #     --reward_model_path \$REWARD_CKPT \
        #     --feedback_type binary --batch_size 16

        # First round: load from checkpoint
        # Subsequent rounds: policy resumes from previous checkpoint; reference model stays at SFT checkpoint
        if [ \$ROUND -eq 1 ]; then
            MODEL_LOAD_ARG=\"++model.load_from=\$CURRENT_CKPT\"
        else
            echo \"Loading from checkpoint: \$CURRENT_CKPT\"
            MODEL_LOAD_ARG=\"++model.from_checkpoint=\$CURRENT_CKPT ++model.load_from=\$MODEL_PATH\"
        fi
        
        # Train PPO model on the newly sampled and labeled data
        accelerate launch \
            --config_file accelerate_config/fsdp_4gpu.yaml \
            --machine_rank \$SLURM_PROCID \
            --main_process_ip \$MASTER_ADDR \
            --main_process_port \$MASTER_PORT \
            launch.py loss=ppo model=llama train_datasets=[\$DATA_FILE] test_datasets=[ultrabin] exp_name=\$EXP_NAME \
            ++cache_dir=\$CACHE_DIR \
            ++model.name_or_path=\$MODEL_PATH \
            ++lr=${LR} \
            ++intermediate_checkpoints=true \
            ++online=true \
            ++eval_every=256 \
            ++save_every=5120 \
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
            ++model.batch_size=4 \
            ++model.max_grad_norm=1 \
            ++model.gradient_accumulation_steps=8 \
            ++model.eval_batch_size=16 \
            \$MODEL_LOAD_ARG 2>&1 | tee \$HALO_DIR/logs/\${EXP_NAME}.log

        NEW_CKPT=\${CACHE_DIR}/\${EXP_NAME}/FINAL

        # Clean up old checkpoint directory if it's not the SFT checkpoint
        if [ \$CURRENT_CKPT != \$MODEL_PATH ] && [ \$SLURM_PROCID -eq 0 ]; then
            OLD_EXP_DIR=\$(dirname \$CURRENT_CKPT)
            echo \"Cleaning up \$OLD_EXP_DIR\"
            rm -rf \$OLD_EXP_DIR
        fi

        CURRENT_CKPT=\$NEW_CKPT
        ROUND=\$((ROUND + 1))
    else
        echo \"Ending training early due to zero new samples ...\"
        break
    fi
done
"
