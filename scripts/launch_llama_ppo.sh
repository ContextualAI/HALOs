#!/bin/bash
#SBATCH --job-name=llama-ppo
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --partition=pli-c
#SBATCH --constraint="rh9"
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# example usage: 
# sbatch launch_llama_ppo.sh 0.05 1e-6 1 tulu_v25_arena23

KL_COEF=$1 # 0.05, 0.0325
LR=$2 # 1e-6
N_EPOCHS=$3 # 1
SPLIT=$4 # tulu_v25_arena23

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
    export HF_ALLOW_CODE_EVAL=1
    export HYDRA_FULL_ERROR=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    echo "Master node: $MASTER_ADDR"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
    echo "GPUs per node: $SLURM_GPUS_PER_NODE"
}

export -f find_free_port
export -f init_env

# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
export MODEL_PATH=allenai/tulu-2-13b
export HALO_DIR=/home/sl2998/workspace/HALOs
export CKPT=/scratch/gpfs/sl2998/models/tulu-2-13b-ppo-split-${SPLIT}-kl-${KL_COEF}-linear-lr-${LR}-eps-${N_EPOCHS}
export EXP_NAME=tulu-2-13b-ppo-split-${SPLIT}-kl-${KL_COEF}-linear-lr-${LR}-eps-${N_EPOCHS}


if [ -n \"\$(ls -alt \$CKPT/ 2>/dev/null | grep step)\" ]; then
    # Get the 2nd latest checkpoint (the latest checkpoint is often not fully written yet)
    export LATEST_CKPT=\$(ls -alt \$CKPT/ | grep step | head -n 2 | tail -n 1 | awk '{print \$NF}')
    MODEL_LOAD_ARG=\"++model.from_checkpoint=\$CKPT/\${LATEST_CKPT}\"
    echo \"Latest checkpoint: \$MODEL_LOAD_ARG\"
else
    MODEL_LOAD_ARG=\"++model.from_checkpoint=null\"
    echo \"No checkpoint found\"
fi

accelerate launch \
    --config_file accelerate_config/fsdp_8gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=ppo model=llama train_datasets=[${SPLIT}] test_datasets=[ultrafeedback_armorm] exp_name=\${EXP_NAME} \
    ++cache_dir=/scratch/gpfs/sl2998/models \
    ++model.name_or_path=\$MODEL_PATH \
    ++lr=${LR} \
    ++intermediate_checkpoints=true \
    ++online=false \
    ++eval_every=256 \
    ++save_every=5120 \
    ++weight_decay=0.0 \
    ++beta1=0.9 \
    ++beta2=0.95 \
    ++eps=1e-5 \
    ++warmup=0.10 \
    ++loss.ppo_epochs=1 \
    ++loss.cliprange=0.2 \
    ++loss.lam=0.95 \
    ++loss.gamma=1.0 \
    ++loss.critic_coef=0.1 \
    ++loss.KL_coef=${KL_COEF} \
    ++n_epochs=${N_EPOCHS} \
    ++model.from_checkpoint=\$CKPT/\${LATEST_CKPT} \
    ++model.batch_size=8 \
    ++model.max_grad_norm=1 \
    ++model.max_length=4096 \
    ++model.max_prompt_length=2048 \
    ++model.gradient_accumulation_steps=8 \
    ++model.eval_batch_size=32 \
    \$MODEL_LOAD_ARG 2>&1 | tee \${HALO_DIR}/logs/\${EXP_NAME}.log

# Evaluate the final model
ssh della-pli \"source ~/.bashrc && \
            conda activate halos && \
            cd \$HALO_DIR/evals && \
            python -c \\\"from datasets import load_dataset; ds = load_dataset('openai/openai_humaneval'); ds = load_dataset('TIGER-Lab/MMLU-Pro'); ds = load_dataset('truthful_qa', 'multiple_choice'); ds = load_dataset('evalplus/humanevalplus'); ds = load_dataset('google-research-datasets/mbpp'); ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')\\\"
            \"

# For GPQA, this dataset is gated, so you will have to accept the terms of use at https://huggingface.co/datasets/Idavidrein/gpqa and login via huggingface-cli login using your HF Hub token before running this task.

lm_eval --model hf \
    --model_args pretrained=\$CKPT/FINAL,tokenizer=\$CKPT/FINAL,parallelize=True \
    --tasks mmlu,gsm8k_cot,bbh_cot_fewshot,humaneval_plus,mbpp_plus,truthfulqa_mc1,toxigen,ifeval,arc_easy,arc_challenge,winogrande \
    --confirm_run_unsafe_code \
    --batch_size 32 2>&1 | tee \$HALO_DIR/evals/outputs/\${EXP_NAME}_lm_eval.log

"