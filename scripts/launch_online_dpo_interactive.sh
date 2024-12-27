#!/bin/bash

BETA=$1
LR=$2

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
    conda activate halos_6

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
export MODEL_PATH=meta-llama/Meta-Llama-3-8B
export SFT_CKPT=/scratch/gpfs/ke7953/models/llama3-8B-sft/FINAL
export R1_CKPT=/scratch/gpfs/ke7953/models/llama3-8B-sft-dpo-R1/FINAL
export R2_CKPT=/scratch/gpfs/ke7953/models/llama3-8B-sft-dpo-R2/FINAL
export REWARD_CKPT=/scratch/gpfs/ke7953/models/llama3-8B-reward/FINAL

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=bradley-terry model=llama datasets=[ultrabin] exp_name=llama3-8B-reward \
    ++cache_dir=/scratch/gpfs/ke7953/models \
    ++model.name_or_path=\$MODEL_PATH \
    ++lr=${LR} \
    ++loss.beta=${BETA} \
    ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++model.eval_batch_size=16 \
    ++n_examples=512

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=dpo model=llama datasets=[ultrabin] exp_name=llama3-8B-sft-dpo-R1 \
    ++cache_dir=/scratch/gpfs/ke7953/models \
    ++model.name_or_path=\$MODEL_PATH \
    ++lr=${LR} \
    ++loss.beta=${BETA} \
    ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++model.eval_batch_size=16 \
    ++model.load_from=\$SFT_CKPT ++n_examples=512

python -m train.sample \$R1_CKPT --output_file R1_samples.json --gpu_count 4 --datasets alpacaeval --num_samples_per_prompt 2 --mode train 

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    label.py \$REWARD_CKPT R1_samples.json R2_data.json --feedback_type pairwise --batch_size 16

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=dpo model=llama datasets=[R2_data.json] exp_name=llama3-8B-sft-dpo-R2 \
    ++cache_dir=/scratch/gpfs/ke7953/models \
    ++model.name_or_path=\$MODEL_PATH \
    ++lr=${LR} \
    ++loss.beta=${BETA} \
    ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++model.eval_batch_size=16 \
    ++model.load_from=\$R1_CKPT ++n_examples=512

python -m train.sample \$R2_CKPT --output_file R2_samples.json --gpu_count 4 --datasets alpacaeval --num_samples_per_prompt 2 --mode train 

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    label.py \$REWARD_CKPT R2_samples.json R3_data.json --feedback_type pairwise --batch_size 16

lm_eval --model hf \
  --model_args pretrained=\$R2_CKPT,tokenizer=\$R2_CKPT,parallelize=True \
  --tasks arc_easy,arc_challenge,winogrande,bbh_cot_fewshot,gsm8k_cot \
  --batch_size 8
"