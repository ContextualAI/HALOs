#!/bin/bash
#SBATCH --job-name=qwen-coefficient
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --partition=pli-c
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

WEIGHTD=$1

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
    export MASTER_PORT=29501
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    
    echo "Master node: $MASTER_ADDR"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
    echo "GPUs per node: $SLURM_GPUS_PER_NODE"
}

export -f init_env

# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
export MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
export CKPT=/scratch/gpfs/ke7953/models/qwen2-5-3B-instruct-kto-01-${WEIGHTD}D-5e-6/FINAL

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=kto model=qwen datasets=[ultrabin] exp_name=qwen2-5-3B-instruct-kto-01-${WEIGHTD}D-5e-6 \
    ++cache_dir=/scratch/gpfs/ke7953/models \
    ++model.name_or_path=$MODEL_PATH \
    ++lr=5e-6 \
    ++loss.beta=0.1 \
    ++model.batch_size=8 ++model.gradient_accumulation_steps=4 ++model.eval_batch_size=8 \
    ++loss.desirable_weight=${WEIGHTD}

lm_eval --model hf \
  --model_args pretrained=$CKPT,tokenizer=$CKPT,parallelize=True \
  --tasks arc_easy,arc_challenge,winogrande,bbh_cot_fewshot,gsm8k_cot \
  --batch_size 4 \
  --output_base_path /home/ke7953/HALOs/outputs

python -m eval.sample_for_alpacaeval $CKPT --gpu_count 2 --output_file outputs/qwen2-5-3b-instruct-kto-01-${WEIGHTD}D-5e-6.json
"


