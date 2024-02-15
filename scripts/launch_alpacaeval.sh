#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --mem=800G
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=a3

source /opt/conda/etc/profile.d/conda.sh 
# source /home/niklas/miniconda/bin/activate
conda activate halos

config_args=$1
output_dir=$2
script=/home/winnie/halos/eval.py

# step 1: run the command
# echo python $script --config-path $config_args
python $script --config-path $config_args
# step 2: config_args here should be exp_name, output_dir is also ignored
# alpaca_eval --model_outputs "$output_dir/alpaca_${config_args}.json" --annotators_config 'alpaca_eval_gpt4'