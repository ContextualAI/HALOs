#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --mem=800G

# Run as sbatch -D `pwd` --job-name=push-kto --output=./outputs/push-kto.out --error=./outputs/push-kto.err --export=loss=kto,model=llama7b ./scripts/push_sbatch.sh

# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

source ~/.bashrc 
source /opt/conda/etc/profile.d/conda.sh 
conda activate halos

cache_dir="/data/models/archangel"

nargs="$#"
cache_dir="/data/models/archangel"
losses=("sft" "sft+csft" "sft+dpo" "sft+kto" "sft+ppo" "sft+slic" "csft" "dpo" "kto" "ppo" "slic")
models=("pythia1-4b" "pythia2-8b" "pythia6-9b" "pythia12-0b" "llama7b" "llama13b" "llama30b")

if [[ $loss != "" ]]; then
    losses=("$loss")
fi

if [[ $model != "" ]]; then
    models=("$model")
fi

# loop through all the losses, and push the corresponding models
for model in "${models[@]}"; do
    if [ $nargs -gt 0 ]; then
        losses=("$1")
    fi
    
    for loss in "${losses[@]}"; do
        exp_name="archangel_${loss}_${model}"
        echo "$exp_name"
        python push.py -c "${cache_dir}/${exp_name}/config.yaml"
    done
done