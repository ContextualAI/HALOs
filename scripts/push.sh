#!/bin/bash

# Sample use is './scripts/push.sh dpo llama7b'. 
# If the model isn't specified, then it will run on all models. If the loss isn't specified, it will run all losses on all models.

nargs="$#"
cache_dir="/data/models/archangel"
losses=("sft" "sft+csft" "sft+dpo" "sft+kto" "sft+ppo" "sft+slic" "csft" "dpo" "kto" "ppo" "slic")
models=("pythia1-4b" "pythia2-8b" "pythia6-9b" "pythia12-0b" "llama7b" "llama13b" "llama30b")

if [ $nargs == 1 ]; then
    losses=("$1")
elif [ $nargs == 2 ]; then
    losses=("$1")
    models=("$2")
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