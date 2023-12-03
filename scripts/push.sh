#!/bin/bash

nargs="$#"
cache_dir="/data/models/archangel"
losses=("sft+dpo" "sft+kto", "sft+ppo")  # ("ppo" "dpo" "kto" "sft" "slic" "sft+ppo")
models=("pythia1-4b" "pythia2-8b" "pythia6-9b" "pythia12-0b" "llama7b" "llama13b" "llama30b")

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