#!/bin/bash

# Sample use is './scripts/compare.sh dpo llama7b'. 
# If the model isn't specified, then it will run on all models. If the loss isn't specified, it will run all losses on all models.

cache_dir="/data/models/archangel"
nargs="$#"
losses=("unaligned" "sft" "dpo" "kto" "ppo" "slic" "csft" "sft+dpo" "sft+kto" "sft+ppo" "sft+slic" "sft+csft")
models=("pythia1-4b" "pythia2-8b" "pythia6-9b" "pythia12-0b" "llama7b" "llama13b" "llama30b")

if [ $nargs == 1 ]; then
    losses=("$1")
elif [ $nargs == 2 ]; then
    losses=("$1")
    models=("$2")
fi

for loss in "${losses[@]}"; do
    for model in "${models[@]}"; do
        exp_name="archangel_${loss}_${model}"
        echo "$exp_name"
        python compare.py -f "samples/${exp_name}.json" -mc 512 -r results.jsonl
    done
done