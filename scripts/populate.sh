#!/bin/bash

# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Sample use is './scripts/populate.sh dpo llama7b'. 
# If the model isn't specified, then it will run on all models. If the loss isn't specified, it will run all losses on all models.

loss="$1"
cache_dir="/data/models/archangel"
nargs="$#"
losses=("sft" "sft+csft" "sft+dpo" "sft+kto" "sft+ppo" "sft+slic" "csft" "dpo" "kto" "ppo" "slic")
models=("pythia1-4b" "pythia2-8b" "pythia6-9b" "pythia12-0b" "llama7b" "llama13b" "llama30b")

if [ $nargs == 1 ]; then
    losses=("$1")
elif [ $nargs == 2 ]; then
    losses=("$1")
    models=("$2")
fi

for loss in "${losses[@]}"; do
    for model in "${models[@]}"; do
        exp_name="test_${loss}_${model}"
        echo "$exp_name"

        # llama30 has to use smaller batch sizes + gradient accumulation to get same effective batch size
        if [[ "$model" == "llama30b" ]] && ! ([[ $alignment_loss == "sft" ]] || [[ $alignment_loss == "ppo" ]]); then
            # PPO needs full 32 batch for stability reasons
            bs=16
            gas=2
        else
            bs=32
            gas=1
        fi

        if [[ $loss == "sft+"* ]]; then
            sft_model="archangel_sft_${model}/LATEST/policy.pt"
            alignment_loss="${loss:4}"
            python train.py loss="$alignment_loss" model="$model" datasets=[shp,hh,oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.load_from="$sft_model" ++model.batch_size="$bs" ++model.gradient_accumulation_steps="$gas"
        else
            python train.py loss="$loss" model="$model" datasets=[shp,hh,oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.batch_size="$bs" ++model.gradient_accumulation_steps="$gas"
        fi
    done
done