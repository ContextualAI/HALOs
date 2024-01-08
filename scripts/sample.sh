#!/bin/bash

# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Sample use is './scripts/sample.sh dpo llama7b'. 
# If the model isn't specified, then it will run on all models. If the loss isn't specified, it will run all losses on all models.

cache_dir="/data/models/archangel"
nargs="$#"
losses=("unaligned" "sft" "sft+csft" "sft+dpo" "sft+kto" "sft+ppo" "sft+slic" "csft" "dpo" "kto" "ppo" "slic")
models=("pythia1-4b" "pythia2-8b" "pythia6-9b" "pythia12-0b" "llama7b" "llama13b" "llama30b")

if [ $nargs == 1 ]; then
    losses=("$1")
elif [ $nargs == 2 ]; then
    losses=("$1")
    models=("$2")
fi

for loss in "${losses[@]}"; do
    for model in "${models[@]}"; do
        if [[ $model == "llama30b" ]]; then
            batch_size=16
        else
            batch_size=32
        fi

        exp_name="archangel_${loss}_${model}"
        echo $exp_name

        if [[ $loss == "unaligned" ]]; then
            python eval.py --config-path="${cache_dir}/archangel_sft_${model}" ++mode=sample ++n_samples=512 ++model.eval_batch_size=$batch_size ++samples_dir=samples/ ++exp_name=$exp_name ++saved_policy=null
        else
            python eval.py --config-path="${cache_dir}/${exp_name}" ++mode=sample ++n_samples=512 ++model.eval_batch_size=$batch_size ++samples_dir=samples/
        fi
    done
done