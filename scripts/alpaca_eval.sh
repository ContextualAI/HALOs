#!/bin/bash

# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Sample use is './scripts/alpaca_eval.sh dpo llama7b'. 
# If the model isn't specified, then it will run on all models. If the loss isn't specified, it will run all losses on all models.
# This should be run after already generating the samples for AlpacaEval.

cache_dir="/data/models/archangel_kto"
losses=("sft" "sft+csft" "sft+dpo" "sft+kto" "sft+ppo" "sft+slic" "csft" "dpo" "kto" "ppo" "slic")
models=("pythia1-4b" "pythia2-8b" "pythia6-9b" "pythia12-0b" "llama7b" "llama13b" "llama30b")

if [ $nargs == 1 ]; then
    losses=("$1")
elif [ $nargs == 2 ]; then
    losses=("$1")
    models=("$2")
fi

# loop through all the losses, and for each corresponding model, call the alpaca_eval.py script
for model in "${models[@]}"; do
    for loss in "${losses[@]}"; do
        alpaca_eval --model_outputs "$samples/alpaca_archangel_${loss}_${model}.json" --annotators_config 'alpaca_eval_gpt4'
        # --reference_outputs <path to outputs if not text_davinci_003 on AlpacaEval>
    done
done
