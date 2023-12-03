#!/bin/bash

loss="$1"
cache_dir="/data/models/archangel"

for model in pythia1-4b pythia2-8b pythia6-9b pythia12-0b llama7b llama13b llama30b; do
    exp_name="archangel_${loss}_${model}"
    echo "$exp_name"
    python compare.py -f "samples/${exp_name}.json" -mc 512
    python compare.py -f "samples/${exp_name}.json" -mc 512 -bk chosen
done