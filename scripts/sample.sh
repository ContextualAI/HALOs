#!/bin/bash

loss="$1"
cache_dir="/data/models/archangel"

for model in pythia1-4b pythia2-8b pythia6-9b pythia12-0b llama7b llama13b; do
    exp_name="archangel_${loss}_${model}"
    echo "$exp_name"
    python eval.py -c "${cache_dir}/${exp_name}/config.yaml" -m sample -n 512 -b 32    
done

for model in llama30b; do
    exp_name="archangel_${loss}_${model}"
    echo "$exp_name"
    python eval.py -c "${cache_dir}/${exp_name}/config.yaml" -m sample -n 512 -b 16    
done