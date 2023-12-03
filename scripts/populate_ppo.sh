#!/bin/bash

model="$1"
cache_dir="/data/models/archangel"
exp_name="archangel_sft+ppo_${model}"
sft_model="archangel_sft_${model}/LATEST/policy.pt"

echo $exp_name
python train.py loss=ppo model="$model" datasets=[shp,hh,oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.load_from="$sft_model"
