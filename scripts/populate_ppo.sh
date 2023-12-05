#!/bin/bash

# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

model="$1"
cache_dir="/data/models/archangel"
exp_name="archangel_sft+ppo_${model}"
sft_model="archangel_sft_${model}/LATEST/policy.pt"

echo $exp_name
python train.py loss=ppo model="$model" datasets=[shp,hh,oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.load_from="$sft_model"
