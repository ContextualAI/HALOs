#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --mem=800G

# Run as sbatch -D `pwd` --job-name=fracdata_001D_100U --output=./outputs/frac_001D_100U.out --error=./outputs/frac_001D_100U.err --export=loss=kto,frac_des=0.01,des_weight=133.3 ./scripts/frac_data_sbatch.sh
# Formula for des_weight is 1.33*(frac_und * und_weight)/frac_des. Formula for und_weight is (frac_des * des_weight)/(1.33 * frac_und)

# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

source ~/.bashrc 
source /opt/conda/etc/profile.d/conda.sh 
conda activate halos

cache_dir="/data/models/archangel"

if [[ $loss == "" ]]; then
    loss="kto"
fi

if [[ $model == "" ]]; then
    model="llama7b"
fi

if [[ $frac_des == "" ]]; then
    frac_des=1.0
    des_weight=1.0
fi

if [[ $frac_und == "" ]]; then
    frac_und=1.0
    und_weight=1.0
fi

exp_name="fracdata_${loss}_${model}_FD${frac_des}_WD${des_weight}_FU${frac_und}_WU${und_weight}"
echo "$exp_name"

if [[ "$model" == "llama30b" ]]; then
    bs=16
    gas=2
else
    bs=32
    gas=1
fi


if [[ $loss == "sft+"* ]]; then
    sft_model="archangel_sft_${model}/LATEST/policy.pt"
    alignment_loss="${loss:4}"

    if [[ $alignment_loss == "kto" ]]; then
        python train.py loss="$alignment_loss" model="$model" datasets=[oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.load_from="$sft_model" ++model.batch_size="$bs" ++model.gradient_accumulation_steps="$gas" ++frac_unique_desirable="$frac_des" ++loss.desirable_weight="$des_weight" ++frac_unique_undesirable="$frac_und" ++loss.undesirable_weight="$und_weight"
    else
        python train.py loss="$alignment_loss" model="$model" datasets=[oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.load_from="$sft_model" ++model.batch_size="$bs" ++model.gradient_accumulation_steps="$gas"
    fi
else
    if [[ $loss == "kto" ]]; then
        python train.py loss="$loss" model="$model" datasets=[oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.batch_size="$bs" ++model.gradient_accumulation_steps="$gas" ++frac_unique_desirable="$frac_des" ++loss.desirable_weight="$des_weight" ++frac_unique_undesirable="$frac_und" ++loss.undesirable_weight="$und_weight"
    else
        python train.py loss="$loss" model="$model" datasets=[oasst] exp_name="$exp_name" mode=train ++cache_dir="$cache_dir" ++model.batch_size="$bs" ++model.gradient_accumulation_steps="$gas"
    fi
fi