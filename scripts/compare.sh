#!/bin/bash

# Copyright (c) 2023 Contextual AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

loss="$1"
cache_dir="/data/models/archangel"

for model in pythia1-4b pythia2-8b pythia6-9b pythia12-0b llama7b llama13b llama30b; do
    exp_name="archangel_${loss}_${model}"
    echo "$exp_name"
    python compare.py -f "samples/${exp_name}.json" -mc 512
    python compare.py -f "samples/${exp_name}.json" -mc 512 -bk chosen
done