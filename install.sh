#!/bin/bash

conda create --name halos python=3.10.14
conda activate halos

conda install pip
pip install packaging ninja
ninja --version
echo $?
conda install pytorch=2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install flash-attn==2.6.3 --no-build-isolation
pip install transformers==4.44.0 
pip install peft==0.12.0
pip install datasets==2.20.0
pip install accelerate==0.33.0
pip install vllm==0.5.5
pip install alpaca-eval immutabledict langdetect wandb omegaconf openai hydra-core==1.3.2

# lm-eval
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
# download tasks for offline eval
python << EOF
from lm_eval import tasks
task_names = ["winogrande", "mmlu", "gsm8k_cot", "bbh_cot_fewshot", "arc_easy", "arc_challenge", "hellaswag", "ifeval"]
task_dict = tasks.get_task_dict(task_names)

from datasets import load_dataset
load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")
EOF
