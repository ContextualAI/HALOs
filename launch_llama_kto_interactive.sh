#!/bin/bash

set -x

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_SHOW_CPP_STACKTRACES=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

accelerate launch --debug \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --main_process_port 29500 \
    launch.py loss=kto model=llama datasets=[ultrafeedback_armorm] \
    exp_name=llama3-8B-instruct-kto-prelec \
    ++cache_dir=/scratch/gpfs/sl2998/models \
    ++model.name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
    ++lr=2.5e-6 \
    ++loss.beta=0.25 \
    ++model.batch_size=32 \
    ++model.gradient_accumulation_steps=1 \
    ++model.eval_batch_size=32 \
    ++n_examples=10_000 \
    ++humanline=true \
    ++humanline_M=1.6 \
    ++humanline_gamma=1.01

set +x