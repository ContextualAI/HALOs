#!/bin/bash

MODEL_PATH=$1        # HF model name or local directory # mmlu,gsm8k_cot,bbh_cot_fewshot,winogrande,arc_easy,arc_challenge,piqa,ifeval,hellaswag \

huggingface-cli download winogrande --repo-type dataset
huggingface-cli download hellaswag --repo-type dataset

lm_eval --model hf \
    --model_args pretrained="$MODEL_PATH",tokenizer="$MODEL_PATH",parallelize=True \
    --tasks winogrande \
    --limit 10 \
    --batch_size auto \
    --apply_chat_template

# HumanEval
accelerate launch eval/bigcode-evaluation-harness/main.py \
  --model "$MODEL_PATH" \
  --tasks humaneval \
  --limit 10 \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution

# AlpacaEval
# Count the number of GPUs, default to 0 if nvidia-smi is not available
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l) || GPU_COUNT=0
OUTPUT_FILE="outputs/alpacaeval/$(echo "$MODEL_PATH" | tr '/' '_').json"
python -m eval.sample_for_alpacaeval "$MODEL_PATH" --gpu_count $GPU_COUNT --output_file "$OUTPUT_FILE"
alpaca_eval evaluate --model_outputs="$OUTPUT_FILE" --is_overwrite_leaderboard=True
