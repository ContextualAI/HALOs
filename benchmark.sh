#!/bin/bash

MODEL_PATH=$1        # HF model name or local directory

# can add any task in LMEval (note: bug if you use 'auto' batch size with gsm8k_cot)
lm_eval --model hf \
  --model_args pretrained="$MODEL_PATH",tokenizer="$MODEL_PATH",parallelize=True \
  --tasks arc_easy,arc_challenge,winogrande,bbh_cot_fewshot,gsm8k_cot \
  --batch_size 4

# AlpacaEval
# Count the number of GPUs, default to 0 if nvidia-smi is not available
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l) || GPU_COUNT=0
OUTPUT_FILE="outputs/alpacaeval/$(echo "$MODEL_PATH" | tr '/' '_').json"
# OPENAI_API_KEY needs to be set
python -m eval.sample_for_alpacaeval "$MODEL_PATH" --gpu_count $GPU_COUNT --output_file "$OUTPUT_FILE"
alpaca_eval evaluate --is_overwrite_leaderboard=True --model_outputs="$OUTPUT_FILE"
