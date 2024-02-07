#!/bin/bash
CKPT=$1
# Make out be the last string after /
OUT=results/${CKPT##*/}
mkdir -p $OUT

DATADIR=evaldata/eval
FORMAT=eval.templates.create_prompt_with_halo_chat_format

pwd=$(pwd)
cd open-instruct

python -m eval.gsm.run_eval \
--data_dir $DATADIR/gsm/ \
--max_num_examples 200 \
--save_dir $OUT \
--model $CKPT \
--tokenizer_name_or_path $CKPT \
--n_shot 8 \
--use_chat_format \
--use_vllm \
--chat_formatting_function $FORMAT

python -m eval.mmlu.run_eval \
--ntrain 0 \
--data_dir $DATADIR/mmlu/ \
--save_dir $OUT \
--model_name_or_path $CKPT \
--tokenizer_name_or_path $CKPT \
--eval_batch_size 4 \
--use_chat_format \
--chat_formatting_function $FORMAT

python -m eval.bbh.run_eval \
--data_dir $DATADIR/bbh \
--save_dir $OUT \
--model $CKPT \
--tokenizer_name_or_path $CKPT \
--max_num_examples_per_task 40 \
--use_chat_format \
--use_vllm \
--chat_formatting_function $FORMAT

python -m eval.tydiqa.run_eval \
--data_dir $DATADIR/tydiqa \
--n_shot 1 \
--max_num_examples_per_lang 100 \
--max_context_length 512 \
--save_dir $OUT \
--model $CKPT \
--tokenizer_name_or_path $CKPT \
--use_chat_format \
--use_vllm \
--chat_formatting_function $FORMAT

cd ../bigcode-evaluation-harness

accelerate launch --config_file /home/niklas/sgpt2/scripts/configs/config_1gpu_vanilla.yml main.py \
--model $CKPT \
--tasks humanevalsynthesize-python \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt halo \
--save_generations_path $OUT/generations_humanevalsynthesizepython.json \
--metric_output_path $OUT/evaluation_humanevalsynthesizepython.json \
--max_length_generation 2048 \
--precision bf16
