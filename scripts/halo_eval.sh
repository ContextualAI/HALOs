#!/bin/bash
CKPT=$1  # bash scripts/halo_eval.sh /data/models/archangel/

# check if file called pytorch_model.bin exists inside the CKPT directory, if not, run the converrsion script below
if [ ! -f "$CKPT/pytorch_model.bin" ]; then
    echo 'Converting checkpoint to pytorch format FIRST!'
    python /home/winnie/halos/scripts/reformat_statedict_halo_eval.py $CKPT
fi

echo 'Starting evals...'

# Make out be the last string after /
OUT=/home/winnie/halo_evals/kto/${CKPT##*/}
mkdir -p $OUT

DATADIR=/data/niklas/sgpt2/evaldata/eval
#FORMAT=eval.templates.create_prompt_with_sgpt2_chat_format
#FORMAT=eval.templates.create_prompt_with_sgpt2_zephyr_chat_format  # zephyr
# FORMAT=eval.templates.create_prompt_with_sgpt2_tulu_chat_format  # ours
# FORMAT=eval.templates.create_prompt_with_yi_chat_format  # yi34b format
# FORMAT=eval.templates.create_prompt_with_mistral_chat_format
# FORMAT=eval.templates.create_prompt_with_halo_chat_format  # human assistant

FORMAT=$2  # eval.templates.create_prompt_with_sgpt2_chat_format
echo FORMAT: $FORMAT

pwd=$(pwd)
cd /home/niklas/sgpt2/evaluation/open-instruct

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

echo "GSM done"

python -m eval.mmlu.run_eval \
--ntrain 0 \
--data_dir $DATADIR/mmlu/ \
--save_dir $OUT \
--model_name_or_path $CKPT \
--tokenizer_name_or_path $CKPT \
--eval_batch_size 4 \
--use_chat_format \
--chat_formatting_function $FORMAT

echo "MMLU done"

python -m eval.bbh.run_eval \
--data_dir $DATADIR/bbh \
--save_dir $OUT \
--model $CKPT \
--tokenizer_name_or_path $CKPT \
--max_num_examples_per_task 40 \
--use_chat_format \
--use_vllm \
--chat_formatting_function $FORMAT

echo "BBH done"

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

echo "TYDIQA done"


cd /home/niklas/sgpt2/evaluation/bigcode-evaluation-harness
# leave these commented out if you're directly running from the temrinal in the srcnik && sgpt activated envs
# source ~/.bashrc 
# source /opt/conda/etc/profile.d/conda.sh # doesn't work 
# source /home/niklas/miniconda/bin/activate
# conda activate sgpt2py310

# PROMPT=tulu
# PROMPT=yi  # yi34b chat models
# if three arguments where passed in, set the third one to prompt
if [ $# -eq 3 ]; then
    echo "Setting PROMPT to $3"
    PROMPT=$3
fi

accelerate launch --config_file /home/niklas/sgpt2/scripts/configs/config_1gpusfsdp_m7.yml main.py \
--model $CKPT \
--tasks humanevalsynthesize-python \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 5 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt $PROMPT \
--save_generations_path $OUT/generations_humanevalsynthesizepython.json \
--metric_output_path $OUT/evaluation_humanevalsynthesizepython.json \
--max_length_generation 2048 \
--precision bf16

echo "Humaneval done"