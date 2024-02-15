#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --mem=800G
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=a3

source /opt/conda/etc/profile.d/conda.sh 
# conda activate dpo
conda activate halos

args=$1
# run the command
echo python train.py $args
python train.py $args

# python train.py loss=kto model=yi_llama34b datasets=[shp,hh,oasst] exp_name=kto_yi34_yifmt_0 mode=train ++cache_dir=/data/models/yi34_kto ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++human_prefix='"<|im_start|>user\n"' ++human_suffix='"<|im_end|>\n"' ++assistant_prefix='"<|im_start|>assistant\n"'
# python train.py loss=kto model=yi_llama34b datasets=[ultrabin] exp_name=kto_yi34_yifmt_2 mode=train ++cache_dir=/data/models/yi34_kto ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++human_prefix='"<|im_start|>user\n"' ++human_suffix='"<|im_end|>\n"' ++assistant_prefix='"<|im_start|>assistant\n"'
# python train.py loss=kto model=yi_llama34b datasets=[shp2] exp_name=kto_yi34_yifmt_3 mode=train ++cache_dir=/data/models/yi34_kto ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++human_prefix='"<|im_start|>user\n"' ++human_suffix='"<|im_end|>\n"' ++assistant_prefix='"<|im_start|>assistant\n"'
# python train.py loss=kto model=yi_llama34b datasets=[oasst] exp_name=kto_yi34_yifmt_4 mode=train ++cache_dir=/data/models/yi34_kto ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++human_prefix='"<|im_start|>user\n"' ++human_suffix='"<|im_end|>\n"' ++assistant_prefix='"<|im_start|>assistant\n"'

# python train.py  --config-name config_yi34 loss=kto model=yi_llama34b_chat datasets=[shp,hh,oasst] exp_name=kto_yi34_chat_yifmt_0 mode=train ++cache_dir=/data/models/yi34_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++human_prefix='"<|im_start|>user\n"' ++human_suffix='"<|im_end|>\n"' ++assistant_prefix='"<|im_start|>assistant\n"'
# python train.py --config-name config_yi34  loss=kto model=yi_llama34b_chat datasets=[ultrabin] exp_name=kto_yi34_chat_yifmt_1 mode=train ++cache_dir=/data/models/yi34_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++human_prefix='"<|im_start|>user\n"' ++human_suffix='"<|im_end|>\n"' ++assistant_prefix='"<|im_start|>assistant\n"'
# python train.py --config-name config_yi34  loss=kto model=yi_llama34b_chat datasets=[shp2] exp_name=kto_yi34_chat_yifmt_2 mode=train ++cache_dir=/data/models/yi34_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++human_prefix='"<|im_start|>user\n"' ++human_suffix='"<|im_end|>\n"' ++assistant_prefix='"<|im_start|>assistant\n"'
# python train.py --config-name config_yi34  loss=kto model=yi_llama34b_chat datasets=[oasst] exp_name=kto_yi34_chat_yifmt_3 mode=train ++cache_dir=/data/models/yi34_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2 ++human_prefix='"<|im_start|>user\n"' ++human_suffix='"<|im_end|>\n"' ++assistant_prefix='"<|im_start|>assistant\n"'

# python train.py loss=kto model=mistral7b_instruct datasets=[distilabel_orca_steamshp_filter] exp_name=kto_mistral7b_instruct_orca mode=train ++cache_dir=/data/models/orca_kto
# python train.py loss=kto model=gen_m7_sq2048_tulu2_ep2 datasets=[distilabel_orca_steamshp_filter] exp_name=kto_gen_m7_sq2048_tulu2_ep2_orca mode=train ++cache_dir=/data/models/debug
# python train.py --config-name config_yi34 loss=kto model=yi_llama34b_chat datasets=[distilabel_orca_steamshp_filter] exp_name=kto_yi34_chat_yifmt_orca mode=train ++cache_dir=/data/models/orca_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2

# python train.py loss=kto model=mistral7b_instruct datasets=[distilabel_orca_argilla_filter] exp_name=kto_mistral7b_instruct_orca_argilla mode=train ++cache_dir=/data/models/orca_kto
# python train.py loss=kto model=gen_m7_sq2048_tulu2_ep2 datasets=[distilabel_orca_argilla_filter] exp_name=kto_gen_m7_sq2048_tulu2_ep2_orca_argilla mode=train ++cache_dir=/data/models/orca_kto
# python train.py --config-name config_yi34 loss=kto model=yi_llama34b_chat datasets=[distilabel_orca_argilla_filter] exp_name=kto_yi34_chat_yifmt_orca_argilla mode=train ++cache_dir=/data/models/orca_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2

############ SLURM COMMANDS ##############
# sbatch -D `pwd` --job-name=kto_yi34b_chat_1 --output=./outputs/yi34b_chat_kto_1.out --export=loss=kto,model=yi_llama34b_chat ./scripts/kto_sota_sbatch.sh  # shp,oasst,hh
# sbatch -D `pwd` --job-name=kto_yi34b_chat_2 --output=./outputs/yi34b_chat_kto_2.out --export=loss=kto,model=yi_llama34b_chat,dataset=ultrabin ./scripts/kto_sota_sbatch.sh
# sbatch -D `pwd` --job-name=kto_yi34b_chat_3 --output=./outputs/yi34b_chat_kto_3.out --export=loss=kto,model=yi_llama34b_chat ./scripts/kto_sota_sbatch.sh  # shp2,ultrabin
# sbatch -D `pwd` --job-name=kto_yi34b_chat_4 --output=./outputs/yi34b_chat_kto_4.out --export=loss=kto,model=yi_llama34b_chat,dataset=oasst ./scripts/kto_sota_sbatch.sh  # shp2,ultrabin
#
# sbatch -D `pwd` --job-name=kto_yi34b_chat_yifmt_1 --output=./outputs/yi34b_chat_kto_yifmt_1.out --export=loss=kto,model=yi_llama34b_chat ./scripts/kto_sota_sbatch.sh  # shp,oasst,hh
# sbatch -D `pwd` --job-name=kto_yi34b_chat_yifmt_2 --output=./outputs/yi34b_chat_kto_yifmt_2.out --export=loss=kto,model=yi_llama34b_chat,dataset=ultrabin ./scripts/kto_sota_sbatch.sh
# sbatch -D `pwd` --job-name=kto_yi34b_chat_yifmt_3 --output=./outputs/yi34b_chat_kto_yifmt_3.out --export=loss=kto,model=yi_llama34b_chat ./scripts/kto_sota_sbatch.sh  # shp2,ultrabin
# sbatch -D `pwd` --job-name=kto_yi34b_chat_yifmt_4 --output=./outputs/yi34b_chat_kto_yifmt_4.out --export=loss=kto,model=yi_llama34b_chat,dataset=oasst ./scripts/kto_sota_sbatch.sh