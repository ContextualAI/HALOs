# cd /home/winnie/core_cp/experimental/dpo
yi_cfg="--config-name config_yi34"
script="launch_train.sh"
betas=(0.3 0.1)
argslist=(
    # "$yi_cfg loss=kto model=yi_llama34b datasets=[shp2] exp_name=kto_yi34_yifmt_shp2 mode=train ++cache_dir=/data/models/yi34_kto ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    # "$yi_cfg loss=kto model=yi34b_chat datasets=[shp,hh,oasst] mode=train ++cache_dir=/data/models/yi34_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    # "$yi_cfg loss=kto model=yi34b_chat datasets=[shp2] mode=train ++cache_dir=/data/models/yi34_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    # "$yi_cfg loss=kto model=yi34b_chat datasets=[oasst] mode=train ++cache_dir=/data/models/yi34_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    # "$yi_cfg loss=kto model=yi34b_chat datasets=[ultrabin] mode=train ++cache_dir=/data/models/yi34_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    # ------ manually reset eos token only -------
    # "$yi_cfg loss=kto model=yi34b_chat datasets=[orca_dpo_pairs] mode=train ++cache_dir=/data/models/yi34_kto ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    # "$yi_cfg loss=kto model=yi34b_chat datasets=[ultrabin] mode=train ++cache_dir=/data/models/yi34_kto_neweos ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    # ------ manually reset eos token && id -------
    # "$yi_cfg loss=kto model=yi34b_chat datasets=[ultrabin,oasst] mode=train ++cache_dir=/data/models/yi34_kto_neweos ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    # "$yi_cfg loss=kto model=yi34b_chat datasets=[orca_dpo] mode=train ++cache_dir=/data/models/yi34_kto_neweos ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    "$yi_cfg loss=kto model=yi34b_chat datasets=[distilabel_orca_argilla_filter] mode=train ++cache_dir=/data/models/yi34_kto_neweos ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    "$yi_cfg loss=kto model=yi34b_chat datasets=[distilabel_orca_steamshp_filter] mode=train ++cache_dir=/data/models/yi34_kto_neweos ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    # "$yi_cfg loss=kto model=yi34b_chat datasets=[ultrabin] mode=train ++cache_dir=/data/models/yi34_kto_neweos ++model.use_fast_tokenizer=false ++model.batch_size=16 ++model.gradient_accumulation_steps=2"
    # ------- dpo pairrm 02/12 ---------
    "loss=dpo model=mistral7b_instruct_v2 datasets=[snorkel_pairrm_iter1] mode=train ++cache_dir=/data/models/yi34_dpo_pairrm ++model.use_fast_tokenizer=false"
    "loss=kto model=mistral7b_instruct_v2 datasets=[snorkel_pairrm_iter1] mode=train ++cache_dir=/data/models/yi34_kto_pairrm ++model.use_fast_tokenizer=false"
    # --- TODO Set the sft model to be the previous iteration's model by looking at the dataset name. If it's *_iter2 then set sft_model to *_iter1 and then train on it ----
    # "loss=dpo model=mistral7b_instruct_v2 datasets=[snorkel_pairrm_iter2] mode=train ++cache_dir=/data/models/yi34_dpo_pairrm ++model.use_fast_tokenizer=false ++model.load_from=dpo_mistral7b_instruct_v2_snorkel_pairrm_iter1_b0.3/LATEST/policy.pt"
    # "loss=kto model=mistral7b_instruct_v2 datasets=[snorkel_pairrm_iter2] mode=train ++cache_dir=/data/models/yi34_kto_pairrm ++model.use_fast_tokenizer=false ++model.load_from=kto_mistral7b_instruct_v2_snorkel_pairrm_iter1_b0.3/LATEST/policy.pt"
    # "loss=dpo model=mistral7b_instruct_v2 datasets=[snorkel_pairrm_iter3] mode=train ++cache_dir=/data/models/yi34_dpo_pairrm ++model.use_fast_tokenizer=false ++model.load_from=dpo_mistral7b_instruct_v2_snorkel_pairrm_iter2_b0.3/LATEST/policy.pt"
    # "loss=kto model=mistral7b_instruct_v2 datasets=[snorkel_pairrm_iter3] mode=train ++cache_dir=/data/models/yi34_kto_pairrm ++model.use_fast_tokenizer=false ++model.load_from=kto_mistral7b_instruct_v2_snorkel_pairrm_iter2_b0.3/LATEST/policy.pt"
)

for args in "${argslist[@]}"; do
    for beta in "${betas[@]}"; do
        echo "python train.py $args"
        # Extract values using regex
        input_string=$args
        [[ $input_string =~ loss=([^[:space:]]+) ]]
        loss=${BASH_REMATCH[1]}
        [[ $input_string =~ model=([^[:space:]]+) ]]
        model=${BASH_REMATCH[1]}
        [[ $input_string =~ datasets=\[([^]]+)\] ]]
        datasets=${BASH_REMATCH[1]}
        # replace all commas with underscores in datasets
        datasets=${datasets//,/}
        # Remove brackets from datasets
        datasets=$(echo $datasets | tr -d '[]')

        # Combine values into job_name
        job_name="${loss}_${model}_${datasets}_b${beta}"
        echo "job_name: $job_name"

        # echo sbatch -D `pwd` --job-name=$job_name --output=./outputs/$job_name.out --export=loss=$loss,model=$model ./scripts/launch_kto_yi34.sh "${args} exp_name=$job_name ++loss.beta=$beta"
        sbatch -D `pwd` --job-name=$job_name --output=./outputs/$job_name.out --export=loss=$loss,model=$model ./scripts/${script} "${args} exp_name=$job_name ++loss.beta=$beta"
    done
done
