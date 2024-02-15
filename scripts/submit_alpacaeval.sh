# This script submits sbatch jobs to generate model outputs from eval.py and then runs alpaca_eval on the output jsons.
# outtxt="${output_dir}/${job_name}.txt"  # this is in interactive shell for doing SomeCommand 2>&1 | tee SomeFile.txt
# Launch:   `bash scripts/submit_alpacaeval.sh`
eval_script="launch_alpacaeval.sh"
output_dir="/home/winnie/halos/alpaca_eval_samples"
mkdir -p $output_dir

###### ADD TO ME! Should contain policy.pt ######
argslist=(
    # "/data/models/archangel/zephyr_sft_dpo"
    # "/data/models/archangel/zephyr_sft_kto"
    # "/data/models/archangel/zephyr_sft_kto_unary"
    # "HuggingFaceH4/mistral-7b-sft-beta"
    # ------- 02/15: new kto / dpo runs -------
    "/data/models/yi34_dpo_pairrm/dpo_mistral7b_instruct_v2_snorkel_pairrm_iter3_b0.1"
    "/data/models/yi34_kto_pairrm/kto_mistral7b_instruct_v2_snorkel_pairrm_iter3_b0.1"
    "/data/models/yi34_kto_pairrm/kto_mistral7b_instruct_v2_snorkel_pairrm_iter3_b0.3"
)
overrides="++mode=alpacaeval ++samples_dir=alpaca_eval_samples/ ++n_samples=null ++model.eval_batch_size=32 ++model.use_fast_tokenizer=true"

for args in "${argslist[@]}"; do
    echo "$eval_script $args"
    # Extract the directory name from the args before /LATEST or the last /
    input_string=$args
    name=$(echo "$input_string" | sed -E 's|.*/([^/]+)(/LATEST)?|\1|')
    job_name=halo_alpacaeval_${name}

    echo "job_name: $job_name"
    # if HuggingFace in the argslist, append ++saved_policy=null to the overrides
    if [[ $args == *"HuggingFace"* ]]; then
        args="/data/models/archangel/zephyr_sft_kto/"
        overrides="${overrides} ++saved_policy=null"
    fi

    # echo sbatch -D `pwd` --job-name=$job_name --output=$output_dir/$job_name.out ./scripts/${eval_script} "${args} ${overrides}"
    # step 1 generate policy outputs for alpacaeval
    sbatch -D `pwd` --job-name=$job_name --output=$output_dir/$job_name.out ./scripts/${eval_script} "${args} ${overrides}"
    # step 2 uncomment alongside in $eval_script for alpacaeval
    # sbatch -D `pwd` --job-name=pt2_$job_name --output=$output_dir/alpacaeval2_$job_name.out ./scripts/{eval_script} ${name} $output_dir
done