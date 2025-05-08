#!/bin/bash

export HALO_DIR=/home/sl2998/workspace/HALOs

for v in 1 2 3; do
    sbatch ${HALO_DIR}/scripts/launch_llama_instruct_dpo_online_v${v}.sh 0.1 1e-6
    sbatch ${HALO_DIR}/scripts/launch_llama_instruct_kto_online_v${v}.sh 0.5 1e-6
    sbatch ${HALO_DIR}/scripts/launch_llama_instruct_ppo_online_v${v}.sh 0.025 1e-6 0.7
    sbatch ${HALO_DIR}/scripts/launch_llama_instruct_grpo_online_v${v}.sh 0.005 1e-6 0.005 1 1
done
