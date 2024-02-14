#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --mem=800G
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=a3

source /opt/conda/etc/profile.d/conda.sh 
# source /home/niklas/miniconda/bin/activate
conda activate sgpt2py310

args=$1
format=$2
prompt=$3
script=/home/winnie/halos/scripts/halo_eval.sh

# run the command
# bash halo_eval.sh $args $format $prompt
bash $script $args $format $prompt