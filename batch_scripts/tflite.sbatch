#!/bin/bash
#
#SBATCH --job-name=example-gpu # Job name for tracking
#SBATCH --partition=gecko     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16     # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=1-00:00:00      # Job time limit set to 2 days (48 hours)
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=tflite%j.out # Standard out from your job
#SBATCH --error=tflite%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
module load CUDA/11.2


source /dcs/large/u2163087/miniconda3/etc/profile.d/conda.sh
conda activate newenv

cd /dcs/large/u2163087/pipeline
export PYTHONNOUSERSITE=1
# pip install -U flatbuffers


## Execute your program(s) ##
/dcs/large/u2163087/miniconda3/envs/newenv/bin/python train_flair.py --project modelmaker --dataset Corpus --seq_len 100 --model bert_classifier
