#!/bin/bash
#
#SBATCH --job-name=example-gpu # Job name for tracking
#SBATCH --partition=gecko     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12     # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=annotated%j.out # Standard out from your job
#SBATCH --error=annotated%j.err  # Standard error from your job

## Initialisation ##
cd /dcs/large/u2163087/pipeline
source /etc/profile.d/modules.sh

module load CUDA/11.2
source myenv/bin/activate

## Execute your program(s) ##
python add_predictions.py --eval_mode segments --window_size 124 --model_version non_quantized --run 2024-04-04_14-04-31__bert_classifier_on_PAN12_with_seq-len-512
