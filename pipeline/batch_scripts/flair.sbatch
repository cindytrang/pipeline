#!/bin/bash
#
#SBATCH --job-name=example-gpu # Job name for tracking
#SBATCH --partition=gecko     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20     # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=1-00:00:00      # Job time limit set to 2 days (48 hours)
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=flair_%j.out # Standard out from your job
#SBATCH --error=flair_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
module load CUDA/11.2

cd /dcs/large/u2163087/pipeline
source myenv/bin/activate
# python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"

## Execute your program(s) ##
python train_flair.py --dataset Corpus  --project flair --seq_len 512 --model bert_classifier
