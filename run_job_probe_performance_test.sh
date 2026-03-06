#!/bin/bash
#SBATCH --job-name=rr-sr_1
#SBATCH --partition=gpu           # or your appropriate partition
#SBATCH --gres=gpu:2              # number of GPUs
#SBATCH --mem=250G
#SBATCH --cpus-per-task=4
#SBATCH --time=7:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u7215128@anu.edu.au

# Define the environment path for Conda inside container
export SINGULARITYENV_CONDAENV="/home/projects/u7215128/pytorch-environment"

# Execute commands inside the Singularity container using a heredoc
singularity exec --nv --bind /home/projects \
    /opt/apps/containers/conda/conda-nvidia-22.04-latest.sif \
    bash <<EOF

# Source container-specific shell config (important!)
source /.singularity_bash

# Setup shell and conda
conda activate "\$CONDAENV"

#clean
rm -rf ~/.cache/huggingface

# HuggingFace login (assumes token is in a file)
export HUGGING_FACE_HUB_TOKEN=$(cat /home/projects/u7215128/Trainingprobe/.hf_token)
#--token < /home/projects/u7215128/Trainingprobe/.hf_token > /dev/null 2>&1

# Change to project directory
cd /home/projects/u7215128/Trainingprobe

# Run Python scripts
echo "Running main script..."
python ProbeClassifierPerformanceLayer-wise.py

EOF
