#!/bin/bash
#SBATCH --job-name=ThermoGradio
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=gpue[01-05,08-12]
#SBATCH --gres=gpu:1

# Afficher les informations
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Lancer le programme
python app_gradio.py