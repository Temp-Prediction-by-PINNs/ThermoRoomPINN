#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

PYTHON_FILE=$1

if [ -z "$PYTHON_FILE" ]; then
    echo "Erreur : aucun fichier Python fourni."
    echo "Usage : sbatch --job-name=Nom run.sh fichier.py"
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

python "$PYTHON_FILE"