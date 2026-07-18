#!/bin/bash
#SBATCH --job-name=datagen_2
#SBATCH --output=slurm/logs/datagen_2_%j.out
#SBATCH --error=slurm/logs/datagen_2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=bigbatch

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs data

python c3_datagen_2.py \
    --n_outer 200 \
    --m_inner 50  \
    --seed    2   \
    --out     data/train_2.h5 \
    --feds    feds200628.csv

echo "End: $(date)"
