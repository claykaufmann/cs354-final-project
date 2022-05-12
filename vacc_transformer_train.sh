#!/bin/bash

#SBATCH --partition=dggpu

# Request Nodes
#SBATCH --nodes=2

# Req CPU Cores
#SBATCH --ntasks=2

# Req GPUs
#SBATCH --gres=gpu:8

# Req Memory
#SBATCH --mem=30G

# Run for x minutes
#SBATCH --time=1200

# Name of job
#SBATCH --job-name=MusicTransformer

# Name output file
#SBATCH --output=%x_j%j.out

# Email
#SBATCH --mail-type=ALL

source ~/.bashrc

cd ${SLURM_SUBMIT_DIR}

# make results folder
mkdir results

# Executable section: echoing some Slurm data
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"
conda activate music_transformer
time python train_transformer.py
