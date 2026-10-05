#!/bin/bash
#SBATCH --job-name=p3vae_004_r001_gpu
#SBATCH --output=/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/slurm_jobs/gpu_job_%j.out
#SBATCH --error=/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/slurm_jobs/gpu_job_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leejv2@vcu.edu

# Environment Setup
module purge
module load cuda

# Activate virtual environment
source /lustre/home/leejv2/git_repos/jvlee_LIBS_ML/.venv/bin/activate

# Run the job
python3 -u /lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_004.py

# To submit the job, type the following in the command line from jvlee_LIBS_ML folder:
# sbatch /lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/slurm_jobs/submit_p3vae_004_gpu.sh