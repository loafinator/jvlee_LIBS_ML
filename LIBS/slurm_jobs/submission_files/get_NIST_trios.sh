#!/bin/bash
#SBATCH --job-name=nist_libs_fetch
#SBATCH --output=/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/nist_%j.out
#SBATCH --error=/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/nist_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00


# 1. Environment & Folder Setup
export PYTHONUNBUFFERED=1
mkdir -p /lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs
mkdir -p /lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_trio_data

# Ensure SLURM_TMPDIR is set
if [ -z "$SLURM_TMPDIR" ]; then
    export SLURM_TMPDIR="/tmp/slurm_${SLURM_JOB_ID}"
    mkdir -p "$SLURM_TMPDIR"
fi

echo "=========================================================="
echo "Job Started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Scratch Directory: $SLURM_TMPDIR"
echo "=========================================================="

# 2. Execute Python script using its absolute path
python -u /lustre/home/leejv2/git_repos/jvlee_LIBS_ML/utils/NIST_data.py

EXIT_CODE=$?

# 3. Cleanup local node scratch space
echo "Cleaning up local scratch..."
rm -rf "${SLURM_TMPDIR:?}"/nist_csv_scratch

echo "=========================================================="
echo "Job Finished with exit code $EXIT_CODE at $(date)"
echo "=========================================================="

exit $EXIT_CODE