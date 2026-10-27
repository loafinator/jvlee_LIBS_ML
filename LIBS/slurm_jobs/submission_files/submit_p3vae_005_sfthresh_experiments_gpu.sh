#!/bin/bash
#SBATCH --job-name=p3vae_005_sfthresh_experiments_r001_gpu
#SBATCH --output=/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/slurm_jobs/expr_sfthresh_gpu_job_%j.out
#SBATCH --error=/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/slurm_jobs/expr_sfthresh_gpu_job_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=athena522
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leejv2@vcu.edu

# Environment Setup
module purge
module load cuda

# Ensure script doesn't exit automatically on non-zero exit status
set +e

# Activate virtual environment
source /lustre/home/leejv2/git_repos/jvlee_LIBS_ML/.venv/bin/activate

echo "=== CUDA & SLURM DIAGNOSTICS ==="
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count())"
echo "================================="

# Helper function to execute experiments safely
run_experiment() {
    local script_path="$1"
    local script_name=$(basename "$script_path")

    echo "=========================================================================="
    echo " 🚀 STARTING EXPERIMENT: ${script_name}"
    echo "=========================================================================="
    
    python3 -u "$script_path"
    local exit_status=$?

    if [ $exit_status -eq 0 ]; then
        echo "✅ SUCCESS: ${script_name} finished cleanly."
    else
        echo "❌ FAILURE: ${script_name} failed with exit code ${exit_status}. Moving to next experiment..."
    fi
    echo ""
}

# Run the experiments sequentially
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_50lambda.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_75lambda.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_DecoderFreezing.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_FixedPerChannellambda.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_FocalRangeAwareLossWeightSm.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_Learnlambda.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_LowerLR.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_LowShotOversampling.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_MatrixResidualConstraint.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_NoHardZero.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_NonZeroStratifiedEvalMetrics.py"
run_experiment "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/p3vae_005_sfthresh_RelativeLossWeightSchedule.py"

echo "=========================================================================="
echo " ALL EXPERIMENT RUNS COMPLETED"
echo "=========================================================================="

# To submit the job, type the following in the command line from jvlee_LIBS_ML folder:

# sbatch /lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/slurm_jobs/submit_p3vae_005_sfthresh_experiments_gpu.sh