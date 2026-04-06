#!/bin/bash
#SBATCH --job-name=assign3_stage1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@utsa.edu

# ── Environment ───────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# Load modules
module purge
module load python/3.10
module load cuda/12.1

# Activate virtual environment
source ~/.venvs/llm_assign3/bin/activate

# Set HuggingFace cache to scratch space
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets

# Move to project directory
cd $SLURM_SUBMIT_DIR

# ── Step 1: Prepare Alpaca data (if not already done) ────────────────────────
echo "=== Preparing Alpaca data ==="
python data/prepare_alpaca.py

# ── Step 2: Run Stage 1 training ──────────────────────────────────────────────
echo "=== Starting Stage 1: Alpaca Fine-Tuning ==="
python train/stage1_train.py

# ── Step 3: Generate Checkpoint 0 and Checkpoint 1 outputs ───────────────────
echo "=== Generating Checkpoint 0 (baseline) outputs ==="
python inference/generate_outputs.py --checkpoint 0

echo "=== Generating Checkpoint 1 (Alpaca-tuned) outputs ==="
python inference/generate_outputs.py --checkpoint 1

# ── Step 4: Run JSON evaluation on Checkpoints 0 and 1 ───────────────────────
echo "=== Running JSON evaluation ==="
python eval/run_json_eval.py

echo "=== Stage 1 pipeline complete ==="
echo "Job ended: $(date)"
