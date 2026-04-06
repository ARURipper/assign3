#!/bin/bash
#SBATCH --job-name=assign3_stage2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@utsa.edu

# ── Environment ───────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

module purge
module load python/3.10
module load cuda/12.1

source ~/.venvs/llm_assign3/bin/activate

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets

cd $SLURM_SUBMIT_DIR

# ── Step 1: Generate JSON Instruct dataset via teacher model ──────────────────
echo "=== Generating teacher-model JSON Instruct dataset ==="
python data/generate_json_instruct.py --task all \
    --model_path /scratch/$USER/models/Llama-3.1-70B-Instruct

# ── Step 2: Run Stage 2 training ──────────────────────────────────────────────
echo "=== Starting Stage 2: JSON Instruct Fine-Tuning ==="
python train/stage2_train.py

# ── Step 3: Generate Checkpoint 2 outputs ────────────────────────────────────
echo "=== Generating Checkpoint 2 (JSON-tuned) outputs ==="
python inference/generate_outputs.py --checkpoint 2

# ── Step 4: Run all evaluations ────────────────────────────────────────────────
echo "=== Running JSON evaluation (all checkpoints) ==="
python eval/run_json_eval.py

echo "=== Running Judge evaluation (all pairs) ==="
python eval/judge_eval.py --pair all \
    --model_path /scratch/$USER/models/Llama-3.1-70B-Instruct

# ── Step 5: Generate figures ───────────────────────────────────────────────────
echo "=== Generating figures ==="
python eval/generate_figures.py

echo "=== Full pipeline complete ==="
echo "Job ended: $(date)"
