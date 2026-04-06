"""
config.py — All hyperparameters and paths for Assignment 3.
Nothing is hardcoded elsewhere; every script imports from here.
"""
import os
from dataclasses import dataclass, field
from typing import List

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
PROMPTS_DIR     = os.path.join(BASE_DIR, "prompts")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

# ── Student Model ──────────────────────────────────────────────────────────────
STUDENT_MODEL   = "microsoft/Phi-3.5-mini-instruct"
# Alternatives: "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"

# ── Teacher / Judge Model ──────────────────────────────────────────────────────
TEACHER_MODEL   = "meta-llama/Llama-3.1-70B-Instruct"   # used on HPC
JUDGE_MODEL     = "meta-llama/Llama-3.1-70B-Instruct"   # same model for judging

# ── QLoRA Parameters ──────────────────────────────────────────────────────────
LORA_RANK       = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# ── Training — Stage 1 (Alpaca) ───────────────────────────────────────────────
STAGE1_EPOCHS         = 3
STAGE1_LR             = 2e-5
STAGE1_BATCH_SIZE     = 4
STAGE1_GRAD_ACCUM     = 8          # effective batch = 32
STAGE1_MAX_SEQ_LEN    = 1024
STAGE1_WARMUP_RATIO   = 0.03
STAGE1_WEIGHT_DECAY   = 0.001
STAGE1_OUTPUT_DIR     = os.path.join(CHECKPOINTS_DIR, "stage1_alpaca")

# ── Training — Stage 2 (JSON Instruct) ────────────────────────────────────────
STAGE2_EPOCHS         = 2
STAGE2_LR             = 1e-5       # lower LR to reduce forgetting
STAGE2_BATCH_SIZE     = 4
STAGE2_GRAD_ACCUM     = 8
STAGE2_MAX_SEQ_LEN    = 1024
STAGE2_WARMUP_RATIO   = 0.03
STAGE2_WEIGHT_DECAY   = 0.001
STAGE2_OUTPUT_DIR     = os.path.join(CHECKPOINTS_DIR, "stage2_json")

# ── Dataset Sizes ──────────────────────────────────────────────────────────────
ALPACA_TRAIN_SIZE     = 10_000     # subset of full Alpaca
ALPACA_EVAL_SIZE      = 200        # held-out eval set
JSON_TRAIN_SIZE       = 2_000      # teacher-generated examples
JSON_EVAL_SIZE        = 200        # held-out JSON eval set

# ── Teacher Generation ─────────────────────────────────────────────────────────
TEACHER_TEMPERATURE   = 0.7
TEACHER_MAX_NEW_TOKENS = 512
JSON_TASK_TYPES = [
    "json_extraction",
    "schema_constrained_generation",
    "exact_label_classification",
    "json_repair",
    "tool_call_generation",
]
EXAMPLES_PER_TASK     = 400        # 5 types × 400 = 2000 total

# ── Judge Evaluation ──────────────────────────────────────────────────────────
JUDGE_TEMPERATURE     = 0.0
JUDGE_MAX_NEW_TOKENS  = 512
JUDGE_EVAL_SAMPLE     = 100        # samples per checkpoint pair

# ── Evaluation ─────────────────────────────────────────────────────────────────
ALPACA_EVAL_PROMPTS   = 100
JSON_EVAL_PROMPTS     = 100

# ── Quantization ──────────────────────────────────────────────────────────────
LOAD_IN_4BIT          = True
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_QUANT_TYPE   = "nf4"
BNB_DOUBLE_QUANT      = True

# ── HPC ───────────────────────────────────────────────────────────────────────
HPC_PARTITION         = "gpu"
HPC_NODES             = 1
HPC_GPUS_PER_NODE     = 1          # A100 80GB recommended
HPC_CPUS_PER_TASK     = 8
HPC_MEMORY            = "64G"
HPC_TIME_STAGE1       = "08:00:00"
HPC_TIME_STAGE2       = "06:00:00"

# ── Random Seed ───────────────────────────────────────────────────────────────
SEED = 42
