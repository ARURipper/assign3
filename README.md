# Assignment 3: Sequential Instruction Tuning of a Small LLM

Two-stage QLoRA fine-tuning pipeline with LLM-as-a-Judge evaluation, run on UTSA HPC.

**Student model:** Phi-3.5 Mini Instruct  
**Teacher/Judge model:** Llama 3.1 70B Instruct  
**Research question:** Does Stage 2 JSON instruction tuning improve structured-output reliability while preserving general instruction-following ability gained in Stage 1?

---

## Project Structure

```
assign3/
├── config.py                        # All hyperparameters — edit here only
├── requirements.txt
├── README.md
├── REPORT.md
├── data/
│   ├── prepare_alpaca.py            # Download + clean Alpaca dataset
│   └── generate_json_instruct.py   # Imitation learning from teacher model
├── train/
│   ├── stage1_train.py              # QLoRA Stage 1 (Alpaca fine-tuning)
│   └── stage2_train.py              # QLoRA Stage 2 (JSON Instruct fine-tuning)
├── inference/
│   └── generate_outputs.py          # Generate outputs at all 3 checkpoints
├── eval/
│   ├── run_json_eval.py             # JSON validity, exact match, field F1
│   ├── judge_eval.py                # LLM judge pairwise evaluation
│   ├── ablation_study.py            # Epoch/LR/dataset-size ablations
│   └── generate_figures.py          # All charts and tables for the report
├── hpc/
│   ├── stage1.slurm                 # SLURM job script for Stage 1
│   └── stage2.slurm                 # SLURM job script for Stage 2
├── prompts/
│   ├── judge_eval.txt               # Judge evaluation prompt
│   ├── teacher_json_extraction.txt
│   ├── teacher_schema_constrained_generation.txt
│   ├── teacher_exact_label_classification.txt
│   ├── teacher_json_repair.txt
│   └── teacher_tool_call_generation.txt
├── src/
│   ├── __init__.py
│   └── utils.py                     # Shared helpers
├── checkpoints/                     # Saved LoRA adapters (auto-created)
├── results/                         # Per-checkpoint model outputs (auto-created)
└── logs/                            # Metrics, figures, judge scores (auto-created)
```

---

## Setup on UTSA HPC

### 1. Connect to HPC
```bash
ssh YOUR_ABC123@hpc.utsa.edu
```

### 2. Clone your repo
```bash
cd /scratch/$USER
git clone https://github.com/YOURUSERNAME/assign3.git
cd assign3
```

### 3. Create virtual environment
```bash
module load python/3.10 cuda/12.1
python -m venv ~/.venvs/llm_assign3
source ~/.venvs/llm_assign3/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download models to scratch (recommended)
```bash
# Phi-3.5 Mini (student)
huggingface-cli download microsoft/Phi-3.5-mini-instruct \
    --local-dir /scratch/$USER/models/Phi-3.5-mini-instruct

# Llama 3.1 70B (teacher/judge) — requires HuggingFace access approval
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
    --local-dir /scratch/$USER/models/Llama-3.1-70B-Instruct
```

---

## Running the Pipeline

### Option A — Submit SLURM jobs (recommended on HPC)

```bash
# Stage 1: Alpaca fine-tuning + Checkpoint 0/1 evaluation
sbatch hpc/stage1.slurm

# After Stage 1 completes, Stage 2: JSON fine-tuning + all evaluations
sbatch hpc/stage2.slurm
```

### Option B — Run scripts manually (for debugging)

**Step 1 — Prepare data**
```bash
python data/prepare_alpaca.py
python data/generate_json_instruct.py --task all
```

**Step 2 — Train**
```bash
python train/stage1_train.py
python train/stage2_train.py
```

**Step 3 — Generate outputs at all checkpoints**
```bash
python inference/generate_outputs.py --checkpoint all
```

**Step 4 — Evaluate**
```bash
python eval/run_json_eval.py
python eval/judge_eval.py --pair all
```

**Step 5 — Run ablations**
```bash
python eval/ablation_study.py --ablation all
```

**Step 6 — Generate figures and tables**
```bash
python eval/generate_figures.py
```

---

## Configuration

All hyperparameters are in `config.py`. Key settings:

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Model | Phi-3.5 Mini Instruct | (continued from Stage 1) |
| Method | QLoRA 4-bit | QLoRA 4-bit |
| LoRA rank | 16 | 16 |
| Learning rate | 2e-5 | 1e-5 |
| Epochs | 3 | 2 |
| Batch size | 4 × 8 accum = 32 | 4 × 8 accum = 32 |
| Max seq length | 1024 | 1024 |

---

## Evaluation Protocol

Three checkpoints are evaluated:

| Checkpoint | Description |
|---|---|
| Checkpoint 0 | Untuned baseline (Phi-3.5 Mini Instruct) |
| Checkpoint 1 | After Stage 1 — Alpaca fine-tuned |
| Checkpoint 2 | After Stage 2 — JSON Instruct fine-tuned |

Two evaluation suites per checkpoint:
- **Alpaca eval:** Judge win-rate comparison against baseline
- **JSON eval:** Validity rate, exact match, field-level F1, per-task breakdown

---

## Academic Disclosure

AI tools (Claude) were used for coding assistance and project structure. All experimental design, model selection, prompt engineering, data curation, result interpretation, and write-up were completed by the author.
