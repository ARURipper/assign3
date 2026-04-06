"""
eval/ablation_study.py
Run ablation experiments varying Stage 2 training decisions:
  1. Number of epochs (1 vs 2 vs 3)
  2. Learning rate (2e-5 vs 1e-5 vs 5e-6)
  3. Dataset size (100% vs 50% vs 25%)

Each ablation re-trains Stage 2 from the Stage 1 checkpoint and
evaluates JSON accuracy + Alpaca retention.

Run:
    python eval/ablation_study.py --ablation epochs
    python eval/ablation_study.py --ablation lr
    python eval/ablation_study.py --ablation dataset_size
    python eval/ablation_study.py --ablation all
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, set_seed,
)
from trl import SFTTrainer
from datasets import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    STUDENT_MODEL, DATA_DIR, STAGE1_OUTPUT_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    RESULTS_DIR, STAGE2_BATCH_SIZE, STAGE2_GRAD_ACCUM, STAGE2_MAX_SEQ_LEN,
    STAGE2_WARMUP_RATIO, STAGE2_WEIGHT_DECAY,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_DOUBLE_QUANT,
    SEED,
)
from src.utils import load_json, save_json, extract_json, compute_field_f1, format_json_prompt

set_seed(SEED)

ABLATION_CONFIGS = {
    "epochs": [
        {"name": "epochs_1", "epochs": 1, "lr": 1e-5, "data_fraction": 1.0},
        {"name": "epochs_2", "epochs": 2, "lr": 1e-5, "data_fraction": 1.0},
        {"name": "epochs_3", "epochs": 3, "lr": 1e-5, "data_fraction": 1.0},
    ],
    "lr": [
        {"name": "lr_2e5",  "epochs": 2, "lr": 2e-5, "data_fraction": 1.0},
        {"name": "lr_1e5",  "epochs": 2, "lr": 1e-5, "data_fraction": 1.0},
        {"name": "lr_5e6",  "epochs": 2, "lr": 5e-6, "data_fraction": 1.0},
    ],
    "dataset_size": [
        {"name": "data_100pct", "epochs": 2, "lr": 1e-5, "data_fraction": 1.0},
        {"name": "data_50pct",  "epochs": 2, "lr": 1e-5, "data_fraction": 0.5},
        {"name": "data_25pct",  "epochs": 2, "lr": 1e-5, "data_fraction": 0.25},
    ],
}


def load_training_data(fraction: float = 1.0) -> Dataset:
    data = load_json(os.path.join(DATA_DIR, "json_instruct_train.json"))
    n = max(1, int(len(data) * fraction))
    data = data[:n]
    formatted = []
    for ex in data:
        prompt = format_json_prompt(
            ex["instruction"], ex.get("input", ""),
            include_output=True, output=ex["output"]
        )
        formatted.append({"text": prompt})
    return Dataset.from_list(formatted)


def train_ablation(config: dict) -> str:
    """Train Stage 2 with a specific ablation config. Returns checkpoint dir."""
    out_dir = os.path.join(CHECKPOINTS_DIR, f"ablation_{config['name']}")
    os.makedirs(out_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_DOUBLE_QUANT,
    )

    tokenizer = AutoTokenizer.from_pretrained(STAGE1_OUTPUT_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, STAGE1_OUTPUT_DIR)
    model = model.merge_and_unload()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_training_data(config["data_fraction"])
    print(f"Ablation '{config['name']}': {len(dataset)} examples, "
          f"lr={config['lr']}, epochs={config['epochs']}")

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=STAGE2_BATCH_SIZE,
        gradient_accumulation_steps=STAGE2_GRAD_ACCUM,
        learning_rate=config["lr"],
        warmup_ratio=STAGE2_WARMUP_RATIO,
        weight_decay=STAGE2_WEIGHT_DECAY,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=dataset, tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=STAGE2_MAX_SEQ_LEN,
        packing=False,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def evaluate_ablation(config: dict, ckpt_dir: str) -> dict:
    """Evaluate a trained ablation checkpoint on JSON and Alpaca held-out sets."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_DOUBLE_QUANT,
    )

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()

    def generate(prompt: str, max_new: int = 256) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=STAGE2_MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    # JSON eval
    json_data = load_json(os.path.join(DATA_DIR, "json_instruct_eval.json"))
    json_valid, json_exact = 0, 0
    for ex in tqdm(json_data, desc=f"JSON eval [{config['name']}]"):
        prompt = format_json_prompt(ex["instruction"], ex.get("input", ""))
        response = generate(prompt, max_new=512)
        _, valid = extract_json(response)
        if valid:
            json_valid += 1
            pred_norm = json.dumps(extract_json(response)[0], sort_keys=True)
            gold_norm = json.dumps(extract_json(ex["output"])[0], sort_keys=True)
            if pred_norm == gold_norm:
                json_exact += 1

    n_json = len(json_data) or 1

    # Alpaca eval — just validity proxy (response is non-empty)
    alpaca_data = load_json(os.path.join(DATA_DIR, "alpaca_eval.json"))
    alpaca_nonempty = 0
    for ex in tqdm(alpaca_data[:50], desc=f"Alpaca eval [{config['name']}]"):
        from src.utils import format_alpaca_prompt
        prompt = format_alpaca_prompt(ex["instruction"], ex.get("input", ""))
        response = generate(prompt)
        if len(response.strip()) > 10:
            alpaca_nonempty += 1

    del model
    torch.cuda.empty_cache()

    return {
        "config":               config["name"],
        "epochs":               config["epochs"],
        "lr":                   config["lr"],
        "data_fraction":        config["data_fraction"],
        "json_validity_rate":   round(json_valid / n_json, 4),
        "json_exact_match":     round(json_exact / n_json, 4),
        "alpaca_response_rate": round(alpaca_nonempty / 50, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", default="all",
                        choices=["epochs", "lr", "dataset_size", "all"])
    args = parser.parse_args()

    ablations = (
        [k for k in ABLATION_CONFIGS]
        if args.ablation == "all"
        else [args.ablation]
    )

    all_results = {}
    for abl_name in ablations:
        configs = ABLATION_CONFIGS[abl_name]
        results = []
        for cfg in configs:
            print(f"\n{'='*50}")
            print(f"Running ablation: {cfg['name']}")
            print(f"{'='*50}")
            ckpt_dir = train_ablation(cfg)
            metrics  = evaluate_ablation(cfg, ckpt_dir)
            results.append(metrics)
            print(f"Results: {metrics}")

        all_results[abl_name] = results
        save_json(
            os.path.join(LOGS_DIR, f"ablation_{abl_name}.json"),
            results
        )

    save_json(os.path.join(LOGS_DIR, "ablation_all.json"), all_results)
    print(f"\nAll ablation results saved to {LOGS_DIR}")


if __name__ == "__main__":
    main()
