"""
inference/generate_outputs.py
Generate model outputs at all three checkpoints for evaluation.

Checkpoints:
    0 — baseline (untuned student model)
    1 — after Stage 1 (Alpaca fine-tuned)
    2 — after Stage 2 (JSON Instruct fine-tuned)

Run:
    python inference/generate_outputs.py --checkpoint 0
    python inference/generate_outputs.py --checkpoint 1
    python inference/generate_outputs.py --checkpoint 2
    python inference/generate_outputs.py --checkpoint all
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    STUDENT_MODEL, STAGE1_OUTPUT_DIR, STAGE2_OUTPUT_DIR,
    DATA_DIR, RESULTS_DIR, LOGS_DIR,
    LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_DOUBLE_QUANT,
    STAGE1_MAX_SEQ_LEN, SEED,
)

set_seed(SEED)

CHECKPOINT_NAMES = {
    0: "checkpoint_0_baseline",
    1: "checkpoint_1_alpaca",
    2: "checkpoint_2_json",
}

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

JSON_PROMPT = (
    "Below is an instruction describing a structured-output task. "
    "Respond ONLY with valid JSON, no explanation or markdown.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def load_model(checkpoint_id: int):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_DOUBLE_QUANT,
    )

    print(f"Loading checkpoint {checkpoint_id}: {CHECKPOINT_NAMES[checkpoint_id]}")

    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if checkpoint_id == 0:
        tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
        model = base
    elif checkpoint_id == 1:
        tokenizer = AutoTokenizer.from_pretrained(STAGE1_OUTPUT_DIR, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, STAGE1_OUTPUT_DIR)
    elif checkpoint_id == 2:
        tokenizer = AutoTokenizer.from_pretrained(STAGE2_OUTPUT_DIR, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, STAGE2_OUTPUT_DIR)

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=STAGE1_MAX_SEQ_LEN,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_alpaca_eval(model, tokenizer, checkpoint_id: int):
    eval_path = os.path.join(DATA_DIR, "alpaca_eval.json")
    data = json.loads(Path(eval_path).read_text(encoding="utf-8"))

    results = []
    for ex in tqdm(data, desc="Alpaca eval"):
        if ex.get("input", "").strip():
            prompt = ALPACA_PROMPT.format(
                instruction=ex["instruction"],
                input=ex["input"],
            )
        else:
            prompt = ALPACA_PROMPT_NO_INPUT.format(instruction=ex["instruction"])

        response = generate_response(model, tokenizer, prompt)
        results.append({
            "instruction":  ex["instruction"],
            "input":        ex.get("input", ""),
            "ground_truth": ex["output"],
            "response":     response,
            "checkpoint":   CHECKPOINT_NAMES[checkpoint_id],
        })

    out_dir = os.path.join(RESULTS_DIR, CHECKPOINT_NAMES[checkpoint_id])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "alpaca_outputs.json")
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Alpaca outputs saved to {out_path}")


def run_json_eval(model, tokenizer, checkpoint_id: int):
    eval_path = os.path.join(DATA_DIR, "json_instruct_eval.json")
    data = json.loads(Path(eval_path).read_text(encoding="utf-8"))

    results = []
    for ex in tqdm(data, desc="JSON eval"):
        prompt = JSON_PROMPT.format(
            instruction=ex["instruction"],
            input=ex.get("input", ""),
        )
        response = generate_response(model, tokenizer, prompt, max_new_tokens=512)
        results.append({
            "task_type":    ex.get("task_type", "unknown"),
            "instruction":  ex["instruction"],
            "input":        ex.get("input", ""),
            "ground_truth": ex["output"],
            "response":     response,
            "checkpoint":   CHECKPOINT_NAMES[checkpoint_id],
        })

    out_dir = os.path.join(RESULTS_DIR, CHECKPOINT_NAMES[checkpoint_id])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "json_outputs.json")
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"JSON outputs saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="all",
                        choices=["0", "1", "2", "all"])
    args = parser.parse_args()

    checkpoints = [0, 1, 2] if args.checkpoint == "all" else [int(args.checkpoint)]

    for ckpt_id in checkpoints:
        model, tokenizer = load_model(ckpt_id)
        run_alpaca_eval(model, tokenizer, ckpt_id)
        run_json_eval(model, tokenizer, ckpt_id)
        del model
        torch.cuda.empty_cache()
        print(f"Checkpoint {ckpt_id} done.\n")


if __name__ == "__main__":
    main()
