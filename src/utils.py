"""
src/utils.py
Shared utility functions used across the pipeline.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── JSON helpers ───────────────────────────────────────────────────────────────

def extract_json(text: str) -> Tuple[Optional[Any], bool]:
    """
    Try to extract a valid JSON object or array from text.
    Handles markdown code fences and partial JSON.
    Returns (parsed_object, is_valid).
    """
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass

    for pattern in [r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', r'\[[^\[\]]*\]']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), True
            except json.JSONDecodeError:
                continue

    return None, False


def is_valid_json(text: str) -> bool:
    _, valid = extract_json(text)
    return valid


def normalize_json_string(text: str) -> str:
    obj, valid = extract_json(text)
    if valid:
        return json.dumps(obj, sort_keys=True)
    return text.strip()


# ── File I/O ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str, data: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def save_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def timestamp_ms() -> int:
    return int(time.time() * 1000)


# ── Prompt formatting ──────────────────────────────────────────────────────────

def format_alpaca_prompt(instruction: str, input_text: str = "", include_output: bool = False, output: str = "") -> str:
    if input_text.strip():
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides "
            "further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )
    if include_output:
        prompt += output
    return prompt


def format_json_prompt(instruction: str, input_text: str = "", include_output: bool = False, output: str = "") -> str:
    if input_text.strip():
        prompt = (
            "Below is an instruction describing a structured-output task. "
            "Respond ONLY with valid JSON, no explanation or markdown.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction describing a structured-output task. "
            "Respond ONLY with valid JSON, no explanation or markdown.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )
    if include_output:
        prompt += output
    return prompt


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def compute_field_f1(pred_obj: Any, gold_obj: Any) -> float:
    """Field-level F1 between two JSON objects."""
    if not isinstance(pred_obj, dict) or not isinstance(gold_obj, dict):
        return float(str(pred_obj).lower().strip() == str(gold_obj).lower().strip())

    gold_keys = set(gold_obj.keys())
    pred_keys = set(pred_obj.keys())
    if not gold_keys:
        return 1.0

    matched = 0.0
    for key in gold_keys:
        if key in pred_keys:
            gv = str(gold_obj[key]).lower().strip()
            pv = str(pred_obj[key]).lower().strip()
            if gv == pv:
                matched += 1.0
            elif gv in pv or pv in gv:
                matched += 0.5

    precision = matched / len(pred_keys) if pred_keys else 0.0
    recall    = matched / len(gold_keys)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(pred: str, gold: str) -> bool:
    """Exact match after JSON normalization."""
    pred_norm = normalize_json_string(pred)
    gold_norm = normalize_json_string(gold)
    return pred_norm == gold_norm


# ── Logging ────────────────────────────────────────────────────────────────────

def log_run(log_dir: str, run_name: str, data: Dict[str, Any]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{run_name}_{timestamp_ms()}.json")
    save_json(path, data)
    print(f"Logged run to {path}")
