"""
data/prepare_alpaca.py
Download, clean, and split Alpaca data into train / eval splits.
Run: python data/prepare_alpaca.py
"""
import json
import os
import random
from pathlib import Path

import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DATA_DIR, ALPACA_TRAIN_SIZE, ALPACA_EVAL_SIZE, SEED)

ALPACA_URL = (
    "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/"
    "main/alpaca_data_cleaned.json"
)

random.seed(SEED)


def download_alpaca(save_path: str) -> list:
    print(f"Downloading Alpaca data from {ALPACA_URL} ...")
    response = requests.get(ALPACA_URL, timeout=60)
    response.raise_for_status()
    data = response.json()
    Path(save_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Saved {len(data)} examples to {save_path}")
    return data


def clean_example(ex: dict) -> dict | None:
    """Return cleaned example or None if it should be dropped."""
    instruction = (ex.get("instruction") or "").strip()
    output      = (ex.get("output") or "").strip()
    inp         = (ex.get("input") or "").strip()

    # Drop malformed or too-short examples
    if len(instruction) < 10 or len(output) < 5:
        return None
    if "[Your" in output or "TODO" in output:
        return None

    return {
        "instruction": instruction,
        "input":       inp,
        "output":      output,
    }


def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "alpaca_raw.json")

    # Download if needed
    if os.path.exists(raw_path):
        print(f"Raw Alpaca data already exists at {raw_path}, skipping download.")
        raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    else:
        raw = download_alpaca(raw_path)

    # Clean
    cleaned = [c for ex in raw if (c := clean_example(ex)) is not None]
    print(f"After cleaning: {len(cleaned)} examples (from {len(raw)})")

    # Shuffle deterministically
    random.shuffle(cleaned)

    # Split
    total_needed = ALPACA_TRAIN_SIZE + ALPACA_EVAL_SIZE
    if len(cleaned) < total_needed:
        raise ValueError(
            f"Not enough clean examples ({len(cleaned)}) for "
            f"train ({ALPACA_TRAIN_SIZE}) + eval ({ALPACA_EVAL_SIZE})."
        )

    train = cleaned[:ALPACA_TRAIN_SIZE]
    eval_ = cleaned[ALPACA_TRAIN_SIZE:ALPACA_TRAIN_SIZE + ALPACA_EVAL_SIZE]

    # Save
    train_path = os.path.join(DATA_DIR, "alpaca_train.json")
    eval_path  = os.path.join(DATA_DIR, "alpaca_eval.json")
    Path(train_path).write_text(json.dumps(train, indent=2), encoding="utf-8")
    Path(eval_path).write_text(json.dumps(eval_, indent=2), encoding="utf-8")

    print(f"Train: {len(train)} examples → {train_path}")
    print(f"Eval:  {len(eval_)} examples → {eval_path}")
    print("Done.")


if __name__ == "__main__":
    prepare()
