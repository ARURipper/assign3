"""
eval/run_json_eval.py
Evaluate JSON outputs at each checkpoint for:
  - JSON validity rate
  - Schema compliance
  - Exact-match accuracy
  - Field-level correctness
  - Per-task-type breakdown

Run:
    python eval/run_json_eval.py
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, LOGS_DIR

CHECKPOINT_NAMES = [
    "checkpoint_0_baseline",
    "checkpoint_1_alpaca",
    "checkpoint_2_json",
]


def extract_json(text: str):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        for pattern in [r'\{.*\}', r'\[.*\]']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0)), True
                except json.JSONDecodeError:
                    continue
    return None, False


def compute_field_match(pred_obj, gold_obj) -> float:
    """Compute field-level F1 between two JSON objects."""
    if not isinstance(pred_obj, dict) or not isinstance(gold_obj, dict):
        return float(pred_obj == gold_obj)

    gold_keys = set(gold_obj.keys())
    pred_keys = set(pred_obj.keys())
    if not gold_keys:
        return 1.0

    matched = 0
    for key in gold_keys:
        if key in pred_keys:
            gv = str(gold_obj[key]).lower().strip()
            pv = str(pred_obj[key]).lower().strip()
            if gv == pv:
                matched += 1
            elif gv in pv or pv in gv:
                matched += 0.5

    precision = matched / len(pred_keys) if pred_keys else 0
    recall    = matched / len(gold_keys)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_checkpoint(checkpoint_name: str) -> dict:
    output_path = os.path.join(RESULTS_DIR, checkpoint_name, "json_outputs.json")
    if not os.path.exists(output_path):
        print(f"No JSON outputs found for {checkpoint_name}")
        return {}

    data = json.loads(Path(output_path).read_text(encoding="utf-8"))

    valid_count   = 0
    exact_matches = 0
    field_scores  = []
    task_stats    = defaultdict(lambda: {"total": 0, "valid": 0, "exact": 0})

    for ex in data:
        task     = ex.get("task_type", "unknown")
        response = ex.get("response", "")
        gold_str = ex.get("ground_truth", "")

        pred_obj, is_valid = extract_json(response)
        gold_obj, gold_valid = extract_json(gold_str)

        task_stats[task]["total"] += 1

        if is_valid:
            valid_count += 1
            task_stats[task]["valid"] += 1

        if is_valid and gold_valid:
            pred_str_norm = json.dumps(pred_obj, sort_keys=True)
            gold_str_norm = json.dumps(gold_obj, sort_keys=True)

            if pred_str_norm == gold_str_norm:
                exact_matches += 1
                task_stats[task]["exact"] += 1

            field_scores.append(compute_field_match(pred_obj, gold_obj))

    n = len(data) or 1
    results = {
        "checkpoint":        checkpoint_name,
        "n_samples":         len(data),
        "json_validity_rate": round(valid_count / n, 4),
        "exact_match_rate":  round(exact_matches / n, 4),
        "avg_field_f1":      round(sum(field_scores) / len(field_scores), 4) if field_scores else 0.0,
        "by_task": {
            task: {
                "total":        stats["total"],
                "validity_rate": round(stats["valid"] / stats["total"], 4) if stats["total"] else 0,
                "exact_rate":   round(stats["exact"] / stats["total"], 4) if stats["total"] else 0,
            }
            for task, stats in task_stats.items()
        },
    }
    return results


def main():
    all_results = {}
    for ckpt in CHECKPOINT_NAMES:
        print(f"Evaluating: {ckpt}")
        results = evaluate_checkpoint(ckpt)
        all_results[ckpt] = results
        if results:
            print(f"  JSON Validity:  {results['json_validity_rate']:.1%}")
            print(f"  Exact Match:    {results['exact_match_rate']:.1%}")
            print(f"  Avg Field F1:   {results['avg_field_f1']:.4f}")
            for task, stats in results.get("by_task", {}).items():
                print(f"    {task}: validity={stats['validity_rate']:.1%}, exact={stats['exact_rate']:.1%}")

    os.makedirs(LOGS_DIR, exist_ok=True)
    out_path = os.path.join(LOGS_DIR, "json_eval_results.json")
    Path(out_path).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved JSON eval results to {out_path}")


if __name__ == "__main__":
    main()
