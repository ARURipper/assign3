"""
eval/judge_eval.py
LLM-as-a-Judge evaluation comparing model outputs across checkpoint pairs.

Pairwise comparisons:
  - Checkpoint 0 vs Checkpoint 1  (did Alpaca training help?)
  - Checkpoint 1 vs Checkpoint 2  (did JSON training hurt general ability?)
  - Checkpoint 0 vs Checkpoint 2  (overall improvement?)

Run:
    python eval/judge_eval.py --pair 0_vs_1
    python eval/judge_eval.py --pair 1_vs_2
    python eval/judge_eval.py --pair all
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    JUDGE_MODEL, RESULTS_DIR, LOGS_DIR, PROMPTS_DIR,
    JUDGE_TEMPERATURE, JUDGE_MAX_NEW_TOKENS, JUDGE_EVAL_SAMPLE, SEED,
)

set_seed(SEED)

CHECKPOINT_NAMES = {
    0: "checkpoint_0_baseline",
    1: "checkpoint_1_alpaca",
    2: "checkpoint_2_json",
}

PAIRS = {
    "0_vs_1": (0, 1),
    "1_vs_2": (1, 2),
    "0_vs_2": (0, 2),
}


def load_judge_prompt() -> str:
    path = os.path.join(PROMPTS_DIR, "judge_eval.txt")
    return Path(path).read_text(encoding="utf-8")


def parse_judge_output(text: str) -> dict:
    """Extract structured fields from judge response."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    result = {}
    for field in ["instruction_following", "correctness", "clarity",
                  "completeness", "structured_output_validity", "hallucination_risk"]:
        match = re.search(rf'"{field}"\s*:\s*(\d)', text)
        if match:
            result[field] = int(match.group(1))

    winner_match = re.search(r'"winner"\s*:\s*"([AB]|Tie)"', text)
    if winner_match:
        result["winner"] = winner_match.group(1)

    just_match = re.search(r'"justification"\s*:\s*"([^"]+)"', text)
    if just_match:
        result["justification"] = just_match.group(1)

    return result


def run_pair_evaluation(
    pipe,
    judge_prompt_template: str,
    ckpt_a: int,
    ckpt_b: int,
    eval_type: str = "alpaca",
) -> list:
    name_a = CHECKPOINT_NAMES[ckpt_a]
    name_b = CHECKPOINT_NAMES[ckpt_b]

    file_name = f"{eval_type}_outputs.json"
    path_a = os.path.join(RESULTS_DIR, name_a, file_name)
    path_b = os.path.join(RESULTS_DIR, name_b, file_name)

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print(f"Missing outputs for {name_a} or {name_b} — skipping {eval_type}")
        return []

    data_a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    data_b = json.loads(Path(path_b).read_text(encoding="utf-8"))

    # Sample for efficiency
    import random
    random.seed(SEED)
    indices = random.sample(range(min(len(data_a), len(data_b))),
                             min(JUDGE_EVAL_SAMPLE, len(data_a)))

    results = []
    for i in tqdm(indices, desc=f"Judge {name_a} vs {name_b} [{eval_type}]"):
        ex_a = data_a[i]
        ex_b = data_b[i]

        prompt = judge_prompt_template.format(
            prompt_id=f"{eval_type}_{i:04d}",
            checkpoint_a=name_a,
            checkpoint_b=name_b,
            instruction=ex_a["instruction"],
            input=ex_a.get("input", ""),
            response_a=ex_a["response"],
            response_b=ex_b["response"],
        )

        try:
            output = pipe(
                prompt,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                temperature=JUDGE_TEMPERATURE,
                do_sample=False,
                return_full_text=False,
            )
            raw_text = output[0]["generated_text"].strip()
            parsed = parse_judge_output(raw_text)
        except Exception as e:
            print(f"Judge error on sample {i}: {e}")
            parsed = {}
            raw_text = ""

        results.append({
            "prompt_id":    f"{eval_type}_{i:04d}",
            "checkpoint_a": name_a,
            "checkpoint_b": name_b,
            "eval_type":    eval_type,
            "instruction":  ex_a["instruction"],
            "response_a":   ex_a["response"],
            "response_b":   ex_b["response"],
            "judge_raw":    raw_text,
            "judge_parsed": parsed,
        })

    return results


def aggregate_results(results: list) -> dict:
    """Compute win rates and average scores."""
    wins_a = sum(1 for r in results if r["judge_parsed"].get("winner") == "A")
    wins_b = sum(1 for r in results if r["judge_parsed"].get("winner") == "B")
    ties   = sum(1 for r in results if r["judge_parsed"].get("winner") == "Tie")
    n = len(results) or 1

    score_fields = [
        "instruction_following", "correctness", "clarity",
        "completeness", "structured_output_validity", "hallucination_risk"
    ]

    avg_scores_a = {}
    avg_scores_b = {}
    for field in score_fields:
        scores_a = [r["judge_parsed"].get("response_a_scores", {}).get(field, 0)
                    for r in results if "response_a_scores" in r.get("judge_parsed", {})]
        scores_b = [r["judge_parsed"].get("response_b_scores", {}).get(field, 0)
                    for r in results if "response_b_scores" in r.get("judge_parsed", {})]
        if scores_a:
            avg_scores_a[field] = round(sum(scores_a) / len(scores_a), 3)
        if scores_b:
            avg_scores_b[field] = round(sum(scores_b) / len(scores_b), 3)

    return {
        "n_samples":      n,
        "wins_a":         wins_a,
        "wins_b":         wins_b,
        "ties":           ties,
        "win_rate_a":     round(wins_a / n, 4),
        "win_rate_b":     round(wins_b / n, 4),
        "tie_rate":       round(ties / n, 4),
        "avg_scores_a":   avg_scores_a,
        "avg_scores_b":   avg_scores_b,
    }


def load_judge_pipeline():
    print(f"Loading judge model: {JUDGE_MODEL}")
    pipe = pipeline(
        "text-generation",
        model=JUDGE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="all",
                        choices=["0_vs_1", "1_vs_2", "0_vs_2", "all"])
    parser.add_argument("--eval_type", default="both",
                        choices=["alpaca", "json", "both"])
    args = parser.parse_args()

    pipe = load_judge_pipeline()
    judge_prompt_template = load_judge_prompt()

    pairs_to_run = PAIRS if args.pair == "all" else {args.pair: PAIRS[args.pair]}
    eval_types = ["alpaca", "json"] if args.eval_type == "both" else [args.eval_type]

    os.makedirs(LOGS_DIR, exist_ok=True)
    all_judge_results = {}

    for pair_name, (ckpt_a, ckpt_b) in pairs_to_run.items():
        all_judge_results[pair_name] = {}
        for et in eval_types:
            results = run_pair_evaluation(pipe, judge_prompt_template, ckpt_a, ckpt_b, et)
            if results:
                agg = aggregate_results(results)
                all_judge_results[pair_name][et] = {
                    "aggregate": agg,
                    "samples":   results,
                }
                print(f"\n{pair_name} [{et}]:")
                print(f"  Win A ({CHECKPOINT_NAMES[ckpt_a]}): {agg['win_rate_a']:.1%}")
                print(f"  Win B ({CHECKPOINT_NAMES[ckpt_b]}): {agg['win_rate_b']:.1%}")
                print(f"  Ties: {agg['tie_rate']:.1%}")

    out_path = os.path.join(LOGS_DIR, "judge_results.json")
    Path(out_path).write_text(json.dumps(all_judge_results, indent=2), encoding="utf-8")
    print(f"\nSaved all judge results to {out_path}")


if __name__ == "__main__":
    main()
