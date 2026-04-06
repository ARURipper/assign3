"""
eval/generate_figures.py
Read evaluation logs and produce all tables and figures for the blog post.

Run:
    python eval/generate_figures.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LOGS_DIR, RESULTS_DIR

FIG_DIR = os.path.join(LOGS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CKPT_LABELS = {
    "checkpoint_0_baseline":  "Checkpoint 0\n(Baseline)",
    "checkpoint_1_alpaca":    "Checkpoint 1\n(Alpaca)",
    "checkpoint_2_json":      "Checkpoint 2\n(JSON)",
}
CKPTS = list(CKPT_LABELS.keys())


def load_json_eval():
    path = os.path.join(LOGS_DIR, "json_eval_results.json")
    if not os.path.exists(path):
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_judge_results():
    path = os.path.join(LOGS_DIR, "judge_results.json")
    if not os.path.exists(path):
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_json_validity(json_eval: dict):
    vals = [json_eval.get(c, {}).get("json_validity_rate", 0) for c in CKPTS]
    labels = [CKPT_LABELS[c] for c in CKPTS]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, [v * 100 for v in vals], color=["#aec6cf", "#6baed6", "#2171b5"])
    plt.title("JSON Validity Rate by Checkpoint", fontsize=13)
    plt.ylabel("Validity Rate (%)")
    plt.ylim(0, 100)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1%}", ha="center", fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "json_validity.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_json_exact_match(json_eval: dict):
    vals = [json_eval.get(c, {}).get("exact_match_rate", 0) for c in CKPTS]
    labels = [CKPT_LABELS[c] for c in CKPTS]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, [v * 100 for v in vals], color=["#aec6cf", "#6baed6", "#2171b5"])
    plt.title("JSON Exact Match Rate by Checkpoint", fontsize=13)
    plt.ylabel("Exact Match Rate (%)")
    plt.ylim(0, 100)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1%}", ha="center", fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "json_exact_match.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_judge_win_rates(judge_results: dict):
    pairs = ["0_vs_1", "1_vs_2", "0_vs_2"]
    pair_labels = ["Ckpt 0 vs 1\n(Alpaca gain?)", "Ckpt 1 vs 2\n(Forgetting?)", "Ckpt 0 vs 2\n(Overall?)"]

    for eval_type in ["alpaca", "json"]:
        win_a, win_b, tie_ = [], [], []
        for pair in pairs:
            agg = judge_results.get(pair, {}).get(eval_type, {}).get("aggregate", {})
            win_a.append(agg.get("win_rate_a", 0) * 100)
            win_b.append(agg.get("win_rate_b", 0) * 100)
            tie_.append(agg.get("tie_rate", 0) * 100)

        x = np.arange(len(pairs))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, win_a, width, label="Win A (earlier)", color="#6baed6")
        ax.bar(x,          tie_,  width, label="Tie",             color="#bdbdbd")
        ax.bar(x + width, win_b, width, label="Win B (later)",   color="#2171b5")

        ax.set_title(f"Judge Win Rates — {eval_type.upper()} Evaluation", fontsize=13)
        ax.set_ylabel("Rate (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels)
        ax.legend()
        ax.set_ylim(0, 100)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, f"judge_win_rates_{eval_type}.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")


def plot_task_breakdown(json_eval: dict):
    """Heatmap of JSON validity by task type and checkpoint."""
    task_types = [
        "json_extraction", "schema_constrained_generation",
        "exact_label_classification", "json_repair", "tool_call_generation"
    ]
    short_labels = ["Extraction", "Schema Gen", "Classification", "Repair", "Tool Call"]

    data_matrix = []
    for ckpt in CKPTS:
        row = []
        by_task = json_eval.get(ckpt, {}).get("by_task", {})
        for task in task_types:
            row.append(by_task.get(task, {}).get("validity_rate", 0))
        data_matrix.append(row)

    matrix = np.array(data_matrix)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(task_types)))
    ax.set_xticklabels(short_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(CKPTS)))
    ax.set_yticklabels([CKPT_LABELS[c] for c in CKPTS])

    for i in range(len(CKPTS)):
        for j in range(len(task_types)):
            ax.text(j, i, f"{matrix[i,j]:.0%}", ha="center", va="center",
                    color="white" if matrix[i,j] > 0.6 else "black", fontsize=9)

    plt.colorbar(im, ax=ax, label="JSON Validity Rate")
    ax.set_title("JSON Validity by Task Type and Checkpoint", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "task_breakdown_heatmap.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def generate_summary_csv(json_eval: dict, judge_results: dict):
    rows = []
    for ckpt in CKPTS:
        jeval = json_eval.get(ckpt, {})
        rows.append({
            "Checkpoint":      ckpt,
            "JSON Validity":   jeval.get("json_validity_rate", "-"),
            "Exact Match":     jeval.get("exact_match_rate", "-"),
            "Avg Field F1":    jeval.get("avg_field_f1", "-"),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(LOGS_DIR, "summary_table.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    print(df.to_string(index=False))


def main():
    json_eval    = load_json_eval()
    judge_results = load_judge_results()

    if json_eval:
        plot_json_validity(json_eval)
        plot_json_exact_match(json_eval)
        plot_task_breakdown(json_eval)
        generate_summary_csv(json_eval, judge_results)
    else:
        print("No JSON eval results found. Run eval/run_json_eval.py first.")

    if judge_results:
        plot_judge_win_rates(judge_results)
    else:
        print("No judge results found. Run eval/judge_eval.py first.")

    print("\nAll figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
