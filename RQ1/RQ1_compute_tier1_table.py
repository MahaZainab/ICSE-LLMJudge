"""
RQ1 — Tier 1 Table Value Computation
Reads all Tier 1 judge output files and computes the exact values
needed to fill Table: tab:rq1_main for CodeQA.

Outputs:
  - Console: exact numbers per model + tier mean + lowest-acc model
  - RQ1_judge_codeqa/tier1_codeqa_table_values.json
  - RQ1_judge_codeqa/tier1_codeqa_table_values.csv

Usage:
  python RQ1_compute_tier1_table.py
"""

import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

JUDGE_DIR  = "RQ1_judge_codeqa"
OUTPUT_DIR = "RQ1_judge_codeqa"
DIMS       = ["accuracy", "completeness", "clarity", "relevance"]

TIER1_MODELS = [
    ("Llama-3.2-1B-Instruct",            "llama3.2_1b_instruct_judged.json"),
    ("Qwen2.5-1.5B-Instruct",            "qwen2.5_1.5b_instruct_judged.json"),
    ("Qwen2.5-Coder-1.5B-Instruct",      "qwen2.5_coder_1.5b_instruct_judged.json"),
    ("DeepSeek-Coder-1.3B-Instruct",     "deepseek_coder_1.3b_instruct_judged.json"),
    ("SmolLM2-1.7B-Instruct",            "smollm2_1.7b_instruct_judged.json"),
    ("stablelm-2-zephyr-1_6b",           "stablelm_2_zephyr_1_6b_judged.json"),
    ("Gemma-2-2B-IT",                    "gemma_2_2b_it_judged.json"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_judged(filepath):
    """Load judge output and return only fully scored records."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid = [r for r in data if all(r.get(d) is not None for d in DIMS)]
        return valid
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def compute_scores(records):
    """Compute mean per dimension + avg across all 4 dims."""
    scores = {}
    for d in DIMS:
        vals = [r[d] for r in records if r.get(d) is not None]
        scores[d] = round(np.mean(vals), 2) if vals else None
    valid_means = [v for v in scores.values() if v is not None]
    scores["avg"] = round(np.mean(valid_means), 2) if valid_means else None
    return scores

def compute_category_breakdown(records):
    """Compute per-category mean accuracy for diagnostics."""
    by_cat = defaultdict(list)
    for r in records:
        if r.get("accuracy") is not None:
            by_cat[r["category"]].append(r["accuracy"])
    return {cat: round(np.mean(vals), 2) for cat, vals in sorted(by_cat.items())}

def main():
    print("=" * 65)
    print("RQ1 Tier 1 — CodeQA Table Values")
    print("=" * 65)

    table_rows  = []
    missing     = []

    for model_name, filename in TIER1_MODELS:
        filepath = os.path.join(JUDGE_DIR, filename)
        records  = load_judged(filepath)

        if records is None:
            print(f"\n  [{model_name}]")
            print(f"  FILE NOT FOUND: {filepath}")
            missing.append(model_name)
            table_rows.append({
                "model": model_name,
                "n_records": 0,
                "accuracy": None, "completeness": None,
                "clarity": None,  "relevance": None,
                "avg": None,
                "status": "missing"
            })
            continue

        scores = compute_scores(records)
        cats   = compute_category_breakdown(records)

        print(f"\n  [{model_name}]  n={len(records)}")
        print(f"  Acc={scores['accuracy']}  "
              f"Comp={scores['completeness']}  "
              f"Clar={scores['clarity']}  "
              f"Rel={scores['relevance']}  "
              f"Avg={scores['avg']}")
        print(f"  Category breakdown (accuracy): {cats}")

        table_rows.append({
            "model":        model_name,
            "n_records":    len(records),
            "accuracy":     scores["accuracy"],
            "completeness": scores["completeness"],
            "clarity":      scores["clarity"],
            "relevance":    scores["relevance"],
            "avg":          scores["avg"],
            "status":       "complete"
        })

    complete = [r for r in table_rows if r["status"] == "complete"]

    print("\n" + "-" * 65)
    if complete:
        tier_mean = {}
        for d in DIMS + ["avg"]:
            vals = [r[d] for r in complete if r[d] is not None]
            tier_mean[d] = round(np.mean(vals), 2) if vals else None

        print(f"  Tier 1 mean  (n={len(complete)} models)")
        print(f"  Acc={tier_mean['accuracy']}  "
              f"Comp={tier_mean['completeness']}  "
              f"Clar={tier_mean['clarity']}  "
              f"Rel={tier_mean['relevance']}  "
              f"Avg={tier_mean['avg']}")
    else:
        tier_mean = {d: None for d in DIMS + ["avg"]}
        print("  Tier 1 mean: no complete models yet.")

    print()
    scored = [r for r in complete if r["accuracy"] is not None]
    if scored:
        lowest = min(scored, key=lambda r: r["accuracy"])
        print(f"  Lowest-Acc model (*): {lowest['model']}")
        print(f"  Acc={lowest['accuracy']}  "
              f"Comp={lowest['completeness']}  "
              f"Clar={lowest['clarity']}  "
              f"Rel={lowest['relevance']}  "
              f"Avg={lowest['avg']}")
        print(f"  → This model is selected for human annotation in RQ2.")
    else:
        lowest = None
        print("  Lowest-Acc model: not determinable yet.")

    if missing:
        print(f"\n  Still pending: {missing}")

    output = {
        "tier": "Tier 1 — Sub-1.5B",
        "models": table_rows,
        "tier_mean": tier_mean,
        "lowest_acc_model": lowest["model"] if lowest else None,
        "lowest_acc_scores": {
            d: lowest[d] for d in DIMS + ["avg"]
        } if lowest else None
    }

    json_path = os.path.join(OUTPUT_DIR, "tier1_codeqa_table_values.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON saved: {json_path}")

    rows_for_csv = []
    for r in table_rows:
        rows_for_csv.append({
            "Model":         r["model"],
            "N":             r["n_records"],
            "Accuracy":      r["accuracy"]     if r["accuracy"]     is not None else "--",
            "Completeness":  r["completeness"] if r["completeness"] is not None else "--",
            "Clarity":       r["clarity"]      if r["clarity"]      is not None else "--",
            "Relevance":     r["relevance"]    if r["relevance"]    is not None else "--",
            "Avg":           r["avg"]          if r["avg"]          is not None else "--",
            "Status":        r["status"],
        })
    rows_for_csv.append({
        "Model": "Tier 1 mean",
        "N": len(complete),
        "Accuracy":     tier_mean["accuracy"]     if tier_mean["accuracy"]     is not None else "--",
        "Completeness": tier_mean["completeness"] if tier_mean["completeness"] is not None else "--",
        "Clarity":      tier_mean["clarity"]      if tier_mean["clarity"]      is not None else "--",
        "Relevance":    tier_mean["relevance"]    if tier_mean["relevance"]    is not None else "--",
        "Avg":          tier_mean["avg"]          if tier_mean["avg"]          is not None else "--",
        "Status": "mean",
    })
    if lowest:
        rows_for_csv.append({
            "Model": f"Lowest-Acc (*): {lowest['model']}",
            "N": lowest["n_records"],
            "Accuracy":     lowest["accuracy"],
            "Completeness": lowest["completeness"],
            "Clarity":      lowest["clarity"],
            "Relevance":    lowest["relevance"],
            "Avg":          lowest["avg"],
            "Status": "selected_for_rq2",
        })

    csv_path = os.path.join(OUTPUT_DIR, "tier1_codeqa_table_values.csv")
    pd.DataFrame(rows_for_csv).to_csv(csv_path, index=False)
    print(f"  CSV saved:  {csv_path}")

    print("\n" + "=" * 65)
    print("Copy these values directly into tab:rq1_main in Overleaf.")
    print("=" * 65)

if __name__ == "__main__":
    main()
