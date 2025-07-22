import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import re

DATASET_FAMILIES = {
    "ceval": lambda name: name.startswith("ceval-"),
    "C3": lambda name: name == "C3",
    "cmmlu": lambda name: name.startswith("cmmlu-"),
    "GaokaoBench": lambda name: name.startswith("GaokaoBench"),
    "gsm8k": lambda name: name == "gsm8k",
    "hellaswag": lambda name: name == "hellaswag",
    "humaneval": lambda name: name == "openai_humaneval",
    "math": lambda name: name == "math",
    "mbpp_cn": lambda name: name == "mbpp_cn",
    "sanitized_mbpp": lambda name: name == "sanitized_mbpp",
    "mmlu": lambda name: name.startswith("lukaemon_mmlu_") or name == "mmlu",
    "nq": lambda name: name == "nq",
    "race": lambda name: name.startswith("race-"),
    "TheoremQA": lambda name: name == "TheoremQA",
    "triviaqa": lambda name: name == "triviaqa",
    "winogrande": lambda name: name == "winogrande",
}
PREFERRED_METRICS = ["score", "accuracy", "humaneval_pass@1"]

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", help="Path to summary CSV file")
parser.add_argument("--outdir", default="figures", help="Directory to save plots")
args = parser.parse_args()

csv_file = args.csv_file
outdir = Path(args.outdir)
outdir.mkdir(exist_ok=True)

df = pd.read_csv(csv_file)
meta_cols = ["dataset", "version", "metric", "mode"]
model_cols = [col for col in df.columns if col not in meta_cols]

# Parse model columns into (model, max_tokens)
model_token_map = {}
for col in model_cols:
    m = re.match(r"(.+)-(\d+)$", col)
    if m:
        model = m.group(1)
        tokens = int(m.group(2))
        if model not in model_token_map:
            model_token_map[model] = []
        model_token_map[model].append((tokens, col))
# Sort token values for each model
for model in model_token_map:
    model_token_map[model].sort()

# Helper: for a given dataset family, get the best metric rows
family_results = {}
for family, match_fn in DATASET_FAMILIES.items():
    fam_rows = df[df["dataset"].apply(match_fn)]
    fam_metric_rows = None
    for metric in PREFERRED_METRICS:
        rows = fam_rows[fam_rows["metric"] == metric]
        if not rows.empty:
            fam_metric_rows = rows
            break
    if fam_metric_rows is not None and not fam_metric_rows.empty:
        family_results[family] = fam_metric_rows

MODEL_NAME_MAP = {
    "internvl2_5-1b": "InternVL 2.5 (1B)",
    "internvl2_5-2b": "InternVL 2.5 (2B)",
    "internvl3-1b-instruct": "InternVL 3 (1B)",
    "internvl3-2b-instruct": "InternVL 3 (2B)",
}


def get_model_label(model):
    return MODEL_NAME_MAP.get(model, model)


# --- Plot: average over all dataset families ---
if family_results:
    # For each model, for each token, average over all families
    avg_per_model_token = {}
    for model, token_cols in model_token_map.items():
        avg_per_model_token[model] = []
        for tokens, col in token_cols:
            vals = []
            for fam, fam_rows in family_results.items():
                if col in fam_rows:
                    vals.extend(fam_rows[col].astype(float).values)
            if vals:
                avg_per_model_token[model].append((tokens, sum(vals) / len(vals)))
            else:
                avg_per_model_token[model].append((tokens, float("nan")))
    plt.figure(figsize=(10, 6))
    for model, vals in avg_per_model_token.items():
        xs, ys = zip(*sorted(vals))
        plt.plot(xs, ys, marker="o", label=get_model_label(model))
    plt.ylabel("Average Score Across All Datasets")
    plt.xlabel("Max New Tokens")
    plt.title("Average Score Across All Datasets")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "score_all_datasets.png", dpi=300)
    plt.close()

# --- Plot: one plot per dataset family ---
for family, fam_rows in family_results.items():
    # Build per-model, per-token values for this family
    model_token_vals = {}
    for model, token_cols in model_token_map.items():
        model_token_vals[model] = []
        for tokens, col in token_cols:
            if col in fam_rows:
                v = fam_rows[col].astype(float).mean()
                model_token_vals[model].append((tokens, v))
            else:
                model_token_vals[model].append((tokens, float("nan")))
    plt.figure(figsize=(10, 6))
    for model, vals in model_token_vals.items():
        xs, ys = zip(*sorted(vals))
        plt.plot(xs, ys, marker="o", label=get_model_label(model))
    plt.ylabel(f"Score: {family}")
    plt.xlabel("Max New Tokens")
    plt.title(f"Score: {family}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"score_{family}.png", dpi=300)
    plt.close()
