import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

# File paths
csv_dmt = "outputs/default/20250714_164426/summary/summary_20250714_164426.csv"
csv_internvl = "outputs/default/sweep_all/summary/summary_20250706_230044.csv"

# Model columns to compare (split into 1B and 2B)
model_columns_1b = [
    ("DMT-1B-10B", "1b_10b-32", csv_dmt),
    ("DMT-1B-20B", "1b_20b-32", csv_dmt),
    ("InternVL2.5-1B", "internvl2_5-1b-instruct-32", csv_internvl),
    ("InternVL3-1B", "internvl3-1b-instruct-32", csv_internvl),
]
model_columns_2b = [
    ("DMT-2B-10B", "2b_10b-32", csv_dmt),
    ("DMT-2B-20B", "2b_20b-32", csv_dmt),
    ("InternVL2.5-2B", "internvl2_5-2b-instruct-32", csv_internvl),
    ("InternVL3-2B", "internvl3-2b-instruct-32", csv_internvl),
]

# Dataset family rules
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

# Read both CSVs
dfs = {
    csv_dmt: pd.read_csv(csv_dmt),
    csv_internvl: pd.read_csv(csv_internvl),
}

# Helper: for a given dataset family, get the best metric rows
family_results = {}
for family, match_fn in DATASET_FAMILIES.items():
    fam_rows = []
    for df in dfs.values():
        fam_rows.append(df[df["dataset"].apply(match_fn)])
    fam_rows = pd.concat(fam_rows)
    fam_metric_rows = None
    for metric in PREFERRED_METRICS:
        rows = fam_rows[fam_rows["metric"] == metric]
        if not rows.empty:
            fam_metric_rows = rows
            break
    if fam_metric_rows is not None and not fam_metric_rows.empty:
        family_results[family] = fam_metric_rows


# Define a fixed color mapping for each model label
MODEL_COLOR_MAP = {
    "DMT-1B-10B": "#2E86AB",  # blue
    "DMT-1B-20B": "#3CB371",  # green
    "InternVL2.5-1B": "#F18F01",  # orange
    "InternVL3-1B": "#A23B72",  # magenta
    "DMT-2B-20B": "#2E86AB",  # blue (same as 1B-10B for DMT)
    "DMT-2B-40B": "#3CB371",  # green (same as 1B-20B for DMT)
    "InternVL2.5-2B": "#F18F01",  # orange
    "InternVL3-2B": "#A23B72",  # magenta
}


def plot_group(model_columns, plot_filename, plot_title):
    plot_families = []
    plot_data = []
    for family, fam_rows in family_results.items():
        vals = []
        for label, col, which_csv in model_columns:
            if col in fam_rows:
                v = fam_rows[col].replace("-", np.nan).astype(float).mean()
            else:
                v = np.nan
            vals.append(v)
        # Always append, even if all are NaN, to keep bar positions consistent
        plot_families.append(family)
        plot_data.append(vals)
    if not plot_families:
        return
    model_labels = [label for label, _, _ in model_columns]
    plot_data_arr = np.array(plot_data)
    ind = np.arange(len(plot_families)) * 1.5
    bar_width = 0.18 if len(model_labels) > 3 else 0.25
    plt.figure(figsize=(10, max(6, len(plot_families) * 0.8)))
    for i, label in enumerate(model_labels):
        color = MODEL_COLOR_MAP.get(label, None)
        plt.barh(
            ind + i * bar_width,
            plot_data_arr[:, i],
            bar_width,
            label=label,
            color=color,
        )
    plt.yticks(ind + bar_width * (len(model_labels) - 1) / 2, plot_families)
    plt.xlabel("Accuracy / Score (%)")
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{plot_filename}", dpi=300, bbox_inches="tight")
    plt.close()


# Plot 1B models
plot_group(
    model_columns_1b,
    "dmt_vs_internvl_1b_comparison.png",
    "DMT vs InternVL Model Comparison (1B, 32 max tokens)",
)

# Plot 2B models
plot_group(
    model_columns_2b,
    "dmt_vs_internvl_2b_comparison.png",
    "DMT vs InternVL Model Comparison (2B, 32 max tokens)",
)
