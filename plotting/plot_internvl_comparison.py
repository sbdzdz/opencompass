import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths to the summary CSV files
csv_1b16 = "./outputs/default/20250707_163415/summary/summary_20250707_163415.csv"
csv_internvl = (
    "./outputs/default/sweep_internvl_2_5_1b/summary/summary_20250526_224716.csv"
)

# Read the CSVs
df_1b16 = pd.read_csv(csv_1b16)
df_internvl = pd.read_csv(csv_internvl)


# Helper: get accuracy rows only
def get_accuracy(df):
    return df[df["metric"].str.lower() == "accuracy"]


df_1b16_acc = get_accuracy(df_1b16)
df_internvl_acc = get_accuracy(df_internvl)

# Dataset family rules (from eval_internvl.py)
dataset_families = {
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

# For each family, average all matching datasets
results = []
for family, match_fn in dataset_families.items():
    # 1B-16
    fam_rows_1b16 = df_1b16_acc[df_1b16_acc["dataset"].apply(match_fn)]
    # InternVL2.5-1B-16
    fam_rows_internvl = df_internvl_acc[df_internvl_acc["dataset"].apply(match_fn)]
    if fam_rows_1b16.empty or fam_rows_internvl.empty:
        continue
    try:
        acc1 = fam_rows_1b16["1b-16"].astype(float).mean()
    except KeyError:
        acc1 = np.nan
    try:
        acc2 = fam_rows_internvl["internvl2_5-1b-16"].astype(float).mean()
    except KeyError:
        acc2 = np.nan
    results.append((family, acc1, acc2))

# Remove any with missing values
results = [r for r in results if not (np.isnan(r[1]) or np.isnan(r[2]))]

results.sort(key=lambda x: x[2])

families = [r[0] for r in results]
acc_1b16 = [r[1] for r in results]
acc_internvl = [r[2] for r in results]

# Plot
plt.figure(figsize=(10, max(6, len(families) * 0.8)))
bar_width = 0.35
ind = np.arange(len(families)) * 1.5

plt.barh(ind, acc_1b16, bar_width, label="DMT-1B", color="#2E86AB", alpha=0.8)
plt.barh(
    ind + bar_width,
    acc_internvl,
    bar_width,
    label="InternVL2.5-1B",
    color="#A23B72",
    alpha=0.8,
)

plt.yticks(ind + bar_width / 2, families)
plt.xlabel("Accuracy (%)")
plt.title("Accuracy Comparison at 16 max new tokens")
plt.legend()
plt.tight_layout()
plt.savefig("internvl_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
