import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    # File paths
    csv_dmt = Path(
        "outputs/default/20250714_164426/summary/summary_20250714_164426.csv"
    )
    csv_internvl = Path(
        "outputs/default/sweep_all_32/summary/summary_20250721_213714.csv"
    )

    # Model columns to compare (split into 1B and 2B)
    model_columns_1b = [
        ("DMT-1B-10B", "1b_10b-32", csv_dmt),
        ("DMT-1B-20B", "1b_20b-32", csv_dmt),
        ("InternVL2.5-1B", "internvl2_5-1b-32", csv_internvl),
        ("InternVL3-1B", "internvl3-1b-instruct-32", csv_internvl),
    ]
    model_columns_2b = [
        ("DMT-2B-10B", "2b_10b-32", csv_dmt),
        ("DMT-2B-20B", "2b_20b-32", csv_dmt),
        ("InternVL2.5-2B", "internvl2_5-2b-32", csv_internvl),
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

    # Build family_results using the new function
    family_results = get_family_results(DATASET_FAMILIES, PREFERRED_METRICS, dfs)

    plot_group(
        model_columns_1b,
        figures_dir / "dmt_vs_internvl_1b_comparison.png",
        "DMT vs InternVL Model Comparison (1B, 32 max tokens)",
        family_results,
    )

    plot_group(
        model_columns_2b,
        figures_dir / "dmt_vs_internvl_2b_comparison.png",
        "DMT vs InternVL Model Comparison (2B, 32 max tokens)",
        family_results,
    )


def get_family_results(DATASET_FAMILIES, PREFERRED_METRICS, dfs):
    family_results = {}
    for family, match_fn in DATASET_FAMILIES.items():
        fam_metric_rows = None
        for metric in PREFERRED_METRICS:
            fam_rows = []
            for which_csv, df in dfs.items():
                # Find all rows for this family and metric
                rows = df[df["metric"].str.lower() == metric.lower()]
                rows = rows[rows["dataset"].apply(match_fn)]
                if not rows.empty:
                    fam_rows.append(rows.set_index("dataset"))
            if fam_rows:
                fam_metric_rows = pd.concat(fam_rows, axis=0)
                break
        if fam_metric_rows is not None:
            family_results[family] = fam_metric_rows
    return family_results


def plot_group(model_columns, plot_filename, plot_title, family_results):
    legend_order = [
        "InternVL3-1B",
        "InternVL2.5-1B",
        "DMT-1B-20B",
        "DMT-1B-10B",
        "InternVL3-2B",
        "InternVL2.5-2B",
        "DMT-2B-20B",
        "DMT-2B-10B",
    ]
    model_labels = [label for label, _, _ in model_columns]
    legend_order = [label for label in legend_order if label in model_labels]

    plot_families = []
    plot_data = []
    for family, fam_rows in family_results.items():
        vals = []
        for label, col, _ in model_columns:
            if col in fam_rows:
                v = fam_rows[col].replace("-", np.nan).astype(float).mean()
            else:
                v = np.nan
            vals.append(v)
        plot_families.append(family)
        plot_data.append(vals)
    if not plot_families:
        return
    plot_data_arr = np.array(plot_data)
    ind = np.arange(len(plot_families)) * 1.5
    bar_width = 0.18 if len(model_labels) > 3 else 0.25
    plt.figure(figsize=(10, max(6, len(plot_families) * 0.8)))
    plt.gca().set_prop_cycle(None)
    bar_handles = {}
    for i, label in enumerate(model_labels):
        bar = plt.barh(ind + i * bar_width, plot_data_arr[:, i], bar_width, label=label)
        bar_handles[label] = bar
    plt.yticks(ind + bar_width * (len(model_labels) - 1) / 2, plot_families)
    plt.xlabel("Accuracy (%)")
    plt.title(plot_title)
    handles = [bar_handles[label] for label in legend_order if label in bar_handles]
    labels = [label for label in legend_order if label in bar_handles]
    plt.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(str(plot_filename), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
