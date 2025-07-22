import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import List

EVAL_DATASETS = [
    'C-Eval',
    'C3',
    'CMMLU',
    'GaokaoBench',
    'GSM8K',
    'HellaSwag',
    'HumanEval',
    'MATH',
    'MBPP-CN',
    'MBPP',
    'MMLU',
    'Natural Questions',
    'RACE',
    'TheoremQA',
    'TriviaQA',
    'Winogrande'
]

def get_model_name(folder_name: str) -> str:
    """Map folder names to nicely formatted model names."""
    model_map = {
        'sweep_internvl_2_5_1b': 'InternVL 2.5 (1B)',
        'sweep_internvl_3_2b': 'InternVL 3 (2B)',
        'sweep_internvl_2_5_2b': 'InternVL 2.5 (2B)'
    }
    return model_map.get(folder_name, folder_name)

parser = argparse.ArgumentParser()
parser.add_argument('folders', nargs='+', help='Folder names in outputs/default (e.g., sweep_internvl_3_2b sweep_llama2_7b)')

def plot_single_folder(folder_path: str):
    """Plot accuracy vs tokens for a single folder."""
    folder = Path(folder_path) / 'summary'
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder}")
        return

    plt.figure(figsize=(10, 6))

    model_name = get_model_name(folder.parent.name)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        token_lengths = [8, 16, 32, 64, 128]
        columns = [col for col in df.columns if any(col.endswith(str(length)) for length in token_lengths)]

        if not columns:
            print(f"No columns found matching the expected format in {csv_file}")
            continue

        lengths = [int(col.split('-')[-1]) for col in columns]
        accuracies = df[columns].mean()

        plt.plot(lengths, accuracies.values, marker='o')

    plt.xlabel('Number of Output Tokens')
    plt.ylabel('Accuracy')
    plt.title(model_name)
    plt.grid(True)

    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)

    output_file = figures_dir / f"{model_name}.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

def plot_comparison(folders: List[str]):
    """Plot accuracy vs tokens for multiple folders."""
    plt.figure(figsize=(10, 6))

    for folder in folders:
        folder_path = Path(folder) / 'summary'
        csv_files = list(folder_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {folder_path}")
            continue

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Get columns that end with token lengths
            token_lengths = [8, 16, 32, 64, 128]
            columns = [col for col in df.columns if any(col.endswith(str(length)) for length in token_lengths)]

            if not columns:
                print(f"No columns found matching the expected format in {csv_file}")
                continue

            # Extract token lengths from column names
            lengths = [int(col.split('-')[-1]) for col in columns]
            accuracies = df[columns].mean()

            model_name = get_model_name(folder_path.parent.name)
            plt.plot(lengths, accuracies.values, marker='o', label=f"{model_name} - {csv_file.name}")

    plt.xlabel('Number of Output Tokens')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Output Tokens')
    plt.grid(True)
    plt.legend()

    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)

    output_file = figures_dir / "token_accuracy_comparison.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

def main():
    args = parser.parse_args()

    if len(args.folders) == 1:
        plot_single_folder(args.folders[0])
    else:
        plot_comparison(args.folders)

if __name__ == "__main__":
    main()