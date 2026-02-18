import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from dotenv import load_dotenv


def get_paths() -> dict:
    data_dir = Path(os.getenv("DATA_FOLDER"))
    return {
        "DATA_DIR": data_dir,
        "PATH_RPPA": data_dir / "data_rppa_zscores.txt",
        "PATH_CASELIST_RPPA": data_dir / "case_lists/cases_rppa.txt",
        "PATH_MASTER": data_dir / "sample_master_table.csv",
    }


def load_master(path_master: Path) -> pd.DataFrame:
    master = pd.read_csv(path_master)
    print("Master table shape:", master.shape)
    return master


def parse_case_list_ids(path_case_list: Path) -> list[str]:
    rppa_ids: list[str] = []
    with open(path_case_list, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("case_list_ids:"):
                ids_str = line.split("case_list_ids:", 1)[1].strip()
                rppa_ids = ids_str.split()
                break

    print("Number of RPPA samples:", len(rppa_ids))
    print("First few RPPA IDs:", rppa_ids[:5])
    return rppa_ids


def load_rppa_matrix(path_rppa: Path) -> pd.DataFrame:
    rppa = pd.read_csv(path_rppa, sep="\t")
    print("RPPA matrix shape:", rppa.shape)
    return rppa


def build_rppa_merged(
    rppa: pd.DataFrame,
    rppa_ids: list[str],
    master: pd.DataFrame,
) -> pd.DataFrame:
    protein_col = rppa.columns[0]
    rppa_sample_cols = rppa.columns[1:]

    valid_samples = set(rppa_sample_cols) & set(rppa_ids) & set(master["SAMPLE_ID"])
    print("Usable RPPA samples:", len(valid_samples))

    rppa_filtered = rppa[[protein_col] + list(valid_samples)]
    rppa_t = rppa_filtered.set_index(protein_col).T
    rppa_t.index.name = "SAMPLE_ID"
    rppa_t.reset_index(inplace=True)

    print("Transposed RPPA shape:", rppa_t.shape)
    merged_rppa = rppa_t.merge(master, on="SAMPLE_ID")
    print("Merged RPPA + mutation shape:", merged_rppa.shape)
    return merged_rppa


def test_proteins(
    merged_rppa: pd.DataFrame,
    proteins: list[str],
    status_col: str,
) -> pd.DataFrame:
    results = []
    for protein in proteins:
        if protein not in merged_rppa.columns:
            print(f"{protein} not found in RPPA data")
            continue

        temp = merged_rppa[[protein, status_col]].dropna()
        mut_vals = temp[temp[status_col] == "Mut"][protein]
        wt_vals = temp[temp[status_col] == "WT"][protein]

        if len(mut_vals) > 5 and len(wt_vals) > 5:
            stat, p = mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")
            results.append((protein, len(mut_vals), len(wt_vals), p))

    return pd.DataFrame(
        results, columns=["Protein", "N_Mut", "N_WT", "p_value"]
    ).sort_values("p_value")


def plot_protein_boxplot(
    merged_rppa: pd.DataFrame,
    protein: str,
    status_col: str,
    gene: str,
) -> None:
    plot_df = merged_rppa[[protein, status_col]].dropna()
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=plot_df, x=status_col, y=protein, palette="Set2")
    plt.title(f"{protein}: {gene}-Mut vs WT")
    plt.ylabel("RPPA Z-score")
    plt.xlabel("")
    plt.show()


def main() -> None:
    load_dotenv()
    paths = get_paths()

    master = load_master(paths["PATH_MASTER"])
    rppa_ids = parse_case_list_ids(paths["PATH_CASELIST_RPPA"])
    rppa = load_rppa_matrix(paths["PATH_RPPA"])
    merged_rppa = build_rppa_merged(rppa, rppa_ids, master)

    gene = "ZFP36L2"
    status_col = "ZFP36L2_status"
    pi3k_proteins = rppa["Composite.Element.REF"].tolist()

    results = test_proteins(merged_rppa, pi3k_proteins, status_col)
    print(results)

    if pi3k_proteins:
        plot_protein_boxplot(merged_rppa, pi3k_proteins[0], status_col, gene)


if __name__ == "__main__":
    main()
