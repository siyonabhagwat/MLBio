from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from dotenv import load_dotenv


def get_config() -> dict:
    data_dir = Path(os.getenv("DATA_FOLDER"))
    outdir = data_dir / "step2_rna_out"
    outdir.mkdir(parents=True, exist_ok=True)
    gene = "ZFP36L2"
    return {
        "DATA_DIR": data_dir,
        "PATH_MASTER": data_dir / "sample_master_table.csv",
        "PATH_CASELIST_RNASEQ": data_dir / "case_lists/cases_RNA_Seq_v2_mRNA.txt",
        "PATH_RNASEQ_MATRIX": data_dir / "data_mrna_seq_v2_rsem.txt",
        "OUTDIR": outdir,
        "GENE": gene,
        "STATUS_COL": f"{gene}_status",
        "PATHWAY_GENES": [
            "PIK3CA",
            "AKT1",
            "PDPK1",
            "MTOR",
            "RPS6KB1",
            "EIF4EBP1",
            "NCOR1",
        ],
        "USE_TCGA_15": False,
    }


def norm_id(s: str, use_tcga_15: bool) -> str:
    s = str(s).strip().replace(".", "-")
    s = re.sub(r"\s+", "", s)
    return s[:15] if use_tcga_15 else s


def parse_case_list(path_case_list: Path, use_tcga_15: bool) -> list[str]:
    rnaseq_ids: list[str] = []
    with open(path_case_list, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("case_list_ids:"):
                ids_str = line.split(":", 1)[1].strip()
                rnaseq_ids = re.split(r"\s+", ids_str) if ids_str else []
                break

    rnaseq_ids = [norm_id(x, use_tcga_15) for x in rnaseq_ids]
    print("2A) RNA-seq case-list IDs:", len(rnaseq_ids))
    print("    first 5:", rnaseq_ids[:5])
    return rnaseq_ids


def load_master(path_master: Path, status_col: str, use_tcga_15: bool) -> pd.DataFrame:
    master = pd.read_csv(path_master)
    if "SAMPLE_ID" not in master.columns:
        raise ValueError("Master table must have a 'SAMPLE_ID' column.")

    master["SAMPLE_ID"] = master["SAMPLE_ID"].map(lambda x: norm_id(x, use_tcga_15))

    if status_col not in master.columns:
        raise ValueError(
            f"Master table is missing '{status_col}'. Available *_status columns: "
            f"{[c for c in master.columns if c.endswith('_status')][:10]} ..."
        )

    print("2B) Master rows:", len(master))
    print("    Mut/WT counts:", master[status_col].value_counts(dropna=False).to_dict())
    return master


def load_rnaseq_matrix(path_rnaseq: Path, use_tcga_15: bool) -> tuple[pd.DataFrame, str]:
    rnaseq = pd.read_csv(path_rnaseq, sep="\t", low_memory=False)
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in rnaseq.columns else rnaseq.columns[0]

    sample_cols_raw = [c for c in rnaseq.columns if c != gene_col]
    rename_map = {c: norm_id(c, use_tcga_15) for c in sample_cols_raw}
    rnaseq = rnaseq.rename(columns=rename_map)

    print("2C) RNA matrix shape:", rnaseq.shape)
    print("    gene col:", gene_col)
    print(
        "    sample columns (first 5):",
        [c for c in rnaseq.columns if c != gene_col][:5],
    )
    return rnaseq, gene_col


def filter_valid_samples(
    rnaseq: pd.DataFrame,
    gene_col: str,
    rnaseq_ids: list[str],
    master: pd.DataFrame,
) -> list[str]:
    expr_sample_cols = [c for c in rnaseq.columns if c != gene_col]
    valid_samples = sorted(
        set(expr_sample_cols) & set(rnaseq_ids) & set(master["SAMPLE_ID"])
    )

    print("2D) Usable RNA samples (intersection):", len(valid_samples))
    if len(valid_samples) == 0:
        print("    DEBUG TIP: set USE_TCGA_15=True, or print some IDs to compare:")
        print("    master IDs first 5:", master["SAMPLE_ID"].head().tolist())
        print("    case list IDs first 5:", rnaseq_ids[:5])
        print("    expr cols first 5:", expr_sample_cols[:5])
        raise ValueError("No overlapping sample IDs found.")

    return valid_samples


def transpose_expression(
    rnaseq: pd.DataFrame,
    gene_col: str,
    valid_samples: list[str],
) -> pd.DataFrame:
    expr = rnaseq[[gene_col] + valid_samples].copy()
    expr[valid_samples] = expr[valid_samples].apply(pd.to_numeric, errors="coerce")

    expr_t = expr.set_index(gene_col).T
    expr_t.index.name = "SAMPLE_ID"
    expr_t.reset_index(inplace=True)

    print("2E) Transposed expr shape (samples x genes):", expr_t.shape)
    print("    columns (first 5):", expr_t.columns[:5].tolist())
    return expr_t


def merge_expression_with_status(
    expr_t: pd.DataFrame,
    master: pd.DataFrame,
    status_col: str,
) -> pd.DataFrame:
    merged = expr_t.merge(master[["SAMPLE_ID", status_col]], on="SAMPLE_ID", how="inner")
    print("2F) Merged shape:", merged.shape)
    print("    Mut/WT in merged:", merged[status_col].value_counts(dropna=False).to_dict())
    return merged


def plot_gene_expression(
    merged: pd.DataFrame,
    gene: str,
    status_col: str,
    outdir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if gene not in merged.columns:
        example_genes = [c for c in merged.columns if c not in ("SAMPLE_ID", status_col)]
        print("2G) Gene not found in expression columns:", gene)
        print("    Example gene columns (first 20):", example_genes[:20])
        raise ValueError(f"Expression matrix does not contain '{gene}' column.")

    sub = merged[[gene, status_col]].dropna()
    mut_vals = sub.loc[sub[status_col] == "Mut", gene].values
    wt_vals = sub.loc[sub[status_col] == "WT", gene].values

    print("2G) Gene:", gene)
    print("    n(Mut) =", len(mut_vals), "n(WT) =", len(wt_vals))
    print("    mean(Mut) =", float(np.mean(mut_vals)), "mean(WT) =", float(np.mean(wt_vals)))

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=sub, x=status_col, y=gene)
    sns.stripplot(data=sub, x=status_col, y=gene, size=3, alpha=0.5)
    plt.title(f"{gene} expression by {status_col}")
    plt.tight_layout()
    plt.savefig(outdir / f"2G_boxplot_{gene}_{status_col}.png", dpi=200)
    plt.show()

    return mut_vals, wt_vals


def run_mann_whitney(mut_vals: np.ndarray, wt_vals: np.ndarray) -> None:
    if len(mut_vals) >= 3 and len(wt_vals) >= 3:
        stat, p = mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")
        print(f"2H) Mann-Whitney U: U={stat:.3g}, p={p:.3e}")
    else:
        print("2H) Not enough samples for test (need >=3 per group).")


def test_pathway_genes(
    merged: pd.DataFrame,
    pathway_genes: list[str],
    status_col: str,
    outdir: Path,
    gene: str,
) -> pd.DataFrame:
    rows = []
    for g in pathway_genes:
        if g not in merged.columns:
            print(f"2I) Skipping {g} (not in expression matrix)")
            continue
        temp = merged[[g, status_col]].dropna()
        mut = temp.loc[temp[status_col] == "Mut", g].values
        wt = temp.loc[temp[status_col] == "WT", g].values
        if len(mut) < 3 or len(wt) < 3:
            continue
        stat, p = mannwhitneyu(mut, wt, alternative="two-sided")
        rows.append(
            {
                "gene": g,
                "n_mut": len(mut),
                "n_wt": len(wt),
                "p_mwu": p,
                "mean_mut": float(np.mean(mut)),
                "mean_wt": float(np.mean(wt)),
            }
        )

    results = pd.DataFrame(rows).sort_values("p_mwu", ascending=True)
    print("2I) Pathway gene results:")
    print(results)

    results.to_csv(
        outdir / f"2I_pathway_tests_{gene}_{status_col}.tsv",
        sep="\t",
        index=False,
    )
    return results


def main() -> None:
    load_dotenv()
    config = get_config()

    rnaseq_ids = parse_case_list(config["PATH_CASELIST_RNASEQ"], config["USE_TCGA_15"])
    master = load_master(
        config["PATH_MASTER"], config["STATUS_COL"], config["USE_TCGA_15"]
    )
    rnaseq, gene_col = load_rnaseq_matrix(
        config["PATH_RNASEQ_MATRIX"], config["USE_TCGA_15"]
    )

    valid_samples = filter_valid_samples(rnaseq, gene_col, rnaseq_ids, master)
    expr_t = transpose_expression(rnaseq, gene_col, valid_samples)
    merged = merge_expression_with_status(expr_t, master, config["STATUS_COL"])

    mut_vals, wt_vals = plot_gene_expression(
        merged, config["GENE"], config["STATUS_COL"], config["OUTDIR"]
    )
    run_mann_whitney(mut_vals, wt_vals)
    test_pathway_genes(
        merged,
        config["PATHWAY_GENES"],
        config["STATUS_COL"],
        config["OUTDIR"],
        config["GENE"],
    )


if __name__ == "__main__":
    main()
