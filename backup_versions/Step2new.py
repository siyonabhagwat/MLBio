# STEP 2 — RNA-level functional impact (debuggable, split into 2A/2B/2C...)
# Keep it simple: pandas/numpy/scipy/matplotlib/seaborn

from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from dotenv import load_dotenv
load_dotenv()

# ======================
# CONFIG (edit if needed)
# ======================
DATA_DIR = Path(os.getenv("DATA_FOLDER"))  # you set this in your .env
PATH_MASTER = DATA_DIR / "sample_master_table.csv"         # from Step 1
PATH_CASELIST_RNASEQ = DATA_DIR / "case_lists/cases_RNA_Seq_v2_mRNA.txt"
PATH_RNASEQ_MATRIX = DATA_DIR / "data_mrna_seq_v2_rsem.txt" # numeric matrix
OUTDIR = DATA_DIR / "step2_rna_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

GENE = "PIK3CA"                 # debug one gene first
STATUS_COL = f"{GENE}_status"   # expects "PIK3CA_status" in master table

# Optional: genes to check beyond the gene itself
PATHWAY_GENES = ["PIK3CA", "AKT1", "PDPK1", "MTOR", "RPS6KB1", "EIF4EBP1"]

# If your IDs are TCGA-like and you see mismatches, set to True (truncates to 15 chars)
USE_TCGA_15 = False


# -----------------------
# Helper: normalize IDs
# -----------------------
def norm_id(s: str) -> str:
    s = str(s).strip().replace(".", "-")
    s = re.sub(r"\s+", "", s)
    return s[:15] if USE_TCGA_15 else s


# ==========================================
# 2A — Parse RNA-seq case list to sample IDs
# ==========================================
rnaseq_ids = []
with open(PATH_CASELIST_RNASEQ, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.lower().startswith("case_list_ids:"):
            ids_str = line.split(":", 1)[1].strip()
            rnaseq_ids = re.split(r"\s+", ids_str) if ids_str else []
            break

rnaseq_ids = [norm_id(x) for x in rnaseq_ids]
print("2A) RNA-seq case-list IDs:", len(rnaseq_ids))
print("    first 5:", rnaseq_ids[:5])


# ==========================================
# 2B — Load Step 1 master table (mutation WT/Mut)
# ==========================================
master = pd.read_csv(PATH_MASTER)

if "SAMPLE_ID" not in master.columns:
    raise ValueError("Master table must have a 'SAMPLE_ID' column.")

master["SAMPLE_ID"] = master["SAMPLE_ID"].map(norm_id) 

if STATUS_COL not in master.columns:
    raise ValueError(f"Master table is missing '{STATUS_COL}'. Available *_status columns: "
                     f"{[c for c in master.columns if c.endswith('_status')][:10]} ...")

print("2B) Master rows:", len(master))
print("    Mut/WT counts:", master[STATUS_COL].value_counts(dropna=False).to_dict())


# ==========================================
# 2C — Load RNA-seq expression matrix (genes x samples)
# ==========================================
rnaseq = pd.read_csv(PATH_RNASEQ_MATRIX, sep="\t", low_memory=False)

# Common cBio format has Hugo_Symbol as a gene column
gene_col = "Hugo_Symbol" if "Hugo_Symbol" in rnaseq.columns else rnaseq.columns[0]

# Normalize sample columns
sample_cols_raw = [c for c in rnaseq.columns if c != gene_col]
rename_map = {c: norm_id(c) for c in sample_cols_raw}
rnaseq = rnaseq.rename(columns=rename_map)

print("2C) RNA matrix shape:", rnaseq.shape)
print("    gene col:", gene_col)
print("    sample columns (first 5):", [c for c in rnaseq.columns if c != gene_col][:5])


# ==========================================
# 2D — Keep only samples present in: master ∩ rnaseq_ids ∩ expression columns
# ==========================================
expr_sample_cols = [c for c in rnaseq.columns if c != gene_col]
valid_samples = sorted(set(expr_sample_cols) & set(rnaseq_ids) & set(master["SAMPLE_ID"]))

print("2D) Usable RNA samples (intersection):", len(valid_samples))
if len(valid_samples) == 0:
    print("    DEBUG TIP: set USE_TCGA_15=True, or print some IDs to compare:")
    print("    master IDs first 5:", master["SAMPLE_ID"].head().tolist())
    print("    case list IDs first 5:", rnaseq_ids[:5])
    print("    expr cols first 5:", expr_sample_cols[:5])
    raise ValueError("No overlapping sample IDs found.")

expr = rnaseq[[gene_col] + valid_samples].copy()


# ==========================================
# 2E — Transpose to samples x genes
# ==========================================
# Make numeric (coerce non-numeric to NaN)
expr[valid_samples] = expr[valid_samples].apply(pd.to_numeric, errors="coerce")

expr_t = expr.set_index(gene_col).T
expr_t.index.name = "SAMPLE_ID"
expr_t.reset_index(inplace=True)

print("2E) Transposed expr shape (samples x genes):", expr_t.shape)
print("    columns (first 5):", expr_t.columns[:5].tolist())


# ==========================================
# 2F — Merge expression with mutation status
# ==========================================
merged = expr_t.merge(master[["SAMPLE_ID", STATUS_COL]], on="SAMPLE_ID", how="inner")
print("2F) Merged shape:", merged.shape)
print("    Mut/WT in merged:", merged[STATUS_COL].value_counts(dropna=False).to_dict())


# ==========================================
# 2G — Compare expression of GENE itself (Mut vs WT)
# ==========================================
if GENE not in merged.columns:
    # sometimes expression matrix uses different symbol naming; debug here
    example_genes = [c for c in merged.columns if c not in ("SAMPLE_ID", STATUS_COL)]
    print("2G) Gene not found in expression columns:", GENE)
    print("    Example gene columns (first 20):", example_genes[:20])
    raise ValueError(f"Expression matrix does not contain '{GENE}' column.")

sub = merged[[GENE, STATUS_COL]].dropna()
mut_vals = sub.loc[sub[STATUS_COL] == "Mut", GENE].values
wt_vals = sub.loc[sub[STATUS_COL] == "WT", GENE].values

print("2G) Gene:", GENE)
print("    n(Mut) =", len(mut_vals), "n(WT) =", len(wt_vals))
print("    mean(Mut) =", float(np.mean(mut_vals)), "mean(WT) =", float(np.mean(wt_vals)))

# Plot
plt.figure(figsize=(6, 4))
sns.boxplot(data=sub, x=STATUS_COL, y=GENE)
sns.stripplot(data=sub, x=STATUS_COL, y=GENE, size=3, alpha=0.5)
plt.title(f"{GENE} expression by {STATUS_COL}")
plt.tight_layout()
plt.savefig(OUTDIR / f"2G_boxplot_{GENE}_{STATUS_COL}.png", dpi=200)
plt.show()


# ==========================================
# 2H — Statistical test (Mann–Whitney U)
# ==========================================
if len(mut_vals) >= 3 and len(wt_vals) >= 3:
    stat, p = mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")
    print(f"2H) Mann–Whitney U: U={stat:.3g}, p={p:.3e}")
else:
    print("2H) Not enough samples for test (need >=3 per group).")


# ==========================================
# 2I — Test multiple downstream pathway genes
# ==========================================
rows = []
for g in PATHWAY_GENES:
    if g not in merged.columns:
        print(f"2I) Skipping {g} (not in expression matrix)")
        continue
    temp = merged[[g, STATUS_COL]].dropna()
    mut = temp.loc[temp[STATUS_COL] == "Mut", g].values
    wt = temp.loc[temp[STATUS_COL] == "WT", g].values
    if len(mut) < 3 or len(wt) < 3:
        continue
    stat, p = mannwhitneyu(mut, wt, alternative="two-sided")
    rows.append({"gene": g, "n_mut": len(mut), "n_wt": len(wt), "p_mwu": p,
                 "mean_mut": float(np.mean(mut)), "mean_wt": float(np.mean(wt))})

results = pd.DataFrame(rows).sort_values("p_mwu", ascending=True)
print("2I) Pathway gene results:")
print(results)

results.to_csv(OUTDIR / f"2I_pathway_tests_{GENE}_{STATUS_COL}.tsv", sep="\t", index=False)
