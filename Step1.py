import pandas as pd
from pathlib import Path

# Set data directory to user's actual location
DATA_DIR = Path("/Users/siyonabhagwat/Downloads/brca_tcga")

# === File paths (EDIT THESE if needed) ===
PATH_MUTSIG = DATA_DIR / "data_mutsig.txt"                 # gene-level MutSig summary
PATH_MUTATIONS = DATA_DIR / .txt"  # per-mutation file (MAF-like)
PATH_CASES_COMPLETE = DATA_DIR / "cases_complete.txt"      # list of complete samples
OUTPUT_MASTER_TABLE = DATA_DIR / "sample_master_table.csv"

# Load MutSig summary
mutsig = pd.read_csv(PATH_MUTSIG, sep="\t")

# Inspect columns (run once to see what's there)
print(mutsig.columns)

# Example: look at top 20 genes by significance
mutsig_sorted = mutsig.sort_values("q").head(20)
print(mutsig_sorted[["gene", "npat", "q"]])

# === Choose genes manually after inspecting ===
chosen_genes = ["PIK3CA", "TP53", "GATA3"]  # EDIT if you pick different ones
print("Chosen genes:", chosen_genes)
# Load per-mutation file (MAF-like)
mut = pd.read_csv(PATH_MUTATIONS, sep="\t")

# Peek at columns to adjust names if needed
print(mut.columns)

# Common column names in TCGA MAFs:
# 'Tumor_Sample_Barcode', 'Hugo_Symbol', 'Variant_Classification'
# If your file uses different names, EDIT them below.
SAMPLE_COL = "Tumor_Sample_Barcode"
GENE_COL = "Hugo_Symbol"
VARCLASS_COL = "Variant_Classification"

# Keep only mutations in chosen genes
mut_chosen = mut[mut[GENE_COL].isin(chosen_genes)].copy()

print("Total rows in mutation file:", len(mut))
print("Rows for chosen genes:", len(mut_chosen))
