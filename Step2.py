import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()
# === File paths (EDIT THESE if needed) ===
DATA_DIR = Path(os.getenv("DATA_FOLDER"))

# === FILE PATHS (EDIT THESE IF NEEDED) ===
PATH_MASTER = DATA_DIR /"sample_master_table.csv"          # from Step 1
PATH_EXPR   = DATA_DIR /"case_lists/cases_RNA_Seq_v2_mRNA.txt"        # or "cases_mRNA.txt"
OUTPUT_DIR  = DATA_DIR /"results_step2_expression"         # folder for tables/plots

os.makedirs(OUTPUT_DIR, exist_ok=True)

# If you used different chosen_genes in Step 1, list them here (for reference)
chosen_genes = ["PIK3CA", "TP53", "GATA3"]   # EDIT if different
# Load the master table from Step 1
master = pd.read_csv(PATH_MASTER)

print("Master table columns:", master.columns.tolist())
print("Number of samples in master table:", len(master))
print(master.head())
# Load expression matrix
# Assumes tab-separated with a 'Hugo_Symbol' (or similar) column for gene names
expr = pd.read_csv(PATH_EXPR, sep="\t")

print("Expression columns (first 10):", expr.columns[:10])

# Guess gene-name column; EDIT if yours is different
if "Hugo_Symbol" in expr.columns:
    GENE_COL = "Hugo_Symbol"
elif "Gene" in expr.columns:
    GENE_COL = "Gene"
else:
    # Fallback: assume first column is gene name
    GENE_COL = expr.columns[0]

print("Using gene name column:", GENE_COL)

# Set gene as index for easier subsetting
expr = expr.set_index(GENE_COL)

# Expression sample IDs = all columns except the gene index
expr_sample_ids = expr.columns.astype(str)
print("Total expression samples:", len(expr_sample_ids))

# Find overlap between expression samples and master table SAMPLE_ID
master_sample_ids = master["SAMPLE_ID"].astype(str)
common_ids = sorted(set(expr_sample_ids).intersection(set(master_sample_ids)))

print("Samples in BOTH expression and master:", len(common_ids))
print("First 10 common IDs:", common_ids[:10])

# Restrict expression matrix to common IDs only
expr_common = expr[common_ids].copy()

# Restrict master table to common IDs only
master_common = master[master["SAMPLE_ID"].isin(common_ids)].copy().reset_index(drop=True)

print("Shape of expr_common:", expr_common.shape)
print("Rows (genes), columns (samples).")
print("Number of samples in master_common:", len(master_common))
PATH_EXPR_CASELIST = "cases_RNA_Seq_v2_mRNA.txt"
PATH_EXPR_MATRIX   = "case_lists/data_RNA_Seq_v2_mRNA.txt"   # <-- YOU MUST UPDATE THIS TO THE REAL MATRIX
# === SECTION 3 (REWRITTEN FOR SAFETY) ===
# Build a table that contains mutation status + expression for each sample.

# First: check if expr_common exists (meaning expression matrix loaded successfully)
if 'expr_common' not in globals():
    raise ValueError(
        "Expression matrix not loaded. "
        "Make sure you defined PATH_EXPR_MATRIX to point to the REAL expression data file."
    )

# Restrict expression matrix to genes_found only
expr_sub = expr_common.loc[genes_found]

# Convert expression matrix so that samples are rows
expr_samples_as_rows = expr_sub.T.copy()
expr_samples_as_rows.index.name = "SAMPLE_ID"
expr_samples_as_rows.reset_index(inplace=True)

print("Expression table with samples as rows:")
print(expr_samples_as_rows.head())

# Merge expression + mutation status
merged = pd.merge(master_common, expr_samples_as_rows, on="SAMPLE_ID", how="inner")

print("\nMerged mutation + expression table:")
print(merged.head())

# Check expression columns exist
expression_cols = [g for g in genes_found if g in merged.columns]

if len(expression_cols) == 0:
    print("\n❗ WARNING: No expression genes were found in the merged table.")
    print("    → This means your expression matrix was NOT successfully merged.")
    print("    → You must correctly load the RNA expression matrix file.")
else:
    print(f"\nExpression genes successfully merged: {expression_cols}")

# Save intermediate file for inspection
merged.to_csv("merged_mutation_expression_table.csv", index=False)
print("\nSaved merged table → merged_mutation_expression_table.csv")
