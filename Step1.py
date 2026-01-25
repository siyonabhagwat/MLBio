import pandas as pd
from pathlib import Path

import os
from .env import load_dotenv

# === File paths (EDIT THESE if needed) ===
DATA_DIR = Path(os.getenv("DATA_FOLDER"))
PATH_MUTSIG = DATA_DIR / "data_mutsig.txt"                 # gene-level MutSig summary
PATH_MUTATIONS = DATA_DIR / "data_mutations.txt"  # per-mutation file (MAF-like)
PATH_CASES_COMPLETE = DATA_DIR / "case_lists/cases_complete.txt"      # list of complete samples
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

# Define which Variant_Classification values count as non-silent
non_silent_classes = [
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins"
]

mut_nonsilent = mut_chosen[mut_chosen[VARCLASS_COL].isin(non_silent_classes)].copy()

print("Non-silent mutations for chosen genes:", len(mut_nonsilent))
print(mut_nonsilent[[SAMPLE_COL, GENE_COL, VARCLASS_COL]].head())

def shorten_tcga_barcode(barcode: str, length: int = 16) -> str:
    if isinstance(barcode, str):
        return barcode[:length]
    return barcode

mut_nonsilent["SAMPLE_ID"] = mut_nonsilent[SAMPLE_COL].apply(shorten_tcga_barcode)

# Also derive PATIENT_ID (first 12 chars: TCGA-XX-XXXX)
mut_nonsilent["PATIENT_ID"] = mut_nonsilent["SAMPLE_ID"].str[:12]

mut_nonsilent.head()


# Get list of all unique samples that have *any* mutation in chosen genes
all_mutated_samples = mut_nonsilent["SAMPLE_ID"].unique()
print("Total samples with at least one mutation in chosen genes:", len(all_mutated_samples))

# Start table with all unique SAMPLE_ID + PATIENT_ID from the non-silent set
sample_table = mut_nonsilent[["SAMPLE_ID", "PATIENT_ID"]].drop_duplicates().reset_index(drop=True)

# For each gene, mark Mut / WT
for gene in chosen_genes:
    # Samples with a non-silent mutation in this gene
    samples_with_gene = mut_nonsilent.loc[mut_nonsilent[GENE_COL] == gene, "SAMPLE_ID"].unique()
    mutant_set = set(samples_with_gene)
    
    # Default to WT
    col_name = f"{gene}_status"
    sample_table[col_name] = "WT"
    
    # Set to Mut where sample is in mutant_set
    sample_table.loc[sample_table["SAMPLE_ID"].isin(mutant_set), col_name] = "Mut"

sample_table.head()


# ==============

# === NEW Section 6: parse cBio-style case list file ===

complete_ids = []

with open(PATH_CASES_COMPLETE, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("case_list_ids:"):
            # Get everything after "case_list_ids:"
            ids_str = line.split("case_list_ids:", 1)[1].strip()
            # IDs are separated by tabs or spaces → split on any whitespace
            complete_ids = ids_str.split()
            break  # we found the line we need

print(f"Number of IDs in cases_complete: {len(complete_ids)}")
print("First 5 IDs:", complete_ids[:5])

# Turn into a DataFrame with column SAMPLE_ID
cases_complete_df = pd.DataFrame({"SAMPLE_ID": complete_ids})

# Filter sample_table to only those in complete_ids
sample_table_complete = sample_table[sample_table["SAMPLE_ID"].isin(complete_ids)].reset_index(drop=True)

print("Samples in mutation table:", len(sample_table))
print("Samples after restricting to complete cases:", len(sample_table_complete))
sample_table_complete.head()


sample_table_complete.to_csv(OUTPUT_MASTER_TABLE, index=False)
print(f"Saved master table to: {OUTPUT_MASTER_TABLE}")

# Quick summary: counts per gene
for gene in chosen_genes:
    col = f"{gene}_status"
    counts = sample_table_complete[col].value_counts()
    print(f"\n{gene} status counts:")
    print(counts)
