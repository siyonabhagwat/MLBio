import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
# === File paths (EDIT if needed) ===
PATH_MASTER = "sample_master_table.csv"
PATH_RPPA = "data_rppa_zscores.txt"
PATH_CASELIST_RPPA = "cases_rppa.txt"
master = pd.read_csv(PATH_MASTER)
print("Master table shape:", master.shape)
master.head()
rppa_ids = []

with open(PATH_CASELIST_RPPA, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("case_list_ids:"):
            ids_str = line.split("case_list_ids:", 1)[1].strip()
            rppa_ids = ids_str.split()
            break

print("Number of RPPA samples:", len(rppa_ids))
print("First few RPPA IDs:", rppa_ids[:5])
rppa = pd.read_csv(PATH_RPPA, sep="\t")

print("RPPA matrix shape:", rppa.shape)
rppa.head()
PROTEIN_COL = rppa.columns[0]
rppa_sample_cols = rppa.columns[1:]

valid_samples = (
    set(rppa_sample_cols)
    & set(rppa_ids)
    & set(master["SAMPLE_ID"])
)

print("Usable RPPA samples:", len(valid_samples))

rppa_filtered = rppa[[PROTEIN_COL] + list(valid_samples)]
rppa_t = rppa_filtered.set_index(PROTEIN_COL).T
rppa_t.index.name = "SAMPLE_ID"
rppa_t.reset_index(inplace=True)

print("Transposed RPPA shape:", rppa_t.shape)
rppa_t.head()
merged_rppa = rppa_t.merge(master, on="SAMPLE_ID")

print("Merged RPPA + mutation shape:", merged_rppa.shape)
merged_rppa.head()
gene = "PIK3CA"
status_col = "PIK3CA_status"

pi3k_proteins = [
    "AKT",
    "AKT_pS473",
    "S6",
    "S6_pS235_S236",
    "4EBP1",
    "4EBP1_pT37_T46"
]
results = []

for protein in pi3k_proteins:
    if protein not in merged_rppa.columns:
        print(f"{protein} not found in RPPA data")
        continue

    temp = merged_rppa[[protein, status_col]].dropna()

    mut_vals = temp[temp[status_col] == "Mut"][protein]
    wt_vals  = temp[temp[status_col] == "WT"][protein]

    if len(mut_vals) > 5 and len(wt_vals) > 5:
        stat, p = mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")
        results.append((protein, len(mut_vals), len(wt_vals), p))

pd.DataFrame(
    results,
    columns=["Protein", "N_Mut", "N_WT", "p_value"]
).sort_values("p_value")
protein_to_plot = "AKT_pS473"

plot_df = merged_rppa[[protein_to_plot, status_col]].dropna()

plt.figure(figsize=(6,4))
sns.boxplot(
    data=plot_df,
    x=status_col,
    y=protein_to_plot,
    palette="Set2"
)
plt.title(f"{protein_to_plot}: {gene}-Mut vs WT")
plt.ylabel("RPPA Z-score")
plt.xlabel("")
plt.show()
gene = "TP53"
status_col = "TP53_status"

tp53_proteins = ["TP53", "BAX", "CDKN1A", "CyclinB1"]

tp53_results = []

for protein in tp53_proteins:
    if protein not in merged_rppa.columns:
        continue

    temp = merged_rppa[[protein, status_col]].dropna()
    mut_vals = temp[temp[status_col]=="Mut"][protein]
    wt_vals  = temp[temp[status_col]=="WT"][protein]

    if len(mut_vals) > 5 and len(wt_vals) > 5:
        stat, p = mannwhitneyu(mut_vals, wt_vals)
        tp53_results.append((protein, p))

pd.DataFrame(tp53_results, columns=["Protein", "p_value"])

