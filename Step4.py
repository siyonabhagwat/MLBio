import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

# === File paths (EDIT THESE if needed) ===
DATA_DIR = Path(os.getenv("DATA_FOLDER"))
PATH_MASTER = DATA_DIR / "sample_master_table.csv"
PATH_CLINICAL = DATA_DIR / "data_clinical_patient.txt"
master = pd.read_csv(PATH_MASTER)

print("Master table shape:", master.shape)
master.head()
clinical = pd.read_csv(PATH_CLINICAL, sep="\t", comment="#")

print("Clinical table shape:", clinical.shape)
clinical.head()
merged_clinical = master.merge(
    clinical,
    on="PATIENT_ID",
    how="inner"
)

print("Merged clinical shape:", merged_clinical.shape)
merged_clinical.head()
gene = "TP53"
status_col = "TP53_status"
receptor_col = "ER_STATUS_BY_IHC"
# Build contingency table
ct = pd.crosstab(
    merged_clinical[status_col],
    merged_clinical[receptor_col]
)

print(ct)
chi2, p, dof, expected = chi2_contingency(ct)

print(f"Chi-square p-value: {p}")
ct_norm = ct.div(ct.sum(axis=1), axis=0)

ct_norm.plot(kind="bar", stacked=True)
plt.title(f"ER status by {gene} mutation")
plt.ylabel("Proportion")
plt.xlabel("Mutation status")
plt.legend(title="ER status")
plt.show()
stage_col = "AJCC_PATHOLOGIC_TUMOR_STAGE"

stage_df = merged_clinical[[status_col, stage_col]].dropna()

print(pd.crosstab(stage_df[status_col], stage_df[stage_col]))
plt.figure(figsize=(7,4))
sns.countplot(
    data=stage_df,
    x=stage_col,
    hue=status_col
)
plt.xticks(rotation=45)
plt.title(f"Tumor stage by {gene} mutation status")
plt.show()
# Clean survival columns
surv_df = merged_clinical[
    [status_col, "OS_MONTHS", "OS_STATUS"]
].copy()

# Coerce invalid OS_MONTHS values (e.g., "[Not Available]") to NaN
surv_df["OS_MONTHS"] = pd.to_numeric(surv_df["OS_MONTHS"], errors="coerce")

# Drop rows with missing survival data
surv_df = surv_df.dropna(subset=["OS_MONTHS", "OS_STATUS"])

# Convert OS_STATUS to binary
# Common format: "DECEASED" / "LIVING"
surv_df["EVENT"] = surv_df["OS_STATUS"].map({
    "1:DECEASED": 1,
    "0:LIVING": 0
})
kmf = KaplanMeierFitter()

plt.figure(figsize=(6,4))

for label, grp in surv_df.groupby(status_col):
    kmf.fit(
        durations=grp["OS_MONTHS"],
        event_observed=grp["EVENT"],
        label=label
    )
    kmf.plot_survival_function()

plt.title(f"Overall survival by {gene} mutation status")
plt.xlabel("Months")
plt.ylabel("Survival probability")
plt.show()
mut = surv_df[surv_df[status_col] == "Mut"]
wt  = surv_df[surv_df[status_col] == "WT"]

lr = logrank_test(
    mut["OS_MONTHS"],
    wt["OS_MONTHS"],
    event_observed_A=mut["EVENT"],
    event_observed_B=wt["EVENT"]
)

print("Log-rank p-value:", lr.p_value)
summary = {
    "Gene": gene,
    "N_mut": (merged_clinical[status_col] == "Mut").sum(),
    "N_WT": (merged_clinical[status_col] == "WT").sum(),
    "ER_assoc_p": p,
    "OS_logrank_p": lr.p_value
}

summary

