from zipfile import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
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
clinical = pd.read_csv(PATH_CLINICAL, sep="\t")

print("Clinical table shape:", clinical.shape)
clinical.head()
merged_clinical = master.merge(
    clinical,
    on="PATIENT_ID",
    how="inner"
)

print("Merged clinical shape:", merged_clinical.shape)
merged_clinical.head()
gene = "PIK3CA"
status_col = "PIK3CA_status"
receptor_col = "ER_STATUS"
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
].dropna()

# Convert OS_STATUS to binary
# Common format: "DECEASED" / "LIVING"
surv_df["EVENT"] = surv_df["OS_STATUS"].map({
    "DECEASED": 1,
    "LIVING": 0
})

