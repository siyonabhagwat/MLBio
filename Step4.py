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

