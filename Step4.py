import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from dotenv import load_dotenv


def get_paths() -> dict:
    data_dir = Path(os.getenv("DATA_FOLDER"))
    return {
        "DATA_DIR": data_dir,
        "PATH_MASTER": data_dir / "sample_master_table.csv",
        "PATH_CLINICAL": data_dir / "data_clinical_patient.txt",
    }


def load_master(path_master: Path) -> pd.DataFrame:
    master = pd.read_csv(path_master)
    print("Master table shape:", master.shape)
    return master


def load_clinical(path_clinical: Path) -> pd.DataFrame:
    clinical = pd.read_csv(path_clinical, sep="\t", comment="#")
    print("Clinical table shape:", clinical.shape)
    return clinical


def merge_clinical(master: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    merged_clinical = master.merge(clinical, on="PATIENT_ID", how="inner")
    print("Merged clinical shape:", merged_clinical.shape)
    return merged_clinical


def analyze_receptor_association(
    merged_clinical: pd.DataFrame,
    status_col: str,
    receptor_col: str,
    gene: str,
) -> float:
    ct = pd.crosstab(merged_clinical[status_col], merged_clinical[receptor_col])
    print(ct)
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"Chi-square p-value: {p}")

    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    ct_norm.plot(kind="bar", stacked=True)
    plt.title(f"ER status by {gene} mutation")
    plt.ylabel("Proportion")
    plt.xlabel("Mutation status")
    plt.legend(title="ER status")
    plt.savefig(f"pictures/{gene}_ER_status.png",pad_inches=0)
    plt.close()
    return p


def plot_stage_distribution(
    merged_clinical: pd.DataFrame,
    status_col: str,
    stage_col: str,
    gene: str,
) -> None:
    stage_df = merged_clinical[[status_col, stage_col]].dropna()
    print(pd.crosstab(stage_df[status_col], stage_df[stage_col]))

    plt.figure(figsize=(7, 4))
    sns.countplot(data=stage_df, x=stage_col, hue=status_col)
    plt.xticks(rotation=45)
    plt.title(f"Tumor stage by {gene} mutation status")
    #plt.show()
    plt.close()


def build_survival_df(merged_clinical: pd.DataFrame, status_col: str) -> pd.DataFrame:
    surv_df = merged_clinical[[status_col, "OS_MONTHS", "OS_STATUS"]].copy()
    surv_df["OS_MONTHS"] = pd.to_numeric(surv_df["OS_MONTHS"], errors="coerce")
    surv_df = surv_df.dropna(subset=["OS_MONTHS", "OS_STATUS"])

    surv_df["EVENT"] = surv_df["OS_STATUS"].map({
        "1:DECEASED": 1,
        "0:LIVING": 0,
    })
    return surv_df


def plot_km_curves(surv_df: pd.DataFrame, status_col: str, gene: str) -> None:
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6, 4))

    for label, grp in surv_df.groupby(status_col):
        kmf.fit(
            durations=grp["OS_MONTHS"],
            event_observed=grp["EVENT"],
            label=label,
        )
        kmf.plot_survival_function()

    plt.title(f"Overall survival by {gene} mutation status")
    plt.xlabel("Months")
    plt.ylabel("Survival probability")
    plt.savefig(f"pictures/{gene}_km_curve.png", pad_inches=0.)
    plt.close()
def run_logrank_test(surv_df: pd.DataFrame, status_col: str) -> float:
    mut = surv_df[surv_df[status_col] == "Mut"]
    wt = surv_df[surv_df[status_col] == "WT"]

    lr = logrank_test(
        mut["OS_MONTHS"],
        wt["OS_MONTHS"],
        event_observed_A=mut["EVENT"],
        event_observed_B=wt["EVENT"],
    )

    print("Log-rank p-value:", lr.p_value)
    return lr.p_value


def summarize_results(
    merged_clinical: pd.DataFrame,
    status_col: str,
    gene: str,
    er_p: float,
    os_p: float,
) -> dict:
    return {
        "Gene": gene,
        "N_mut": (merged_clinical[status_col] == "Mut").sum(),
        "N_WT": (merged_clinical[status_col] == "WT").sum(),
        "ER_assoc_p": er_p,
        "OS_logrank_p": os_p,
    }


def main() -> None:
    load_dotenv()
    paths = get_paths()

    master = load_master(paths["PATH_MASTER"])
    clinical = load_clinical(paths["PATH_CLINICAL"])
    merged_clinical = merge_clinical(master, clinical)

    gene = "ASXL2"
    status_col = "ASXL2_status"
    receptor_col = "ER_STATUS_BY_IHC"
    stage_col = "AJCC_PATHOLOGIC_TUMOR_STAGE"

    er_p = analyze_receptor_association(merged_clinical, status_col, receptor_col, gene)
    plot_stage_distribution(merged_clinical, status_col, stage_col, gene)

    surv_df = build_survival_df(merged_clinical, status_col)
    plot_km_curves(surv_df, status_col, gene)
    os_p = run_logrank_test(surv_df, status_col)

    summary = summarize_results(merged_clinical, status_col, gene, er_p, os_p)
    print(summary)


if __name__ == "__main__":
    main()

