import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def load_paths() -> dict:
    data_dir = Path(os.getenv("DATA_FOLDER"))
    return {
        "DATA_DIR": data_dir,
        "PATH_MUTSIG": data_dir / "data_mutsig.txt",
        "PATH_MUTATIONS": data_dir / "data_mutations.txt",
        "PATH_CASES_COMPLETE": data_dir / "case_lists/cases_complete.txt",
        "OUTPUT_MASTER_TABLE": data_dir / "sample_master_table.csv",
    }


def inspect_mutsig(path_mutsig: Path) -> pd.DataFrame:
    mutsig = pd.read_csv(path_mutsig, sep="\t")
    print(mutsig.columns)
    mutsig_sorted = mutsig.sort_values("q").head(20)
    print(mutsig_sorted[["gene", "npat", "q"]])
    return mutsig


def choose_genes() -> list[str]:
    return ["GATA3", "NCOR1", "GPS2", "TCP10"]


def load_mutations(path_mutations: Path) -> pd.DataFrame:
    mut = pd.read_csv(path_mutations, sep="\t")
    print(mut.columns)
    return mut


def shorten_tcga_barcode(barcode: str, length: int = 16) -> str:
    if isinstance(barcode, str):
        return barcode[:length]
    return barcode


def filter_non_silent_mutations(
    mut: pd.DataFrame,
    chosen_genes: list[str],
    sample_col: str,
    gene_col: str,
    varclass_col: str,
) -> pd.DataFrame:
    mut_chosen = mut[mut[gene_col].isin(chosen_genes)].copy()
    print("Total rows in mutation file:", len(mut))
    print("Rows for chosen genes:", len(mut_chosen))

    non_silent_classes = [
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "Splice_Site",
        "In_Frame_Del",
        "In_Frame_Ins",
    ]

    mut_nonsilent = mut_chosen[mut_chosen[varclass_col].isin(non_silent_classes)].copy()
    print("Non-silent mutations for chosen genes:", len(mut_nonsilent))
    print(mut_nonsilent[[sample_col, gene_col, varclass_col]].head())

    mut_nonsilent["SAMPLE_ID"] = mut_nonsilent[sample_col].apply(shorten_tcga_barcode)
    mut_nonsilent["PATIENT_ID"] = mut_nonsilent["SAMPLE_ID"].str[:12]
    return mut_nonsilent


def build_sample_table(
    mut_nonsilent: pd.DataFrame,
    chosen_genes: list[str],
    gene_col: str,
) -> pd.DataFrame:
    all_mutated_samples = mut_nonsilent["SAMPLE_ID"].unique()
    print("Total samples with at least one mutation in chosen genes:", len(all_mutated_samples))

    sample_table = (
        mut_nonsilent[["SAMPLE_ID", "PATIENT_ID"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    for gene in chosen_genes:
        samples_with_gene = mut_nonsilent.loc[
            mut_nonsilent[gene_col] == gene, "SAMPLE_ID"
        ].unique()
        mutant_set = set(samples_with_gene)

        col_name = f"{gene}_status"
        sample_table[col_name] = "WT"
        sample_table.loc[sample_table["SAMPLE_ID"].isin(mutant_set), col_name] = "Mut"

    return sample_table


def parse_case_list_ids(path_case_list: Path) -> list[str]:
    complete_ids: list[str] = []
    with open(path_case_list, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("case_list_ids:"):
                ids_str = line.split("case_list_ids:", 1)[1].strip()
                complete_ids = ids_str.split()
                break

    print(f"Number of IDs in cases_complete: {len(complete_ids)}")
    print("First 5 IDs:", complete_ids[:5])
    return complete_ids


def restrict_to_complete_cases(
    sample_table: pd.DataFrame,
    complete_ids: list[str],
) -> pd.DataFrame:
    sample_table_complete = sample_table[
        sample_table["SAMPLE_ID"].isin(complete_ids)
    ].reset_index(drop=True)

    print("Samples in mutation table:", len(sample_table))
    print("Samples after restricting to complete cases:", len(sample_table_complete))
    return sample_table_complete


def save_master_table(sample_table_complete: pd.DataFrame, output_path: Path) -> None:
    sample_table_complete.to_csv(output_path, index=False)
    print(f"Saved master table to: {output_path}")


def print_gene_status_counts(sample_table_complete: pd.DataFrame, chosen_genes: list[str]) -> None:
    for gene in chosen_genes:
        col = f"{gene}_status"
        counts = sample_table_complete[col].value_counts()
        print(f"\n{gene} status counts:")
        print(counts)


def main() -> None:
    load_dotenv()
    paths = load_paths()

    inspect_mutsig(paths["PATH_MUTSIG"])
    chosen_genes = choose_genes()
    print("Chosen genes:", chosen_genes)

    mut = load_mutations(paths["PATH_MUTATIONS"])

    sample_col = "Tumor_Sample_Barcode"
    gene_col = "Hugo_Symbol"
    varclass_col = "Variant_Classification"

    mut_nonsilent = filter_non_silent_mutations(
        mut, chosen_genes, sample_col, gene_col, varclass_col
    )
    sample_table = build_sample_table(mut_nonsilent, chosen_genes, gene_col)

    complete_ids = parse_case_list_ids(paths["PATH_CASES_COMPLETE"])
    _ = pd.DataFrame({"SAMPLE_ID": complete_ids})
    sample_table_complete = restrict_to_complete_cases(sample_table, complete_ids)

    save_master_table(sample_table_complete, paths["OUTPUT_MASTER_TABLE"])
    print_gene_status_counts(sample_table_complete, chosen_genes)


if __name__ == "__main__":
    main()
