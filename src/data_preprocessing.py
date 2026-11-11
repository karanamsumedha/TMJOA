import os
from config import *
from utils import *


def main():
    print("\n===== LOADING DATA =====")

    clinical = load_excel(CLINICAL_FILE)
    fossa = load_excel(FOSSA_FILE)
    condyle = load_excel(CONDYLE_FILE)

    print("Clinical shape:", clinical.shape)
    print("Fossa shape:", fossa.shape)
    print("Condyle shape:", condyle.shape)
    print("\nCLINICAL COLUMNS:", clinical.columns.tolist())
    print("\nFOSSA COLUMNS:", fossa.columns.tolist())
    print("\nCONDYLE COLUMNS:", condyle.columns.tolist())
    # -------------------------------
    # CLEAN COLUMN NAMES
    # -------------------------------
    clinical = standardize_column_names(clinical)
    fossa = standardize_column_names(fossa)
    condyle = standardize_column_names(condyle)

    # -------------------------------
    # MERGE DATA
    # -------------------------------
    print("\n===== MERGING DATA =====")

    merged = merge_data(clinical, fossa, condyle)

    print("Merged shape:", merged.shape)

    # -------------------------------
    # REMOVE DUPLICATE COLUMNS
    # -------------------------------
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # -------------------------------
    # SAVE OUTPUT
    # -------------------------------
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    save_csv(merged, MERGED_FILE)

    # -------------------------------
    # PRINT SUMMARY
    # -------------------------------
    print("\n===== FINAL COLUMNS =====\n")
    for i, col in enumerate(merged.columns):
        print(f"{i}: {col}")

    print("\nTotal columns:", len(merged.columns))


if __name__ == "__main__":
    main()