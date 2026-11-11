import pandas as pd


def load_excel(file_path):
    return pd.read_excel(file_path)


def standardize_column_names(df):
    """Clean column names"""
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def find_patient_column(df):
    """Find patient ID column automatically"""
    for col in df.columns:
        if "patient" in col.lower():
            return col
    raise ValueError("❌ No patient column found!")


def merge_data(clinical, fossa, condyle):
    """Robust merging"""

    # Detect patient columns
    c_col = find_patient_column(clinical)
    f_col = find_patient_column(fossa)
    d_col = find_patient_column(condyle)

    print(f"\nDetected keys:")
    print(f"Clinical → {c_col}")
    print(f"Fossa → {f_col}")
    print(f"Condyle → {d_col}")

    # Rename all to common name
    clinical = clinical.rename(columns={c_col: "Patient"})
    fossa = fossa.rename(columns={f_col: "Patient"})
    condyle = condyle.rename(columns={d_col: "Patient"})

    # 🔥 IMPORTANT: Ensure same datatype
    clinical["Patient"] = clinical["Patient"].astype(str).str.strip()
    fossa["Patient"] = fossa["Patient"].astype(str).str.strip()
    condyle["Patient"] = condyle["Patient"].astype(str).str.strip()

    # Merge
    merged = pd.merge(clinical, fossa, on="Patient", how="inner")
    merged = pd.merge(merged, condyle, on="Patient", how="inner")

    return merged


def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"\n✅ Saved to {path}")