import pandas as pd
import pickle
from config import *


def get_shap_contributions(cph, row):
    coefficients = cph.params_
    contributions = {}

    for feature in row.index:
        contributions[feature] = row[feature] * coefficients[feature]

    return contributions


# ✅ Proper percentage normalization
def normalize_contributions(contributions):
    total = sum(abs(v) for v in contributions.values())

    if total == 0:
        return {k: 0 for k in contributions}

    return {k: (abs(v) / total) * 100 for k, v in contributions.items()}


def main():
    print("\n===== SHAP-LIKE EXPLANATION =====")

    df = pd.read_csv(PROCESSED_DATA_PATH + "/survival_data.csv")

    # Load trained Cox model
    with open(PROCESSED_DATA_PATH + "/cox_model.pkl", "rb") as f:
        cph = pickle.load(f)

    # -------------------------------
    # FEATURE ENGINEERING (same as training)
    # -------------------------------
    df["BVTV_change"] = df["C_BVTV_FU"] - df["C_BVTV_BL"]
    df["Pain_score"] = df["Facial_Pain_Intensity"]

    features = [
        "age", "gender", "Pain_score",
        "JawClickChewing", "JawGratingGrinding",
        "BVTV_change", "C_BVTV_BL"
    ]

    df = df[features + ["event", "time"]]

    print("Before handling missing:", df.shape)

    # -------------------------------
    # ✅ HANDLE MISSING VALUES (FIX)
    # -------------------------------
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    print("After handling missing:", df.shape)

    # -------------------------------
    # ENCODE CATEGORICAL
    # -------------------------------
    df["gender"] = df["gender"].astype("category").cat.codes

    # -------------------------------
    # PREPARE DATA
    # -------------------------------
    X = df.drop(columns=["event", "time"])

    if len(X) == 0:
        print("❌ No data available after preprocessing!")
        return

    # Take 3 sample patients
    sample = X.iloc[:3]

    # Load patient IDs
    df_full = pd.read_csv(PROCESSED_DATA_PATH + "/survival_data.csv")
    patient_ids = df_full["Patient"].iloc[:3]

    # -------------------------------
    # SHAP-LIKE EXPLANATION
    # -------------------------------
    for i in range(len(sample)):
        row = sample.iloc[i]

        print(f"\n===== Patient ID: {patient_ids.iloc[i]} =====")

        contributions = get_shap_contributions(cph, row)
        norm_contrib = normalize_contributions(contributions)

        sorted_contributions = sorted(
            norm_contrib.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("\nTop Risk Factors (Percentage Contribution):")
        for feat, val in sorted_contributions[:3]:
            direction = "⬆️ increases risk" if contributions[feat] > 0 else "⬇️ reduces risk"
            print(f"- {feat}: {val:.2f}% ({direction})")


if __name__ == "__main__":
    main()