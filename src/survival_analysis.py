import pandas as pd
import pickle
import matplotlib.pyplot as plt
from config import *


def main():
    print("\n===== LOADING DATA & MODEL =====")

    df = pd.read_csv(PROCESSED_DATA_PATH + "/survival_data.csv")

    # Load trained model
    with open(PROCESSED_DATA_PATH + "/cox_model.pkl", "rb") as f:
        cph = pickle.load(f)

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    df["BVTV_change"] = df["C_BVTV_FU"] - df["C_BVTV_BL"]
    df["Pain_score"] = df["Facial_Pain_Intensity"]

    features = [
        "age",
        "gender",
        "Pain_score",
        "JawClickChewing",
        "JawGratingGrinding",
        "BVTV_change",
        "C_BVTV_BL"
    ]

    # -------------------------------
    # SELECT DATA (NO DROPNA ❌)
    # -------------------------------
    df = df[features + ["event", "time", "Patient"]]

    # -------------------------------
    # HANDLE MISSING VALUES ✅
    # -------------------------------
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # -------------------------------
    # ENCODE GENDER
    # -------------------------------
    df["gender"] = df["gender"].astype("category").cat.codes

    # -------------------------------
    # RESET INDEX (IMPORTANT)
    # -------------------------------
    df = df.reset_index(drop=True)

    print("Final data shape:", df.shape)

    # -------------------------------
    # SELECT SAMPLE PATIENTS
    # -------------------------------
    print("\nSelecting sample patients...")

    X = df.drop(columns=["event", "time", "Patient"])

    sample_patients = X.iloc[:3]
    sample_ids = df["Patient"].iloc[:3]  # ✅ aligned properly

    # -------------------------------
    # SURVIVAL FUNCTION
    # -------------------------------
    print("\nGenerating survival curves...")

    survival_functions = cph.predict_survival_function(sample_patients)

    # -------------------------------
    # PLOT
    # -------------------------------
    plt.figure()

    for i, col in enumerate(survival_functions.columns):
        plt.plot(
            survival_functions.index,
            survival_functions[col],
            label=f"Patient {sample_ids.iloc[i]}"
        )

    plt.title("Survival Curves (TMJ Progression Risk)")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()

    plt.savefig(PROCESSED_DATA_PATH + "/survival_curves.png")
    plt.show()

    print("\n✅ Survival curves saved!")

    # -------------------------------
    # RISK SCORES
    # -------------------------------
    print("\n===== RISK SCORES =====")

    risk_scores = cph.predict_partial_hazard(sample_patients)

    for i, score in enumerate(risk_scores):
        print(f"\n--- Patient ID: {sample_ids.iloc[i]} ---")
        print(f"Risk Score: {score:.4f}")


if __name__ == "__main__":
    main()