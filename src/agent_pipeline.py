import pandas as pd
import pickle
import os
import warnings

# -------------------------------
# 🔕 SUPPRESS WARNINGS
# -------------------------------
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import *
from genai_explainer import generate_explanation, get_contributions
from rag_module import load_documents, create_vector_db, retrieve_context


# -------------------------------
# 🧠 RAG INIT
# -------------------------------
folder_path = os.path.join("data", "knowledge")

texts = load_documents(folder_path)
db = create_vector_db(texts)


# -------------------------------
# ✅ FIXED RAG (FEATURE FILTERING)
# -------------------------------
def get_rag_context(feature):
    query_map = {
        "BVTV_change": "trabecular bone volume TMJ osteoarthritis",
        "C_BVTV_BL": "baseline bone structure TMJ",
        "Pain_score": "facial pain TMJ disorder",
        "JawClickChewing": "jaw clicking TMJ symptoms",
        "JawGratingGrinding": "jaw grinding TMJ disorder",
        "age": "age risk TMJ osteoarthritis"
    }

    query = query_map.get(feature, "TMJ osteoarthritis")
    results = retrieve_context(db, query)

    keyword = feature.split("_")[0].lower()
    filtered = [r for r in results if keyword in r.lower()]

    return filtered[:2] if filtered else results[:2]


# -------------------------------
# 🧩 FEATURE ALIGNMENT
# -------------------------------
def align_features_with_model(patient_row, cph):
    model_features = list(cph.params_.index)

    aligned = {}
    for feat in model_features:
        aligned[feat] = patient_row.get(feat, 0)

    return pd.Series(aligned)


# -------------------------------
# 🤖 AGENT PIPELINE
# -------------------------------
def run_agent(patient_row, cph):
    try:
        # ✅ Align features
        aligned_row = align_features_with_model(patient_row, cph)

        # ✅ Raw risk
        raw_risk = cph.predict_partial_hazard(
            aligned_row.to_frame().T
        ).iloc[0]

        # ✅ Contributions
        contributions = get_contributions(cph, aligned_row)

        # ✅ Explanation (FIXED CALL)
        explanation, top_features, risk_score = generate_explanation(
            aligned_row, contributions, raw_risk, cph
        )

        # ✅ RAG Evidence
        evidence = {}
        for feat, _ in top_features:
            evidence[feat] = get_rag_context(feat)

        # ✅ Safety wording
        if "diagnosis" in explanation.lower():
            explanation = explanation.replace("diagnosis", "assessment")

        return risk_score, explanation, evidence

    except Exception as e:
        print("❌ Error in run_agent:", e)
        return None, "Error occurred", {}


# -------------------------------
# 🛡️ SAFE INPUT HANDLING
# -------------------------------
def safe_int_input(prompt, default=0):
    val = input(prompt).strip()
    if val == "":
        print(f"⚠️ Empty input → using default {default}")
        return default
    try:
        return int(val)
    except:
        print(f"⚠️ Invalid input → using default {default}")
        return default


def safe_float_input(prompt, default=0.0):
    val = input(prompt).strip()
    if val == "":
        print(f"⚠️ Empty input → using default {default}")
        return default
    try:
        return float(val)
    except:
        print(f"⚠️ Invalid input → using default {default}")
        return default


# -------------------------------
# 👤 CUSTOM PATIENT TEST
# -------------------------------
def run_custom_patient(cph):
    print("\n===== CUSTOM PATIENT TEST =====")

    patient = {
        "age": safe_int_input("Enter age: "),
        "gender": safe_int_input("Enter gender (0=Female, 1=Male): "),
        "Pain_score": safe_float_input("Enter pain score (0-10): "),
        "JawClickChewing": safe_int_input("Jaw clicking? (0/1): "),
        "JawGratingGrinding": safe_int_input("Jaw grinding? (0/1): "),
        "BVTV_change": safe_float_input("Enter BVTV change: "),
        "C_BVTV_BL": safe_float_input("Enter baseline bone volume: ")
    }

    row = pd.Series(patient)

    risk, explanation, evidence = run_agent(row, cph)

    if risk is None:
        print("❌ Could not compute risk")
        return

    print("\n==============================")
    print("Custom Patient Input:")
    print(patient)

    print(f"\nNormalized Risk Score: {risk:.4f}")

    print("\nExplanation:")
    print(explanation)

    print("\nEvidence:")
    for feat, contexts in evidence.items():
        print(f"\n[{feat}]")
        for c in contexts:
            print("-", c)


# -------------------------------
# 🚀 MAIN PIPELINE
# -------------------------------
def main():
    print("\n===== AGENT PIPELINE EXECUTION =====")

    df = pd.read_csv(PROCESSED_DATA_PATH + "/survival_data.csv")

    # Load model
    with open(PROCESSED_DATA_PATH + "/cox_model.pkl", "rb") as f:
        cph = pickle.load(f)

    # Feature engineering
    df["BVTV_change"] = df["C_BVTV_FU"] - df["C_BVTV_BL"]
    df["Pain_score"] = df["Facial_Pain_Intensity"]

    features = [
        "age", "gender", "Pain_score",
        "JawClickChewing", "JawGratingGrinding",
        "BVTV_change", "C_BVTV_BL"
    ]

    df_clean = df[features + ["event", "time"]].dropna()

    df_clean["gender"] = df_clean["gender"].astype("category").cat.codes

    X = df_clean.drop(columns=["event", "time"])

    df_ids = df["Patient"].iloc[:3]
    sample = X.iloc[:3]

    for i in range(len(sample)):
        row = sample.iloc[i]

        risk, explanation, evidence = run_agent(row, cph)

        if risk is None:
            continue

        print("\n==============================")
        print(f"Patient ID: {df_ids.iloc[i]}")
        print(f"Normalized Risk Score: {risk:.4f}")

        print("\nExplanation:")
        print(explanation)

        print("\nEvidence:")
        for feat, contexts in evidence.items():
            print(f"\n[{feat}]")
            for c in contexts:
                print("-", c)

    # 🔥 Custom test
    run_custom_patient(cph)


if __name__ == "__main__":
    main()