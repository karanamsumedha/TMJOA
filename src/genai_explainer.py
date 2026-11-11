import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import pickle
from config import *
import warnings
warnings.filterwarnings("ignore")

# ✅ RAG IMPORTS
from rag_module import load_documents, create_vector_db, retrieve_context

# -------------------------------
# ✅ RAG INITIALIZATION (RUN ONCE)
# -------------------------------
folder_path = os.path.join("data", "knowledge")

texts = load_documents(folder_path)
db = create_vector_db(texts)

# -------------------------------
# ✅ CLEAN FEATURE NAME
# -------------------------------
def clean_name(feat):
    return feat.replace("_", " ").capitalize()


# -------------------------------
# ✅ NEW: RISK NORMALIZATION
# -------------------------------
def normalize_risk(risk, cph):
    baseline = cph.predict_partial_hazard(
        cph._norm_mean.to_frame().T
    ).iloc[0]
    return risk / (risk + baseline)


# -------------------------------
# CONTRIBUTION LOGIC
# -------------------------------
def get_contributions(cph, row):
    coefficients = cph.params_
    contributions = {}

    for feature in row.index:
        contributions[feature] = row[feature] * coefficients[feature]

    return contributions


# -------------------------------
# ✅ RAG QUERY MAPPING + FILTER
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
    context = retrieve_context(db, query)

    # ✅ FIX: feature-specific filtering
    keyword = feature.split("_")[0].lower()
    filtered = [c for c in context if keyword in c.lower()]

    return filtered[:2] if filtered else context[:2]


# -------------------------------
# NORMALIZATION
# -------------------------------
def normalize_contributions(contributions):
    total = sum(abs(v) for v in contributions.values())

    if total == 0:
        return {k: 0 for k in contributions}

    return {k: (abs(v) / total) * 100 for k, v in contributions.items()}


def get_strength(percent):
    if percent > 40:
        return "strong"
    elif percent > 20:
        return "moderate"
    else:
        return "mild"


# -------------------------------
# FEATURE INTERPRETATION (FIXED)
# -------------------------------
def interpret_feature(feat, percent, row, contributions):
    strength = get_strength(percent)

    if feat == "BVTV_change":
        if row["BVTV_change"] < 0:
            return f"A decrease in trabecular bone volume shows a {strength} contribution to structural degradation."
        else:
            return f"Stable bone volume indicates a {strength} protective effect."

    elif feat == "Pain_score":
        if row["Pain_score"] > 5:
            return f"High facial pain intensity has a {strength} impact on disease progression."
        else:
            return f"Pain levels are relatively controlled with a {strength} influence."

    elif feat == "JawClickChewing":
        if row["JawClickChewing"] == 1:
            return f"Jaw clicking during chewing reflects joint dysfunction with a {strength} effect."

    elif feat == "JawGratingGrinding":
        if row["JawGratingGrinding"] == 1:
            return f"Grinding sounds indicate joint stress with a {strength} contribution."

    elif feat == "C_BVTV_BL":
        # ✅ FIX: use actual contribution sign
        if contributions["C_BVTV_BL"] > 0:
            return f"Higher baseline bone volume is associated with progression risk ({strength} effect)."
        else:
            return f"Baseline bone condition appears stable with a {strength} protective role."

    elif feat == "age":
        return f"Age contributes a {strength} influence on disease progression."

    elif feat == "gender":
        if row["gender"] == 1:
            return f"Male gender shows a {strength} association with TMJ risk."
        else:
            return f"Female gender shows a {strength} association with TMJ risk."

    return f"{feat} has a {strength} effect on risk."


# -------------------------------
# ✅ EXPLANATION GENERATION (FIXED)
# -------------------------------
def generate_explanation(row, contributions, raw_risk, cph):

    # ✅ FIX: normalize risk
    risk_score = normalize_risk(raw_risk, cph)

    norm_contrib = normalize_contributions(contributions)

    sorted_features = sorted(
        norm_contrib.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_features = sorted_features[:3]

    # ✅ FIXED thresholds
    if risk_score < 0.3:
        level = "LOW"
    elif risk_score < 0.7:
        level = "MODERATE"
    else:
        level = "HIGH"

    top_feat, top_percent = top_features[0]

    explanation = f"The patient is at {level} risk of TMJ progression. "
    explanation += f"The primary driver of risk is {clean_name(top_feat)}, contributing approximately {top_percent:.1f}% of the total risk. "

    for feat, percent in top_features:
        explanation += f"{clean_name(feat)} contributes about {percent:.1f}% of the overall risk. "
        explanation += interpret_feature(feat, percent, row, contributions) + " "

    explanation += "This assessment is for educational purposes only and not a medical diagnosis."

    return explanation, top_features, risk_score


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    print("\n===== GENERATING EXPLANATIONS =====")

    df = pd.read_csv(PROCESSED_DATA_PATH + "/survival_data.csv")

    with open(PROCESSED_DATA_PATH + "/cox_model.pkl", "rb") as f:
        cph = pickle.load(f)

    df["BVTV_change"] = df["C_BVTV_FU"] - df["C_BVTV_BL"]
    df["Pain_score"] = df["Facial_Pain_Intensity"]

    features = [
        "age", "gender", "Pain_score",
        "JawClickChewing", "JawGratingGrinding",
        "BVTV_change", "C_BVTV_BL"
    ]

    df = df[features + ["event", "time"]].dropna()
    df["gender"] = df["gender"].astype("category").cat.codes

    X = df.drop(columns=["event", "time"])
    sample = X.iloc[:3]

    df_full = pd.read_csv(PROCESSED_DATA_PATH + "/survival_data.csv")
    patient_ids = df_full["Patient"].iloc[:3]

    raw_risks = cph.predict_partial_hazard(sample)

    for i in range(len(sample)):
        row = sample.iloc[i]
        raw_risk = raw_risks.iloc[i]

        contributions = get_contributions(cph, row)

        explanation, top_features, risk_score = generate_explanation(
            row, contributions, raw_risk, cph
        )

        print(f"\n--- Patient ID: {patient_ids.iloc[i]} ---")
        print(f"Raw Risk: {raw_risk:.4f}")
        print(f"Normalized Risk: {risk_score:.4f}")

        print("\nExplanation:")
        print(explanation)

        print("\nRetrieved Evidence:")

        for feat, _ in top_features:
            contexts = get_rag_context(feat)

            print(f"\n[{feat}]")

            seen = set()
            for c in contexts:
                if c not in seen:
                    print("-", c)
                    seen.add(c)


if __name__ == "__main__":
    main()