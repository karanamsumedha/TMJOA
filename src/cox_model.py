import pandas as pd
import numpy as np
import pickle

from lifelines import CoxPHFitter
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config import *


def main():
    print("\n===== LOADING DATA =====")

    df = pd.read_csv(PROCESSED_DATA_PATH + "/survival_data.csv")
    print("Original Shape:", df.shape)

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    print("\nCreating derived features...")

    df["BVTV_change"] = df["C_BVTV_FU"] - df["C_BVTV_BL"]
    df["Pain_score"] = df["Facial_Pain_Intensity"]

    # -------------------------------
    # SELECT FEATURES
    # -------------------------------
    features = [
        "age",
        "gender",
        "Pain_score",
        "JawClickChewing",
        "JawGratingGrinding",
        "BVTV_change",
        "C_BVTV_BL",
        "event",
        "time"
    ]

    df = df[features]

    # -------------------------------
    # HANDLE MISSING VALUES (FIX 🔥)
    # -------------------------------
    print("\nHandling missing values (NO ROW DROPS)...")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    print("After filling missing values:", df.shape)

    # -------------------------------
    # ENCODE CATEGORICAL
    # -------------------------------
    df["gender"] = df["gender"].astype("category").cat.codes

    # -------------------------------
    # HANDLE IMBALANCE
    # -------------------------------
    df["weight"] = df["event"].apply(lambda x: 2 if x == 1 else 1)

    # ===============================
    # 🔁 STRATIFIED K-FOLD
    # ===============================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    c_index_scores = []
    acc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_scores = []

    fold = 1

    for train_idx, test_idx in skf.split(df, df["event"]):
        print(f"\n===== FOLD {fold} =====")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # -------------------------------
        # COX MODEL
        # -------------------------------
        cph = CoxPHFitter()
        cph.fit(train_df, duration_col="time", event_col="event", weights_col="weight")

        c_index = cph.score(test_df, scoring_method="concordance_index")
        print("C-index:", c_index)
        c_index_scores.append(c_index)

        # -------------------------------
        # LOGISTIC REGRESSION
        # -------------------------------
        X_train = train_df.drop(columns=["event", "time", "weight"])
        y_train = train_df["event"]

        X_test = test_df.drop(columns=["event", "time", "weight"])
        y_test = test_df["event"]

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]

        acc_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        roc_scores.append(roc_auc_score(y_test, y_prob))

        fold += 1

    # ===============================
    # 📊 FINAL AVERAGE RESULTS
    # ===============================
    print("\n===== FINAL AVERAGE RESULTS =====")

    print("Cox C-Index:", np.mean(c_index_scores))
    print("Accuracy:", np.mean(acc_scores))
    print("Precision:", np.mean(precision_scores))
    print("Recall:", np.mean(recall_scores))
    print("F1 Score:", np.mean(f1_scores))
    print("ROC-AUC:", np.mean(roc_scores))

    # ===============================
    # SAVE FINAL MODEL (FULL DATA)
    # ===============================
    print("\n===== TRAINING FINAL MODEL ON FULL DATA =====")

    cph_final = CoxPHFitter()
    cph_final.fit(df, duration_col="time", event_col="event", weights_col="weight")

    model_path = PROCESSED_DATA_PATH + "/cox_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(cph_final, f)

    print(f"\n✅ Final Cox model saved at: {model_path}")


if __name__ == "__main__":
    main()