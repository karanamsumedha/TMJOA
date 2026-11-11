import pandas as pd
import numpy as np
from config import *


def create_event(row):
    try:
        baseline = (
            float(row["Baseline_Health_status_x"]) +
            float(row["Baseline_Health_status_y"])
        ) / 2

        followup = (
            float(row["Follow_up_Health_status_x"]) +
            float(row["Follow_up_Health_status_y"])
        ) / 2

        # Health deterioration
        health_event = 1 if followup < baseline else 0

        # Radiomic degradation
        bone_event = 1 if row["C_BVTV_FU"] < row["C_BVTV_BL"] else 0

        return 1 if (health_event or bone_event) else 0

    except:
        return 0


def main():
    print("\n===== LOADING MERGED DATA =====")

    df = pd.read_csv(MERGED_FILE)
    print("Shape:", df.shape)

    # -------------------------------
    # CREATE EVENT COLUMN
    # -------------------------------
    print("\nCreating event column...")
    df["event"] = df.apply(create_event, axis=1)

    # -------------------------------
    # CREATE TIME COLUMN (UPDATED)
    # -------------------------------
    print("Creating simulated time column...")

    np.random.seed(42)
    df["time"] = np.random.uniform(0.5, 3.0, size=len(df))

    # -------------------------------
    # BASIC CHECKS
    # -------------------------------
    print("\nEvent distribution:")
    print(df["event"].value_counts())

    print("\nEvent distribution (percentage):")
    print(df["event"].value_counts(normalize=True) * 100)

    print("\nTime summary:")
    print(df["time"].describe())

    # -------------------------------
    # SAVE
    # -------------------------------
    output_path = PROCESSED_DATA_PATH + "/survival_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Saved survival dataset to: {output_path}")


if __name__ == "__main__":
    main()