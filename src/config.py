import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

# File names
CLINICAL_FILE = os.path.join(RAW_DATA_PATH, "clinical.xlsx")
FOSSA_FILE = os.path.join(RAW_DATA_PATH, "fossa.xlsx")
CONDYLE_FILE = os.path.join(RAW_DATA_PATH, "condyle.xlsx")

# Output
MERGED_FILE = os.path.join(PROCESSED_DATA_PATH, "merged.csv")