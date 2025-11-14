# src/paths.py
"""
Centralized path management for the Business Outcome Predictor project.

This file:
- auto-detects the project root
- sets DATA, OUTPUT, FIG, REPORT, MODEL directories
- auto-creates missing folders
- ensures all scripts always use correct file paths
"""

import os

# ------------------------------------------------------------------------------
# 1. Resolve main directories
# ------------------------------------------------------------------------------

# Absolute path to THIS file (src/paths.py)
THIS_FILE = os.path.abspath(__file__)

# src directory
SRC_DIR = os.path.dirname(THIS_FILE)

# project root (parent of src/)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# ------------------------------------------------------------------------------
# 2. Data directory
# ------------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ------------------------------------------------------------------------------
# 3. Outputs directory + subdirectories
# ------------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

# ------------------------------------------------------------------------------
# 4. Create directories automatically
# ------------------------------------------------------------------------------
for folder in [OUTPUT_DIR, FIG_DIR, REPORT_DIR, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

# ------------------------------------------------------------------------------
# 5. Debug helper (optional)
# ------------------------------------------------------------------------------
def summarize_paths():
    print("\n--- PATH SUMMARY ---")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("FIG_DIR:", FIG_DIR)
    print("REPORT_DIR:", REPORT_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("---------------------\n")

if __name__ == "__main__":
    summarize_paths()