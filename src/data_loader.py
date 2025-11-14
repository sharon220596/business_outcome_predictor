import os
import json
import pandas as pd
from paths import DATA_DIR, REPORT_DIR

def load_data(path: str):
   
    if not os.path.isabs(path):
        # allow passing just filename located in DATA_DIR
        candidate = os.path.join(DATA_DIR, path)
        if os.path.exists(candidate):
            path = candidate
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def initial_explore(df, n=5):
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "null_counts": df.isna().sum().to_dict(),
        "sample_head": df.head(n).to_dict(orient="records")
    }
    outfile = os.path.join(REPORT_DIR, "initial_explore.json")
    with open(outfile, "w") as f:
        json.dump(info, f, indent=2)
    return info

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/TelecomCustomerChurn.csv")
    args = parser.parse_args()
    df = load_data(args.data)
    info = initial_explore(df)
    print("Saved:", os.path.join(REPORT_DIR, "initial_explore.json"))
    print(info)