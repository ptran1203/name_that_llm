"""Prints a comparison table of every run under experiments/, best mean score first."""
import pandas as pd

from exp_log import load_experiments

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    df = load_experiments()
    if df.empty:
        print("no experiments logged yet")
    else:
        cols = [c for c in df.columns if c not in ("git_commit",)]
        print(df[cols].to_string(index=False))
