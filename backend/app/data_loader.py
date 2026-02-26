import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "titanic.csv"

def load_titanic_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.lower().str.strip()
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].astype(str).str.strip()

    return df
