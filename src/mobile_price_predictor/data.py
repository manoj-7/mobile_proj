import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data and do minimal cleaning.

    Expects Kaggle mobile price dataset columns.
    """
    df = pd.read_csv(path)
    # basic cleanup placeholder
    df = df.copy()
    return df
