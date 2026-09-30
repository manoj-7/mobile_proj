import pandas as pd
from sklearn.model_selection import train_test_split


def split_file(input_csv: str, train_out: str, test_out: str, test_size=0.2, random_state=None):
    df = pd.read_csv(input_csv)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)
