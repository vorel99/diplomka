import pandas as pd


def load_municipality_data(path: str) -> pd.DataFrame:
    """Load municipality data from a CSV file.

    Args:
        path (str): Path to the CSV file.
    """
    df = pd.read_csv(path, skiprows=3, sep=";", skipfooter=4, engine="python")
    df.rename(columns={"Unnamed: 0": "MU_ID", "Unnamed: 1": "Municipality"}, inplace=True)
    # drop first two rows which contain metadata about the file
    df = df.iloc[2:]
    df
    return df
