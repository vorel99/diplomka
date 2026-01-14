import pandas as pd


def load_municipality_data(path: str) -> pd.DataFrame:
    """Load municipality data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded municipality data.
            DataFrame includes columns `AGS` with 8-character municipality codes.
            `MU_ID`, `Municipality`, `Persons`, `Area`, `Population Density` and `AGS`.
    """
    df = pd.read_csv(path, skiprows=3, sep=";", skipfooter=4, engine="python")
    df.rename(columns={"Unnamed: 0": "MU_ID", "Unnamed: 1": "Municipality"}, inplace=True)
    # drop first two rows which contain metadata about the file
    df = df.iloc[2:]

    # Create AGS column by removing the Verbandsgemeinde (collective municipality) level from MU_ID
    df["AGS"] = df["MU_ID"].str.slice(0, 5) + df["MU_ID"].str.slice(9, 12)

    return df
