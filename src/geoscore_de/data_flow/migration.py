import pandas as pd


def load_migration_data(path: str) -> pd.DataFrame:
    """Load migration data from a CSV file.
    Data were obtained from the GENESIS-Online regional statistics database of the German Federal Statistical Office.
    https://www.regionalstatistik.de/genesis//online?operation=table&code=12711-91-01-5&bypass=true&levelindex=0&levelid=1768376496916#abreadcrumb

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded migration data.
    """
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        skiprows=5,
        skipfooter=4,
        engine="python",
        na_values=["-", "."],
        names=["Year", "MU_ID", "Municipality", "in_migration", "out_migration"],
        header=None,
    )

    # drop first row which contains column descriptions
    df = df.iloc[1:].reset_index(drop=True)

    df["MU_ID"] = df["MU_ID"].astype(str)
    df["in_migration"] = pd.to_numeric(df["in_migration"])
    df["out_migration"] = pd.to_numeric(df["out_migration"])

    # fill MU_ID on the right with zeros to a total length of 8 characters
    df["AGS"] = df["MU_ID"].str.ljust(8, "0")

    return df
