import pandas as pd


def load_migration_data(path: str) -> pd.DataFrame:
    """Load migration data from a CSV file.
    Data were obtained from https://www.regionalstatistik.de/genesis/online?operation=previous&levelindex=2&step=1&titel=Tabellenaufbau&levelid=1765296626951&levelid=1765296564668#abreadcrumb

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded migration data.
    """
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        skiprows=4,
        skipfooter=4,
        na_values=["-", "."],
    )
    df.rename(
        columns={
            "Unnamed: 0": "Year",
            "Unnamed: 1": "MU_ID",
            "Unnamed: 2": "Municipality",
            "Zuzüge über die Gemeindegrenzen": "in_migration",
            "Fortzüge über die Gemeindegrenzen": "out_migration",
        },
        inplace=True,
    )

    # drop first row which contains column descriptions
    df = df.iloc[1:].reset_index(drop=True)

    df["MU_ID"] = df["MU_ID"].astype(str)
    df["in_migration"] = pd.to_numeric(df["in_migration"])
    df["out_migration"] = pd.to_numeric(df["out_migration"])

    # fill MU_ID to have trailing zeros if necessary (to have 8 characters)
    df["AGS"] = df["MU_ID"].str.ljust(8, "0")

    return df
