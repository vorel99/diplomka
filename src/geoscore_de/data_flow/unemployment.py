import pandas as pd


def load_unemployment_data(path: str) -> pd.DataFrame:
    """Load unemployment data from a CSV file.
    Data were obtained from the GENESIS-Online regional statistics database of the German Federal Statistical Office.
    https://www.regionalstatistik.de/genesis//online?operation=table&code=13211-01-03-5&bypass=true&levelindex=1&levelid=1768376127943#abreadcrumb

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded unemployment data.
    """
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        skiprows=7,
        skipfooter=4,
        engine="python",
        na_values=["-", "."],
    )
    df.rename(
        columns={"Unnamed: 0": "MU_ID", "Unnamed: 1": "Municipality", "Unnamed: 2": "unemployment_total"}, inplace=True
    )

    # drop first row which contains column descriptions
    df = df.iloc[1:].reset_index(drop=True)

    df["MU_ID"] = df["MU_ID"].astype(str)
    df["unemployment_total"] = pd.to_numeric(df["unemployment_total"])

    # fill MU_ID on the right with zeros to a total length of 8 characters
    df["AGS"] = df["MU_ID"].str.ljust(8, "0")
    # enrich data with land, kreis, gemeinde
    df["Land"] = df["MU_ID"].str[:2]
    df["Kreis"] = df["MU_ID"].str[2:5]
    df["Gemeinde"] = df["MU_ID"].str[5:]

    return df
