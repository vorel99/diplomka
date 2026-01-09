import pandas as pd


def load_population_data(path: str) -> pd.DataFrame:
    """Load population data from a CSV file.
    Data were obtained from https://www.regionalstatistik.de/genesis//online?operation=table&code=12411-02-03-5&bypass=true&levelindex=1&levelid=1765292926381#abreadcrumb

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded population data.
    """
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        skiprows=5,
        skipfooter=4,
        na_values=["-", "."],
    )
    df.rename(
        columns={
            "Unnamed: 0": "date",
            "Unnamed: 1": "MU_ID",
            "Unnamed: 2": "Municipality",
            "Unnamed: 3": "age_group",
            "Insgesamt": "people_count",
            "männlich": "male_count",
            "weiblich": "female_count",
        },
        inplace=True,
    )

    # add AGS column by filling MU_ID to have trailing zeros if necessary (to have 8 characters)
    df["AGS"] = df["MU_ID"].str.ljust(8, "0")

    return df
