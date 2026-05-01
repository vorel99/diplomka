import pandas as pd

from geoscore_de.data_flow.feature_engineering.kbins_binning import KBinsDiscretizerBinning


def test_kbins_binning_fits_without_ags_and_preserves_missing_values():
    df = pd.DataFrame({"value": [1.0, 2.0, None, 4.0, 5.0, 6.0]})

    transformer = KBinsDiscretizerBinning(input_columns=["value"], output_column="binned", n_bins=2)
    transformer.fit(df)
    result = transformer.transform(df)

    assert "binned" in result.columns
    assert pd.isna(result.loc[2, "binned"])
    assert result.loc[[0, 1, 3, 4, 5], "binned"].notna().all()


def test_kbins_binning_constant_values_yield_missing_bins():
    df = pd.DataFrame({"value": [5.0, 5.0, 5.0, 5.0]})

    transformer = KBinsDiscretizerBinning(input_columns=["value"], output_column="binned", n_bins=4)
    transformer.fit(df)
    result = transformer.transform(df)

    assert pd.isna(result["binned"]).all()
