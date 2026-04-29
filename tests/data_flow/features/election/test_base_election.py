from unittest.mock import patch

import pandas as pd
import pytest

from geoscore_de.data_flow.features.election_21 import Election21Feature


class TestFixMissing:
    """Tests for the _fix_missing method of Election21Feature."""

    @pytest.fixture
    def mock_municipality_data_simple(self):
        """Create simple mock municipality data."""
        return pd.DataFrame(
            {"AGS": ["01001000", "01002000", "01003000"], "Municipality": ["Flensburg", "Kiel", "Lübeck"]}
        )

    @pytest.fixture
    def mock_election_data_simple(self):
        """Create simple mock election data."""
        return pd.DataFrame(
            {
                "AGS": ["01001000", "01002000"],
                "Gemeinde Name": ["Flensburg", "Kiel"],
                "eligible_voters": [100, 200],
                "total_voters": [80, 160],
                "E_CDU": [30, 60],
                "E_SPD": [20, 50],
                "Z_CDU": [32, 62],
                "Z_SPD": [22, 52],
            }
        )

    def test_fix_missing_no_missing_municipalities(self, mock_election_data_simple, mock_municipality_data_simple):
        """Test when there are no missing municipalities - data unchanged."""
        # All municipalities in election data are also in municipality data
        election_data = mock_election_data_simple.copy()
        municipality_data = mock_municipality_data_simple[
            mock_municipality_data_simple["AGS"].isin(["01001000", "01002000"])
        ]

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election.base.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = municipality_data

            result = feature._fix_missing(election_data.copy(), muni_name_col="Gemeinde Name")

        # Since there are no missing municipalities, data should be unchanged
        assert len(result) == len(election_data)
        assert list(result["AGS"].values) == list(election_data["AGS"].values)

    def test_fix_missing_finds_einschl_municipality(self, mock_election_data_simple, mock_municipality_data_simple):
        """Test fixing missing municipality found in einschl. municipality."""
        # Create election data where Lübeck is missing but included in Flensburg
        election_data = pd.DataFrame(
            {
                "AGS": ["01001000", "01002000"],
                "Gemeinde Name": ["Flensburg (einschl. Lübeck)", "Kiel"],
                "eligible_voters": [150, 200],
                "total_voters": [120, 160],
                "E_CDU": [45, 60],
                "E_SPD": [30, 50],
                "Z_CDU": [48, 62],
                "Z_SPD": [32, 52],
            }
        )

        # Municipality data includes all three municipalities
        municipality_data = pd.DataFrame(
            {"AGS": ["01001000", "01002000", "01003000"], "Municipality": ["Flensburg", "Kiel", "Lübeck"]}
        )

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election.base.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = municipality_data

            result = feature._fix_missing(election_data.copy(), muni_name_col="Gemeinde Name")

        # Should have one more row (Lübeck added)
        assert len(result) > len(election_data)
        # Check that Lübeck's AGS is now present
        assert "01003000" in result["AGS"].values
        # Verify the added row has data from Flensburg
        luebeck_row = result[result["AGS"] == "01003000"].iloc[0]
        assert luebeck_row["eligible_voters"] == 150
        assert luebeck_row["E_CDU"] == 45

    def test_fix_missing_preserves_original_data(self, mock_election_data_simple, mock_municipality_data_simple):
        """Test that _fix_missing preserves original data for non-missing municipalities."""
        election_data = mock_election_data_simple.copy()
        original_eligible_voters = election_data.iloc[0]["eligible_voters"]
        original_ags = election_data.iloc[0]["AGS"]

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election.base.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = mock_municipality_data_simple

            result = feature._fix_missing(election_data.copy(), muni_name_col="Gemeinde Name")

        # Original data should still be present
        original_row = result[result["AGS"] == original_ags].iloc[0]
        assert original_row["eligible_voters"] == original_eligible_voters

    def test_fix_missing_copies_row_correctly(self):
        """Test that copied rows have correct values from the einschl. municipality."""
        election_data = pd.DataFrame(
            {
                "AGS": ["01001000"],
                "Gemeinde Name": ["Flensburg (einschl. Lübeck)"],
                "eligible_voters": [150],
                "total_voters": [120],
                "E_CDU": [45],
                "E_SPD": [30],
                "Z_CDU": [48],
                "Z_SPD": [32],
            }
        )

        municipality_data = pd.DataFrame({"AGS": ["01001000", "01003000"], "Municipality": ["Flensburg", "Lübeck"]})

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election.base.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = municipality_data

            result = feature._fix_missing(election_data.copy(), muni_name_col="Gemeinde Name")

        # Find the added row (Lübeck)
        luebeck_row = result[result["AGS"] == "01003000"]
        assert len(luebeck_row) == 1

        # Check that it copied the values from Flensburg
        luebeck_row = luebeck_row.iloc[0]
        assert luebeck_row["eligible_voters"] == 150
        assert luebeck_row["total_voters"] == 120
        assert luebeck_row["E_CDU"] == 45
        assert luebeck_row["Z_SPD"] == 32

    def test_fix_missing_no_match_not_added(self, mock_election_data_simple):
        """Test that municipalities not found in einschl. are not added."""

        municipality_data = pd.DataFrame(
            {"AGS": ["01001000", "01002000", "01003000"], "Municipality": ["Flensburg", "Kiel", "NonExistentCity"]}
        )

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election.base.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = municipality_data

            result = feature._fix_missing(mock_election_data_simple.copy(), muni_name_col="Gemeinde Name")

        # NonExistentCity should not be added (not mentioned in any einschl. name)
        assert "01003000" not in result["AGS"].values
        # Original data should remain unchanged
        assert len(result) == len(mock_election_data_simple)

    def test_fix_missing_disabled(self, mock_election_data_simple):
        """Test that fix_missing is disabled when flag is False."""
        election_data = mock_election_data_simple.copy()

        feature = Election21Feature(fix_missing=False)

        # When fix_missing is False, _fix_missing should not be called on missing data
        # The function should still work but won't add missing municipalities
        result = feature._fix_missing(election_data, muni_name_col="Gemeinde Name")

        # Data should remain unchanged
        assert len(result) == len(election_data)

    def test_fix_missing_with_nan_values(self, mock_municipality_data_simple):
        """Test that _fix_missing handles NaN values gracefully."""
        election_data = pd.DataFrame(
            {
                "AGS": ["01001000", "01002000"],
                "Gemeinde Name": ["Flensburg (einschl. Lübeck)", "Kiel"],
                "eligible_voters": [100.0, 200.0],
                "total_voters": [80.0, float("nan")],
                "E_CDU": [30.0, float("nan")],
                "E_SPD": [20.0, 50.0],
                "Z_CDU": [32.0, 62.0],
                "Z_SPD": [22.0, 52.0],
            }
        )

        feature = Election21Feature(fix_missing=True)

        with patch("geoscore_de.data_flow.features.election.base.MunicipalityFeature") as mock_muni_feature:
            mock_muni_instance = mock_muni_feature.return_value
            mock_muni_instance.load.return_value = mock_municipality_data_simple

            # Should not raise an error
            result = feature._fix_missing(election_data.copy(), muni_name_col="Gemeinde Name")
            assert isinstance(result, pd.DataFrame)
