import pytest

from geoscore_de.address.base import BaseStructAddressRetriever
from geoscore_de.address.models import Position, StructAddress


# Create a concrete implementation for testing
class ConcreteAddressRetriever(BaseStructAddressRetriever):
    """Concrete implementation of BaseStructAddressRetriever for testing."""

    def _get_struct_address(self, raw_address: str) -> StructAddress | None:
        """Minimal implementation for testing - returns None."""
        return None


@pytest.fixture
def retriever():
    """Fixture providing a concrete instance of the abstract class."""
    return ConcreteAddressRetriever()


def test_base_struct_address_retriever_initialization(retriever):
    """Test that the retriever initializes and loads the GeoJSON."""
    assert retriever.geojson is not None
    assert len(retriever.geojson) > 0


def test_get_area_metadata_rostock(retriever):
    """Test get_area_metadata with Rostock coordinates."""
    # Rostock, Germany coordinates
    latitude = 54.0888
    longitude = 12.1359

    result = retriever.get_area_metadata(latitude, longitude)

    assert result is not None
    assert "AGS" in result
    assert "GEN" in result  # Municipality name
    assert result["GEN"] == "Rostock"


def test_get_area_metadata_hamburg(retriever):
    """Test get_area_metadata with Hamburg coordinates."""
    # Hamburg, Germany coordinates
    latitude = 53.5511
    longitude = 9.9937

    result = retriever.get_area_metadata(latitude, longitude)

    assert result is not None
    assert "AGS" in result
    assert "GEN" in result
    assert result["GEN"] == "Hamburg"


def test_get_area_metadata_outside_germany(retriever):
    """Test get_area_metadata with coordinates outside Germany."""
    # Paris, France coordinates (should not be in the dataset)
    latitude = 48.8566
    longitude = 2.3522

    result = retriever.get_area_metadata(latitude, longitude)

    assert result is None


def test_get_area_metadata_contains_destatis(retriever):
    """Test that area metadata contains destatis information."""
    latitude = 54.0888
    longitude = 12.1359

    result = retriever.get_area_metadata(latitude, longitude)

    assert result is not None
    assert "destatis" in result

    destatis_data = result["destatis"]
    assert "population" in destatis_data
    assert "area" in destatis_data


def test_get_ags_valid_position(retriever):
    """Test get_ags with valid Position object."""
    position = Position(latitude=54.0888, longitude=12.1359)

    ags = retriever.get_ags(position)

    assert ags is not None
    assert isinstance(ags, str)
    assert len(ags) > 0


def test_get_ags_outside_germany(retriever):
    """Test get_ags with position outside Germany."""
    position = Position(latitude=48.8566, longitude=2.3522)  # Paris

    ags = retriever.get_ags(position)

    assert ags is None


def test_get_struct_address_with_none_result(retriever):
    """Test get_struct_address when _get_struct_address returns None."""
    result = retriever.get_struct_address("some address")

    assert result is None


def test_get_struct_address_with_valid_result():
    """Test get_struct_address when _get_struct_address returns a valid address."""

    # Create a more sophisticated test implementation
    class MockRetriever(BaseStructAddressRetriever):
        def _get_struct_address(self, raw_address: str) -> StructAddress | None:
            # Return a mock address with Rostock coordinates
            return StructAddress(
                name="Test Street 1, 18055 Rostock, Germany",
                street="Test Street",
                municipality="Rostock",
                region="Mecklenburg-Vorpommern",
                postal_code="18055",
                country="Germany",
                country_code="DE",
                position=Position(latitude=54.0888, longitude=12.1359),
            )

    retriever = MockRetriever()
    result = retriever.get_struct_address("Test Street 1, Rostock")

    assert result is not None
    assert result.AGS is not None
    assert isinstance(result.AGS, str)
