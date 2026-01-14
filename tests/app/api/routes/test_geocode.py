"""Tests for geocoding API endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from geoscore_de.address.models import Position, StructAddress
from geoscore_de.app.config import Settings
from geoscore_de.app.main import create_app


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(
        _env_file=None,
        **{
            "mapy_com_api_key": "test-api-key",
            "geojson_path": "test/path/data.geojson",
        },
    )


@pytest.fixture
def client(test_settings):
    """Create test client with test settings."""
    app = create_app(settings=test_settings)
    return TestClient(app)


@pytest.fixture
def mock_struct_address():
    """Create a mock StructAddress for testing."""
    return StructAddress(
        AGS="01001000",
        name="Hauptstraße 1, 24103 Kiel, Deutschland",
        street="Hauptstraße 1",
        municipality="Kiel",
        region="Schleswig-Holstein",
        postal_code="24103",
        country="Deutschland",
        country_code="DE",
        position=Position(latitude=54.3233, longitude=10.1394),
    )


def test_geocode_success(client, mock_struct_address):
    """Test successful geocoding of an address."""
    with patch("geoscore_de.app.api.routes.geocode.MapyComStructAddressRetriever") as mock_retriever_class:
        # Setup mock
        mock_retriever = MagicMock()
        mock_retriever.get_struct_address.return_value = mock_struct_address
        mock_retriever_class.return_value = mock_retriever

        # Make request
        response = client.post("/api/v1/geocode", json={"address": "Hauptstraße 1, Kiel"})

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["address"] is not None
        assert data["address"]["AGS"] == "01001000"
        assert data["address"]["municipality"] == "Kiel"
        assert data["address"]["street"] == "Hauptstraße 1"
        assert data["error"] is None

        # Verify retriever was called correctly
        mock_retriever_class.assert_called_once_with(api_key="test-api-key", geojson_path="test/path/data.geojson")
        mock_retriever.get_struct_address.assert_called_once_with("Hauptstraße 1, Kiel")


def test_geocode_address_not_found(client):
    """Test geocoding when address cannot be found."""
    with patch("geoscore_de.app.api.routes.geocode.MapyComStructAddressRetriever") as mock_retriever_class:
        # Setup mock to return None
        mock_retriever = MagicMock()
        mock_retriever.get_struct_address.return_value = None
        mock_retriever_class.return_value = mock_retriever

        # Make request
        response = client.post("/api/v1/geocode", json={"address": "NonexistentAddress123"})

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["address"] is None
        assert "Could not geocode address" in data["error"]
        assert "NonexistentAddress123" in data["error"]


def test_geocode_error_handling(client):
    """Test error handling when geocoding raises an exception."""
    with patch("geoscore_de.app.api.routes.geocode.MapyComStructAddressRetriever") as mock_retriever_class:
        # Setup mock to raise exception
        mock_retriever = MagicMock()
        mock_retriever.get_struct_address.side_effect = Exception("API connection error")
        mock_retriever_class.return_value = mock_retriever

        # Make request
        response = client.post("/api/v1/geocode", json={"address": "Test Address"})

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["address"] is None
        assert "Error geocoding address" in data["error"]
        assert "API connection error" in data["error"]


def test_geocode_missing_address_field(client):
    """Test request validation when address field is missing."""
    response = client.post("/api/v1/geocode", json={})

    # Should return 422 Unprocessable Entity for validation error
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_geocode_empty_address(client, mock_struct_address):
    """Test geocoding with an empty address string."""
    with patch("geoscore_de.app.api.routes.geocode.MapyComStructAddressRetriever") as mock_retriever_class:
        mock_retriever = MagicMock()
        mock_retriever.get_struct_address.return_value = mock_struct_address
        mock_retriever_class.return_value = mock_retriever

        # Make request with empty string
        response = client.post("/api/v1/geocode", json={"address": ""})

        # Should return 422 Unprocessable Entity for validation error
        assert response.status_code == 422
        mock_retriever.get_struct_address.assert_not_called()


def test_geocode_address_without_ags(client):
    """Test geocoding when address has no AGS code."""
    address_without_ags = StructAddress(
        AGS=None,
        name="Foreign Address, Paris, France",
        street="Rue de Example",
        municipality="Paris",
        region="Île-de-France",
        postal_code="75001",
        country="France",
        country_code="FR",
        position=Position(latitude=48.8566, longitude=2.3522),
    )

    with patch("geoscore_de.app.api.routes.geocode.MapyComStructAddressRetriever") as mock_retriever_class:
        mock_retriever = MagicMock()
        mock_retriever.get_struct_address.return_value = address_without_ags
        mock_retriever_class.return_value = mock_retriever

        response = client.post("/api/v1/geocode", json={"address": "Paris, France"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["address"]["AGS"] is None
        assert data["address"]["municipality"] == "Paris"


def test_geocode_special_characters(client, mock_struct_address):
    """Test geocoding with special characters in address."""
    with patch("geoscore_de.app.api.routes.geocode.MapyComStructAddressRetriever") as mock_retriever_class:
        mock_retriever = MagicMock()
        mock_retriever.get_struct_address.return_value = mock_struct_address
        mock_retriever_class.return_value = mock_retriever

        # Address with umlauts and special characters
        address = "Münchner Straße 123, München"
        response = client.post("/api/v1/geocode", json={"address": address})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_retriever.get_struct_address.assert_called_once_with(address)


def test_geocode_invalid_json(client):
    """Test request with invalid JSON."""
    response = client.post("/api/v1/geocode", data="invalid json", headers={"Content-Type": "application/json"})

    assert response.status_code == 422


def test_geocode_response_model_structure(client, mock_struct_address):
    """Test that response follows the expected model structure."""
    with patch("geoscore_de.app.api.routes.geocode.MapyComStructAddressRetriever") as mock_retriever_class:
        mock_retriever = MagicMock()
        mock_retriever.get_struct_address.return_value = mock_struct_address
        mock_retriever_class.return_value = mock_retriever

        response = client.post("/api/v1/geocode", json={"address": "Test"})

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "success" in data
        assert "address" in data
        assert "error" in data

        # Verify address structure when present
        assert "AGS" in data["address"]
        assert "name" in data["address"]
        assert "street" in data["address"]
        assert "municipality" in data["address"]
        assert "region" in data["address"]
        assert "postal_code" in data["address"]
        assert "country" in data["address"]
        assert "country_code" in data["address"]
        assert "position" in data["address"]
        assert "latitude" in data["address"]["position"]
        assert "longitude" in data["address"]["position"]
