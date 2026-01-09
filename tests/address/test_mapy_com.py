from unittest.mock import patch

from geoscore_de.address.mapy_com import MapyComStructAddressRetriever
from geoscore_de.address.models import Position, StructAddress


def test_mapy_com_struct_address_retriever():
    """
    Test MapyComStructAddressRetriever with mocked API response.
    Testing that the retriever correctly parses the structured address from the mocked response.
    """
    addr = "TEST"

    mock_response = {
        "items": [
            {
                "name": "Test Address, Test City, Test Country",
                "position": {"lat": 12.34, "lon": 56.78},
                "zip": "12345",
                "regionalStructure": [
                    {"type": "regional.street", "name": "Test Street"},
                    {"type": "regional.municipality", "name": "Test City"},
                    {"type": "regional.region", "name": "Test Region"},
                    {"type": "regional.country", "name": "Test Country", "isoCode": "TC"},
                ],
            }
        ]
    }
    expected_position = Position(latitude=12.34, longitude=56.78)
    expected_struct_address = StructAddress(
        name="Test Address, Test City, Test Country",
        street="Test Street",
        municipality="Test City",
        region="Test Region",
        postal_code="12345",
        country="Test Country",
        country_code="TC",
        position=expected_position,
    )

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        retriever = MapyComStructAddressRetriever(api_key="dummy_key")
        struct_address = retriever.get_struct_address(addr)

        assert struct_address == expected_struct_address


def test_mapy_com_struct_address_retriever_no_results():
    """
    Test MapyComStructAddressRetriever with no results from the API.
    Testing that the retriever returns None when no results are found.
    """
    addr = "NON_EXISTENT_ADDRESS"

    mock_response = {"items": []}

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        retriever = MapyComStructAddressRetriever(api_key="dummy_key")
        struct_address = retriever.get_struct_address(addr)

        assert struct_address is None


def test_mapy_com_struct_address_retriever_api_error():
    """
    Test MapyComStructAddressRetriever with an API error response.
    Testing that the retriever returns None when the API returns a non-200 status code.
    """
    addr = "TEST"

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 500  # Simulate server error

        retriever = MapyComStructAddressRetriever(api_key="dummy_key")
        struct_address = retriever.get_struct_address(addr)

        assert struct_address is None
