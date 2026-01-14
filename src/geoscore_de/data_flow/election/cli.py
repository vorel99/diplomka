"""CLI entry point for election data processing."""

import typer

from geoscore_de.data_flow.election.federal_21 import load_raw_election_21_data
from geoscore_de.data_flow.election.federal_25 import load_election_25_data

election_cli_app = typer.Typer()


@election_cli_app.command()
def load_federal_25():
    """Load and extract election 25 data from a ZIP file."""
    load_election_25_data()


@election_cli_app.command()
def load_federal_21():
    """Load and extract election 21 data from a ZIP file."""
    load_raw_election_21_data()
