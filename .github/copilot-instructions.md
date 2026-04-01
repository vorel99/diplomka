# Project Guidelines

## Code Style
- Use Python 3.10+ with 4-space indentation.
- Keep line length at 120 characters (Ruff config in `pyproject.toml`).
- Prefer explicit names, type hints on public APIs, and concise docstrings for non-obvious logic.

## Architecture
- Production code lives in `src/geoscore_de/`; tests mirror this under `tests/`.
- Domain boundaries:
  - `src/geoscore_de/app/`: FastAPI app, routes, middleware, settings.
  - `src/geoscore_de/address/`: geocoding/address retrieval logic.
  - `src/geoscore_de/data_flow/`: feature ingestion and matrix building.
  - `src/geoscore_de/modelling/`: filtering, training, evaluation, plots.
- Keep workflows config-driven through Pydantic models and YAML/env files rather than ad-hoc dicts.

## Build And Test
- Setup dev environment: `make install_dev`
- Lint: `ruff check`
- Format: `ruff format`
- Run tests: `pytest`
- Run API locally: `python -m geoscore_de.app`

## Conventions
- Validate app/config objects with Pydantic (`BaseSettings`/`BaseModel`).
- Feature and training setup must come from:
  - `configs/features.yaml`
  - `configs/training.yaml`
  - `configs/.env` (template: `configs/example.env`)
- Prefer vectorized pandas/geopandas transforms over row-wise loops.
- Add/update pytest coverage for behavior changes, especially in config parsing, filtering, and API routes.
- Keep reusable logic in `.py` modules; use notebooks in `analyses/` for exploration only.

## Pitfalls
- `MAPY_COM_API_KEY` is required for geocoding-related paths.
- Default virtual environment is `.venv`; keep commands compatible with it.
- Avoid editing generated/artifact-heavy outputs unless explicitly requested:
  - `mlruns/`
  - `reports/`
  - `data/final/`
  - notebook outputs
- Do not hardcode secrets or add new dependencies when existing project libraries already cover the use case.

## Reference Files
- `src/geoscore_de/app/config.py`
- `src/geoscore_de/app/main.py`
- `src/geoscore_de/data_flow/features/base.py`
- `src/geoscore_de/data_flow/features/matrix_builder.py`
- `src/geoscore_de/modelling/config.py`
- `src/geoscore_de/modelling/train.py`
