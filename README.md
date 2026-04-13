# Geoscore for germany

This repository contains code for the diploma thesis "GeoScore Framework for Predicting Municipal Socio-economic Targets in Germany". The thesis focuses on building a generalizable framework that ingests open geospatial and statistical data for German municipalities and predicts an arbitrary target variable defined at the municipal level.

## Dev Installation

To set up the development environment, run the following command in the terminal:

```bash
make install_dev
```

## CLI Usage

After installation the `geoscore` command is available.

### Build the feature matrix

```bash
geoscore create-feature-matrix --config configs/features.yaml
```

### Train a model

```bash
geoscore train configs/training.yaml data/final/feature_matrix.csv
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--report` / `-r` | `reports/training_report.qmd` | Path to the Quarto report template to render and log after training. |

The `train` command:
1. Loads `TrainingConfig` from the supplied YAML file.
2. Reads the feature matrix CSV.
3. Starts an MLflow run and logs `config_path` and `input_path` as parameters.
4. Runs `Trainer.train()` (grid-search, evaluation, artifact logging).
5. Renders the Quarto report and logs both the `.qmd` source and the rendered HTML as MLflow artifacts under the `report/` artifact path.
