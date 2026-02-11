"""Feature Matrix Builder for combining multiple feature datasets."""

import logging
from pathlib import Path

import pandas as pd
import yaml
from pydantic import ValidationError

from geoscore_de.data_flow.features.base import BaseFeature, get_feature_class
from geoscore_de.data_flow.features.config import FeaturesYAMLConfig

logger = logging.getLogger(__name__)


class FeatureMatrixBuilder:
    """Builds a feature matrix by loading and combining multiple feature datasets.

    This class reads a YAML configuration file that specifies which features to
    include and how to combine them into a single feature matrix.
    """

    def __init__(self, config_path: str = "configs/features.yaml"):
        """Initialize the FeatureMatrixBuilder.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.features: dict[str, BaseFeature] = {}
        self.municipalities: BaseFeature | None = None
        self.load_features()

    def _load_config(self) -> FeaturesYAMLConfig:
        """Load and validate the YAML configuration file.

        Returns:
            Validated FeaturesYAMLConfig object.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the config file is invalid YAML.
            ValidationError: If the config doesn't match the expected schema.
        """
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(config_file, "r") as f:
            try:
                raw_config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML config: {e}")
                raise

        try:
            config = FeaturesYAMLConfig(**raw_config)
            logger.info(f"Validated configuration with {len(config.features)} features")
            return config
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def load_features(self) -> None:
        """Load all features from the configuration."""
        # load municipalities feature first
        self.municipalities = get_feature_class(self.config.municipalities)

        if not self.config.features:
            logger.warning("No features found in configuration")
            return

        for feature_config in self.config.features:
            try:
                feature_instance = get_feature_class(feature_config)
                self.features[feature_config.name] = feature_instance
            except Exception as e:
                logger.error(f"Failed to load feature {feature_config.name}: {e}")
                # Depending on requirements, you might want to raise or continue
                # For now, we'll continue to try loading other features
                continue

        logger.info(f"Loaded {len(self.features)} features")

    def build_matrix(self) -> pd.DataFrame:
        """Build the feature matrix by loading and merging all features.

        Returns:
            DataFrame containing the combined feature matrix.

        Raises:
            ValueError: If no features are loaded or no data can be merged.
            Exception: If any feature fails to load or process.
        """
        if not self.features:
            raise ValueError("No features loaded. Call load_features() first.")

        matrix_config = self.config.matrix
        join_key = matrix_config.join_key

        logger.info(f"Building feature matrix with {len(self.features)} features")

        # Load and transform each feature
        feature_dfs = []
        for feature_name, feature_instance in self.features.items():
            try:
                logger.info(f"Processing feature: {feature_name}")

                # Load and transform data
                df = feature_instance.load_transform()

                # Verify join key exists
                if join_key not in df.columns:
                    msg = f"Join key '{join_key}' not found in {feature_name} dataframe"
                    logger.error(msg)
                    raise KeyError(msg)

                # Add feature name prefix to columns (except join key)
                df = df.rename(columns={col: f"{feature_name}_{col}" for col in df.columns if col != join_key})

                feature_dfs.append(df)
                logger.info(f"Added {feature_name} with shape {df.shape}")

            except Exception as e:
                logger.error(f"Failed to process feature {feature_name}: {e}")
                raise

        if not feature_dfs:
            raise ValueError("No feature data could be loaded")

        # Merge all feature dataframes
        logger.info(f"Merging {len(feature_dfs)} feature dataframes")
        logger.info("Loading municipalities as base dataframe")
        result_df = self.municipalities.load_transform()

        for df in feature_dfs:
            result_df = result_df.merge(df, on=join_key, how="left")
            logger.info(f"Merged dataframe shape: {result_df.shape}")

        # Handle missing values
        if matrix_config.missing_values == "drop":
            logger.info("Dropping rows with missing values")
            result_df = result_df.dropna()
        elif matrix_config.missing_values == "fill":
            logger.info(f"Filling missing values with {matrix_config.fill_value}")
            result_df = result_df.fillna(matrix_config.fill_value)

        logger.info(f"Final feature matrix shape: {result_df.shape}")

        # Save output if configured
        if matrix_config.save_output:
            output_path = Path(matrix_config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)
            logger.info(f"Saved feature matrix to {output_path}")

        return result_df

    def get_feature_names(self) -> list[str]:
        """Get the names of all loaded features.

        Returns:
            List of feature names.
        """
        return list(self.features.keys())

    def get_feature(self, name: str) -> BaseFeature | None:
        """Get a specific feature instance by name.

        Args:
            name: Feature name.

        Returns:
            Feature instance or None if not found.
        """
        return self.features.get(name)
