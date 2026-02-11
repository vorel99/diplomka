import importlib
import logging
from abc import ABCMeta, abstractmethod

import pandas as pd

from geoscore_de.data_flow.feature_engineering.base import BaseFeatureEngineering, get_feature_engineering_class
from geoscore_de.data_flow.features.config import FeatureConfig, FeatureEngineeringConfig

logger = logging.getLogger(__name__)


class BaseFeature(metaclass=ABCMeta):
    """Abstract base class for data flow features.

    Args:
        before_transforms (list[FeatureEngineeringConfig] | None):
            List of feature engineering transformations to apply to the raw data.
    """

    before_transforms: list[BaseFeatureEngineering]

    def __init__(self, before_transforms: list[FeatureEngineeringConfig] | None = None):
        """Initialize the feature with optional parameters."""
        self.before_transforms = []
        if before_transforms:
            for transform_config in before_transforms:
                logger.info(f"Instantiating before_transform: '{transform_config.name}'")
                transform_class = get_feature_engineering_class(transform_config)
                transform_instance = transform_class(
                    input_columns=transform_config.input_columns,
                    output_column=transform_config.output_column,
                    **transform_config.params,
                )
                self.before_transforms.append(transform_instance)

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load raw data for the feature."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into feature data."""
        pass

    def load_transform(self) -> pd.DataFrame:
        """Load and transform data in one step."""
        raw_data = self.load()
        engineered_features = []

        if self.before_transforms:
            logger.info("Applying before_transforms")
            for transform_instance in self.before_transforms:
                result = transform_instance.apply(raw_data)
                engineered_features.append(result)

        transformed_data = self.transform(raw_data)

        # Join all engineered features to transformed_data on AGS
        if engineered_features:
            for feature_df in engineered_features:
                transformed_data = transformed_data.merge(feature_df, on="AGS", how="left")

        return transformed_data


def instantiate_feature(config: FeatureConfig) -> BaseFeature:
    """Dynamically instantiate a feature class from configuration.

    Args:
        config: FeatureConfig object containing feature configuration.

    Returns:
        An instance of the specified feature class.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class doesn't exist in the module.
    """
    module_name = config.module
    class_name = config.class_name
    params = config.params

    try:
        # Import the module
        module = importlib.import_module(module_name)
        # Get the class from the module
        feature_class = getattr(module, class_name)
        # Instantiate the class with parameters
        feature_instance = feature_class(**params, before_transforms=config.before_transforms)
        logger.info(f"Instantiated {class_name} for feature '{config.name}'")
        return feature_instance
    except ImportError as e:
        logger.error(f"Failed to import module {module_name}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Class {class_name} not found in module {module_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to instantiate {class_name}: {e}")
        raise
