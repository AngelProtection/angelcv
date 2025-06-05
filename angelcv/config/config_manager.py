from pathlib import Path
import sys
from typing import Any, Optional, Self, get_type_hints

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
import yaml

from angelcv.config.config_registry import Config
from angelcv.utils.logging_manager import get_logger
from angelcv.utils.path_utils import resolve_file_path

# Configure logging
logger = get_logger(__name__)


# NOTE: this is a singleton class for loading and managing configuration
# it's implemented like this to avoid having to pass the config object around
# and to ensure that the config object is always up to date
class ConfigManager:
    """Singleton class for loading and managing configuration."""

    _instance: Optional["ConfigManager"] = None  # required Optional keyword (for type annotation)
    _config: Config | None = None
    _overrides: DictConfig = OmegaConf.create({})

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = None
        return cls._instance

    @classmethod
    def set_overrides(cls, overrides: list[str] | None, use_cli_args: bool = True) -> None:
        """Add configuration overrides, preserving existing ones with newer values taking precedence."""
        # Create a new list to hold current overrides
        current_overrides = []

        # Add new overrides if provided
        if overrides is not None:
            current_overrides.extend(overrides)

        # Add CLI arguments if requested
        if use_cli_args:
            current_overrides.extend(sys.argv[1:])

        # Convert to OmegaConf and merge with existing overrides (newer values take precedence)
        if current_overrides:
            new_overrides = OmegaConf.from_dotlist(current_overrides)
            cls._overrides = OmegaConf.merge(cls._overrides, new_overrides)

    @classmethod
    def upsert_config(
        cls,
        general_file: str = "general.yaml",
        model_file: str | None = None,
        dataset_file: str | None = None,
        config_dir: str = "angelcv/config",
        overrides: list[str] | None = None,
        use_cli_args: bool = True,
    ) -> Config:
        """
        Create or update a configuration by loading from multiple YAML files and merging with CLI arguments and kwargs.

        Args:
            general_file: Name of the general config file (without .yaml extension)
            model_file: Name of the model config file (without .yaml extension but including the model/ prefix)
            dataset_file: Name of the dataset config file (without .yaml extension but including the dataset/ prefix)
            config_dir: Directory containing the config files
            overrides: Additional configuration overrides
            use_cli_args: Whether to use CLI arguments for overrides

        Returns:
            The configuration instance
        """

        if overrides or use_cli_args:
            cls.set_overrides(overrides, use_cli_args)

        # Load the general config
        general_file_path = Path(config_dir) / general_file
        if not general_file_path.exists():
            raise FileNotFoundError(f"General config file not found: {general_file_path}")

        # Only load the general.yaml when the config is created (not updated)
        if cls._config is None:
            with open(general_file_path, "r") as f:  # noqa: UP015
                general_config_dict = yaml.safe_load(f)
            config_oc = OmegaConf.create(general_config_dict)
        else:
            config_oc = OmegaConf.create(cls._config.model_dump())

        # If a file is provided, set it (relative path, relative to the config directory, or absolute)
        if model_file:
            model_file_path = resolve_file_path(model_file, download_from_s3=False)
            with open(model_file_path, "r") as f:  # noqa: UP015
                model_config_dict = yaml.safe_load(f)
            # Overwrite the model config with the new model config
            model_config = OmegaConf.create(model_config_dict)
            config_oc.model = model_config

        if dataset_file:
            dataset_file_path = resolve_file_path(dataset_file, download_from_s3=False)
            with open(dataset_file_path, "r") as f:  # noqa: UP015
                dataset_config_dict = yaml.safe_load(f)
            # Overwrite the dataset config with the new dataset config
            # NOTE: this is overwitten instead of merged, because if we merge it, then the
            # resulting config.dataset.names can keep the old names, which is not what we want
            dataset_config = OmegaConf.create(dataset_config_dict)
            config_oc.dataset = dataset_config

        # Apply overrides
        if cls._overrides:
            config_oc = OmegaConf.merge(config_oc, cls._overrides)
        base_config_dict = OmegaConf.to_container(config_oc, resolve=True)

        # Initialize the config instance once, then update it
        # NOTE: without this, the config object is not updated when the config is updated
        if cls._config is None:
            # First initialization
            cls._config = Config(**base_config_dict)
        else:
            # Update existing config by transferring all attributes
            updated_config = Config(**base_config_dict)
            # Update the class-level config with new values
            for key, value in updated_config.__dict__.items():
                setattr(cls._config, key, value)

        logger.info("Configuration updated successfully!")
        return cls._config

    @classmethod
    def wildcard_config_merge(cls, kwargs: dict[str, Any]) -> Config:
        """
        Recursively merges wildcard configuration values by matching leaf node names with keys in kwargs.

        This function traverses the entire config structure and updates any field whose name
        matches a key in the provided kwargs dictionary, regardless of nesting level.

        Args:
            kwargs: Dictionary of key-value pairs to apply to matching fields

        Returns:
            The updated configuration object
        """

        # Helper function to recursively update configuration
        def _update_config_recursive(obj: BaseModel, updates: dict[str, Any]) -> None:
            for field_name, field_value in obj.__dict__.items():
                # Check if the current field name matches any key in updates
                if field_name in updates:
                    # Get type hints to ensure type safety when updating
                    type_hints = get_type_hints(obj.__class__)
                    expected_type = type_hints.get(field_name)

                    # Update the field with the new value, respecting type hints if available
                    new_value = updates[field_name]
                    if expected_type and not isinstance(new_value, expected_type):
                        try:
                            # Try to convert the value to the expected type
                            new_value = expected_type(new_value)
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Cannot convert {new_value} to {expected_type} for field {field_name}."
                                f"Skipping this update."
                            )
                            continue

                    setattr(obj, field_name, new_value)
                    logger.debug(f"Updated config field '{field_name}' with value: {new_value}")

                # If this is a nested BaseModel, recurse into it
                if isinstance(field_value, BaseModel):
                    _update_config_recursive(field_value, updates)

                # If this is a list that might contain BaseModel instances, check each item
                elif isinstance(field_value, list):
                    for item in field_value:
                        if isinstance(item, BaseModel):
                            _update_config_recursive(item, updates)

                # If this is a dict that might contain BaseModel instances, check each value
                elif isinstance(field_value, dict):
                    for _key, value in field_value.items():
                        if isinstance(value, BaseModel):
                            _update_config_recursive(value, updates)

        # Apply updates recursively
        _update_config_recursive(cls._config, kwargs)

        return cls._config

    @classmethod
    def get_config(cls) -> Config:
        """
        Get the configuration instance.
        """
        return cls._config

    @classmethod
    def set_config(cls, config: Config) -> Config:
        """
        Set the configuration directly from a Config object.

        Args:
            config: A Config object to use as the configuration

        Returns:
            The set configuration instance
        """
        if not isinstance(config, Config):
            raise TypeError(f"Expected Config object, got {type(config)}")

        # Convert Config to dict
        config_dict = config.model_dump()

        if cls._config is None:
            cls._config = Config(**config_dict)
        else:
            # Update existing config by transferring all attributes
            for key, value in config.__dict__.items():
                setattr(cls._config, key, value)

        logger.info("Configuration set successfully!")
        return cls._config

    @classmethod
    def reset_config(cls) -> None:
        """Reset all configuration related state."""
        cls._config = None
        cls._overrides = OmegaConf.create({})
        logger.info("Configuration reset successfully!")


if __name__ == "__main__":
    from pprint import pprint

    config_1 = ConfigManager.upsert_config(
        model_file="yolov10s.yaml",
        # dataset_file="coco.yaml",
        # overrides=["model.channels_scale=1.0"],
    )
    # pprint(config_1)

    # Get the same config object again but update it with new settings
    config_2 = ConfigManager.upsert_config(
        model_file="yolov10n.yaml",
        dataset_file="coco.yaml",
    )
    # pprint(config_2)

    print("config_1.model.channels_scale", config_1.model.channels_scale)
    print("config_2.model.channels_scale", config_2.model.channels_scale)

    # Example of using wildcard config merge
    config_3 = ConfigManager.wildcard_config_merge(
        {
            "channels_scale": 0.75,  # updates model.channels_scale
            "batch_size": 32,  # updates train.data.batch_size and validation.data.batch_size
            "image_size": 512,  # updates train.data.image_size, validation.data.image_size, and root image_size
        },
    )
    print("config_1.model.channels_scale:", config_1.model.channels_scale)
    print("config_2.model.channels_scale:", config_2.model.channels_scale)
    print("config_3.model.channels_scale:", config_3.model.channels_scale)
    print("config_3.train.data.batch_size:", config_3.train.data.batch_size)
    print("config_3.image_size:", config_3.image_size)
    print("config_3.train.data.image_size:", config_3.train.data.image_size)
    print("config_3.validation.data.image_size:", config_3.validation.data.image_size)
