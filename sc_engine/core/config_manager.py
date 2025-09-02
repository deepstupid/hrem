import yaml
import os
from typing import Dict, Any
from rich.console import Console
from pydantic import ValidationError
from .schemas import ChallengeConfigSchema, ModelConfigSchema

console = Console()

class ConfigManager:
    """Manages loading and validation of configuration files."""

    def __init__(self, config_dir: str = 'config'):
        self.config_dir = config_dir

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Loads a single YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: Config file not found at {path}[/red]")
            return {}
        except yaml.YAMLError as e:
            console.print(f"[red]Error parsing YAML file at {path}: {e}[/red]")
            return {}

    def load_challenge_configs(self) -> ChallengeConfigSchema:
        """Loads and validates the main challenge configuration file."""
        path = os.path.join(self.config_dir, 'challenge_config.yaml')
        raw_data = self._load_yaml(path)
        if not raw_data:
            raise ValueError("Challenge config is empty or could not be loaded.")

        try:
            return ChallengeConfigSchema.model_validate(raw_data)
        except ValidationError as e:
            console.print(f"[bold red]Error validating challenge config '{path}':[/bold red]")
            console.print(e)
            raise ValueError("Challenge configuration validation failed.") from e

    def load_model_configs(self) -> Dict[str, ModelConfigSchema]:
        """Loads and validates all model configurations from the 'models' directory."""
        models_dir = os.path.join(self.config_dir, 'models')
        model_configs = {}
        if not os.path.isdir(models_dir):
            return model_configs

        for filename in os.listdir(models_dir):
            if filename.endswith((".yaml", ".yml")):
                path = os.path.join(models_dir, filename)
                config_data = self._load_yaml(path)
                if config_data and 'name' in config_data:
                    try:
                        validated_config = ModelConfigSchema.model_validate(config_data)
                        model_configs[validated_config.name] = validated_config
                    except ValidationError as e:
                        console.print(f"[bold red]Error validating model config '{filename}':[/bold red]")
                        console.print(e)
                        # Decide if one bad config should stop everything. For now, we'll skip it.
                        continue
        return model_configs

    def load_search_spaces(self) -> Dict[str, Any]:
        """Loads all search space configurations from the 'search' directory."""
        search_dir = os.path.join(self.config_dir, 'search')
        search_spaces = {}
        if not os.path.isdir(search_dir):
            return search_spaces

        for filename in os.listdir(search_dir):
            if filename.endswith((".yaml", ".yml")):
                path = os.path.join(search_dir, filename)
                config = self._load_yaml(path)
                if config:
                    search_spaces.update(config)
        return search_spaces
