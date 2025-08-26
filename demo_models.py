import os
import yaml
from typing import Dict, List, Any, Optional
from hrm_system.config import ModelConfig

class ModelRegistry:
    """Registry for available models, loaded from YAML configuration."""

    def __init__(self, models_dir="config/models", search_dir="config/search"):
        """Initialize the model registry by loading configurations from YAML files."""
        self._models: Dict[str, ModelConfig] = {}
        self._search_spaces: Dict[str, Dict[str, Any]] = {}

        if not os.path.isdir(models_dir):
            # In a testing environment, the path might be different.
            # For now, we'll just skip loading if the directory doesn't exist.
            return

        for model_file in os.listdir(models_dir):
            if model_file.endswith(".yaml"):
                model_path = os.path.join(models_dir, model_file)
                with open(model_path, 'r') as f:
                    model_data = yaml.safe_load(f)

                model_name = model_data["name"]

                # Create ModelConfig
                model_config = ModelConfig(
                    name=model_name,
                    algorithm_class=model_data["algorithm_class"],
                    base_arch_config=model_data["base_arch_config"]
                )
                self._models[model_name] = model_config

                # Load search space if specified
                if "search_space_config" in model_data:
                    search_space_file = f"{model_data['search_space_config']}.yaml"
                    search_path = os.path.join(search_dir, search_space_file)
                    if os.path.exists(search_path):
                        with open(search_path, 'r') as sf:
                            self._search_spaces[model_name] = yaml.safe_load(sf)

    def register_model(self, name: str, model_config: ModelConfig, search_space: Optional[Dict[str, Any]] = None):
        """Register a new model configuration."""
        self._models[name] = model_config
        if search_space:
            self._search_spaces[name] = search_space

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get a model configuration by name."""
        return self._models.get(name)

    def get_search_space(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a search space by model name."""
        return self._search_spaces.get(name)

    def get_models(self, names: List[str]) -> List[ModelConfig]:
        """Get multiple model configurations by names."""
        return [self._models[name] for name in names if name in self._models]

    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self._models.keys())

    def list_model_configs(self) -> Dict[str, ModelConfig]:
        """List all available model configurations."""
        return self._models.copy()

# Global model registry instance
model_registry = ModelRegistry()

def get_model_config(name: str) -> Optional[ModelConfig]:
    """Get a model configuration from the global registry."""
    return model_registry.get_model(name)

def get_model_search_space(name: str) -> Optional[Dict[str, Any]]:
    """Get a model's search space from the global registry."""
    return model_registry.get_search_space(name)

def get_model_configs(names: List[str]) -> List[ModelConfig]:
    """Get multiple model configurations from the global registry."""
    return model_registry.get_models(names)

def list_available_models() -> List[str]:
    """List all available models in the global registry."""
    return model_registry.list_models()

def get_all_model_configs() -> Dict[str, ModelConfig]:
    """Get all model configurations from the global registry."""
    return model_registry.list_model_configs()