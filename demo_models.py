"""Model management for the HRM/HREM demo system."""

from typing import Dict, List, Any
from hrm_system.config import ModelConfig

class ModelRegistry:
    """Registry for available models in the demo system."""
    
    # Default model configurations
    DEFAULT_MODELS = {
        "HRM": ModelConfig(
            name="HRM",
            algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm",
            base_arch_config="hrm_v1"
        ),
        "HREM": ModelConfig(
            name="HREM",
            algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm",
            base_arch_config="hrem_v1"
        ),
        "EnhancedHREM": ModelConfig(
            name="EnhancedHREM",
            algorithm_class="hrm_system.algorithms.enhanced_hrem.EnhancedHREMAlgorithm",
            base_arch_config="enhanced_hrem_v1"
        )
    }
    
    def __init__(self):
        """Initialize the model registry with default models."""
        self._models = self.DEFAULT_MODELS.copy()
    
    def register_model(self, name: str, model_config: ModelConfig):
        """Register a new model configuration."""
        self._models[name] = model_config
    
    def get_model(self, name: str) -> ModelConfig:
        """Get a model configuration by name."""
        return self._models.get(name)
    
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

def register_custom_model(name: str, model_config: ModelConfig):
    """Register a custom model in the global registry."""
    model_registry.register_model(name, model_config)

def get_model_config(name: str) -> ModelConfig:
    """Get a model configuration from the global registry."""
    return model_registry.get_model(name)

def get_model_configs(names: List[str]) -> List[ModelConfig]:
    """Get multiple model configurations from the global registry."""
    return model_registry.get_models(names)

def list_available_models() -> List[str]:
    """List all available models in the global registry."""
    return model_registry.list_models()

def get_all_model_configs() -> Dict[str, ModelConfig]:
    """Get all model configurations from the global registry."""
    return model_registry.list_model_configs()