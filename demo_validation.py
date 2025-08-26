"""Configuration validation for the HRM/HREM demo system."""

from typing import List, Dict, Any
from hrm_system.config import ModelConfig

class ConfigValidator:
    """Validator for demo configurations."""
    
    @staticmethod
    def validate_model_names(model_names: List[str], available_models: List[str]) -> List[str]:
        """Validate and filter model names, returning only valid ones."""
        valid_models = []
        invalid_models = []
        
        for name in model_names:
            if name in available_models:
                valid_models.append(name)
            else:
                invalid_models.append(name)
        
        if invalid_models:
            from rich.console import Console
            console = Console()
            console.print(f"[bold yellow]Warning: Unknown models {invalid_models}, available models: {available_models}[/bold yellow]")
        
        return valid_models
    
    @staticmethod
    def validate_model_config(model_config: ModelConfig) -> bool:
        """Validate a single model configuration."""
        if not model_config.name:
            return False
        if not model_config.algorithm_class:
            return False
        if not model_config.base_arch_config:
            return False
        return True
    
    @staticmethod
    def validate_model_configs(model_configs: List[ModelConfig]) -> bool:
        """Validate a list of model configurations."""
        if not model_configs:
            return False
        
        for config in model_configs:
            if not ConfigValidator.validate_model_config(config):
                return False
        
        # Check for duplicate names
        names = [config.name for config in model_configs]
        if len(names) != len(set(names)):
            return False
            
        return True
    
    @staticmethod
    def validate_challenge_key(challenge_key: str, available_challenges: List[str]) -> bool:
        """Validate a challenge key."""
        return challenge_key in available_challenges or any(
            challenge_key.lower() in challenge.lower() or 
            challenge.lower() in challenge_key.lower() 
            for challenge in available_challenges
        )