from typing import List, Dict, Any
from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry
from sc_engine.core.schemas import ChallengeSchema, ModelConfigSchema

def get_available_challenges(challenge_registry: ChallengeRegistry) -> List[ChallengeSchema]:
    """
    Retrieves a list of all available scientific challenges.

    Args:
        challenge_registry: An instance of ChallengeRegistry.

    Returns:
        A list of ChallengeSchema objects.
    """
    return challenge_registry.get_all_challenges()

def get_available_models(config_manager: ConfigManager) -> Dict[str, ModelConfigSchema]:
    """
    Retrieves a dictionary of all available models.

    Args:
        config_manager: An instance of ConfigManager.

    Returns:
        A dictionary mapping model names to ModelConfigSchema objects.
    """
    return config_manager.load_model_configs()
