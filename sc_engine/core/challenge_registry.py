from typing import List, Dict, Optional
from .schemas import ChallengeSchema
from .config_manager import ConfigManager


class ChallengeRegistry:
    """
    Manages the loading and accessing of challenge configurations.
    """
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._challenges = self._load_challenges()

    def _load_challenges(self) -> Dict[str, ChallengeSchema]:
        """
        Loads the challenge configurations using the ConfigManager.
        """
        try:
            challenge_config = self.config_manager.load_challenge_configs()
            return {challenge.id: challenge for challenge in challenge_config.challenges}
        except (ValueError, FileNotFoundError):
            # In case of validation or file error, return an empty dict
            return {}

    def get_all_challenges(self) -> List[ChallengeSchema]:
        """
        Returns a list of all available challenges.
        """
        return list(self._challenges.values())

    def get_challenge_by_id(self, challenge_id: str) -> Optional[ChallengeSchema]:
        """
        Returns a specific challenge by its ID.
        """
        return self._challenges.get(challenge_id)
