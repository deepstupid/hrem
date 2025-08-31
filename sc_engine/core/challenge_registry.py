import yaml
from typing import List, Dict, Any, Optional

class ChallengeRegistry:
    """
    Manages the loading and accessing of challenge configurations.
    """
    def __init__(self, config_path: str = 'config/challenge_config.yaml'):
        self.config_path = config_path
        self._challenges = self._load_challenges()

    def _load_challenges(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads the challenge configurations from the YAML file.
        """
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            challenges = {challenge['id']: challenge for challenge in config_data.get('challenges', [])}
            return challenges
        except FileNotFoundError:
            return {}

    def get_all_challenges(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all available challenges.
        """
        return list(self._challenges.values())

    def get_challenge_by_id(self, challenge_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns a specific challenge by its ID.
        """
        return self._challenges.get(challenge_id)
