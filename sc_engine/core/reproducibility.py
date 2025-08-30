import json
import git
from datetime import datetime
from typing import Dict, Any

class ReproducibilityManager:
    """
    Manages the reproducibility of experiments.
    """
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_data: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "git_commit": self.get_git_commit(),
            "parameters": {},
            "results": {}
        }

    def get_git_commit(self) -> str:
        """
        Gets the current git commit hash.
        """
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha
        except git.InvalidGitRepositoryError:
            return "Not a git repository"

    def log_parameters(self, params: Dict[str, Any]):
        """
        Logs the parameters of an experiment.
        """
        self.experiment_data["parameters"].update(params)

    def log_results(self, results: Dict[str, Any]):
        """
        Logs the results of an experiment.
        """
        self.experiment_data["results"].update(results)

    def save_experiment(self, output_dir: str):
        """
        Saves the experiment data to a JSON file.
        """
        output_path = f"{output_dir}/{self.experiment_name}_reproducibility.json"
        with open(output_path, "w") as f:
            json.dump(self.experiment_data, f, indent=4)
