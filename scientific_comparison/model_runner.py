"""Scientific model runner that integrates with the discovery engine."""

import optuna
from typing import Dict, Any, Tuple, List
from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .engine import ScientificDiscoveryEngine, DiscoveryResults
from .patience_manager import ScientificInsight
from demo_config_manager import ConfigManager
from demo_models import get_model_config, get_model_search_space
from hrm_system.config import HREMParams
from demo_model_runner import run_trial_with_timing
from rich.console import Console

console = Console()

class ScientificModelRunner:
    """Runner for discovery-oriented algorithm comparison."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        
    def run_discovery_oriented_comparison(self, challenge: ChallengeConfig, 
                                       algorithms: List[AlgorithmConfig],
                                       patience_budget: PatienceBudget) -> DiscoveryResults:
        """
        Run algorithm comparison optimized for scientific discovery.
        
        Args:
            challenge: Challenge configuration
            algorithms: List of algorithms to compare
            patience_budget: Patience budget for the comparison
            
        Returns:
            DiscoveryResults with comparison results and insights
        """
        console.print("[bold blue]🔬 Starting Scientific Algorithm Comparison[/bold blue]")
        
        # Initialize the discovery engine
        engine = ScientificDiscoveryEngine(challenge, algorithms)
        
        # Execute the discovery session
        results = engine.execute_discovery_session(patience_budget)
        
        # Display results
        self._display_results(results)
        
        return results
    
    def run_comparison_from_config(self, challenge_id: str, 
                                 patience_level: str = "medium") -> DiscoveryResults:
        """
        Run comparison based on configuration files.
        
        Args:
            challenge_id: ID of the challenge to run
            patience_level: Patience level (low, medium, high)
            
        Returns:
            DiscoveryResults with comparison results and insights
        """
        # Load challenge configuration
        challenge_configs = self.config_manager.load_challenge_config()
        challenge_data = None
        for config in challenge_configs:
            if config.get("id") == challenge_id:
                challenge_data = config
                break
        
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{challenge_id}' not found")
        
        # Convert to ChallengeConfig
        from hrm_system.config import DataConfig
        from .config import ChallengeLevel, PatienceBudget
        
        # Get data config
        dataset_config = challenge_data.get("dataset", {})
        dataset_name = dataset_config.get("dataset", "synthetic")
        data_config = self._get_data_config(dataset_name)
        
        challenge = ChallengeConfig(
            name=challenge_data["name"],
            id=challenge_data["id"],
            description=challenge_data["description"],
            dataset=data_config,
            difficulty=ChallengeLevel[challenge_data.get("difficulty", "BEGINNER")],
            scientific_question=challenge_data.get("scientific_question", "Compare algorithm performance"),
            hypothesis_space=challenge_data.get("hypothesis_space", [])
        )
        
        # Load algorithm configurations
        algorithm_names = challenge_data.get("models", [])
        algorithms = self._load_algorithms(algorithm_names)
        
        # Create patience budget
        patience_budget = PatienceBudget(level=patience_level)
        
        # Run the comparison
        return self.run_discovery_oriented_comparison(challenge, algorithms, patience_budget)
    
    def _get_data_config(self, dataset_name: str) -> Any:
        """Get data configuration for a dataset."""
        from demo_model_runner import get_dataset_config
        try:
            return get_dataset_config(dataset_name, smoke_test=False)
        except:
            # Fallback to synthetic
            return get_dataset_config("synthetic", smoke_test=False)
    
    def _load_algorithms(self, algorithm_names: List[str]) -> List[AlgorithmConfig]:
        """Load algorithm configurations."""
        algorithms = []
        
        for name in algorithm_names:
            # Get model config from registry
            model_config = get_model_config(name)
            if not model_config:
                console.print(f"[yellow]⚠️  Model '{name}' not found in registry[/yellow]")
                continue
            
            # Get search space
            search_space = get_model_search_space(name) or {}
            
            # Create algorithm config
            alg_config = AlgorithmConfig(
                name=model_config.name,
                algorithm_class=model_config.algorithm_class,
                theoretical_advantages=[],  # These would come from config
                theoretical_limitations=[],  # These would come from config
                search_space=search_space
            )
            
            algorithms.append(alg_config)
        
        return algorithms
    
    def _display_results(self, results: DiscoveryResults):
        """Display the results of the discovery session."""
        console.print("\n[bold green]=== SCIENTIFIC DISCOVERY RESULTS ===[/bold green]")
        
        # Display challenge information
        console.print(f"\n[bold]Challenge:[/bold] {results.challenge.name}")
        console.print(f"[bold]Scientific Question:[/bold] {results.challenge.scientific_question}")
        
        # Display algorithm results summary
        console.print(f"\n[bold]Algorithm Performance:[/bold]")
        for alg_name, metrics in results.algorithm_results.items():
            loss = metrics.get('all/lm_loss', 'N/A')
            timing = metrics.get('timing', 'N/A')
            console.print(f"  {alg_name}: Loss={loss:.4f}, Time={timing:.2f}s")
        
        # Display insights
        if results.insights:
            console.print(f"\n[bold]Scientific Insights:[/bold]")
            for i, insight in enumerate(results.insights, 1):
                console.print(f"  {i}. [{insight.type.upper()}] (Confidence: {insight.confidence:.2f})")
                for implication in insight.implications:
                    console.print(f"      - {implication}")
        else:
            console.print(f"\n[yellow]No significant scientific insights generated[/yellow]")
        
        # Display timing summary
        console.print(f"\n[bold]Session Summary:[/bold]")
        console.print(f"  Total Time: {results.metadata.get('total_elapsed_time', 0):.2f}s")
        console.print(f"  Budget Remaining: {results.metadata.get('remaining_budget', 0):.2f}s")