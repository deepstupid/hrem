"""Centralized parameter definitions for the HRM/HREM demo system."""

from typing import Dict, List

# Display parameters for the challenge selector
CHALLENGE_SELECTOR_PARAMS = {
    "difficulty_display": {
        "BEGINNER": "🌱 Beginner",
        "INTERMEDIATE": "🌿 Intermediate",
        "ADVANCED": "🔥 Advanced",
        "RESEARCH": "🔬 Research"
    },
    "hardware_icon": "🖥️",
    "duration_icon": "⏱️",
    "title": "[bold blue]🎯 HRM vs HREM Challenge Selection[/bold blue]",
    "prompt": "\n[bold green]Select a challenge[/bold green] (enter number or name)",
    "invalid_selection": "[red]Invalid selection. Please try again.[/red]",
    "exit_message": "\n[yellow]Exiting...[/yellow]"
}

# Display parameters for experiment runner
EXPERIMENT_RUNNER_PARAMS = {
    "optimization_spinner": "Running {model_name} hyperparameter optimization...",
    "optimization_completion": "⏱️  {model_name} optimization completed in {elapsed_time:.1f} seconds",
    "evaluation_spinner": "Running final evaluation...",
    "evaluation_completion": "⏱️  Final evaluation completed in {elapsed_time:.1f} seconds",
    "dataset_not_found_title": "[bold red]❌ Dataset not found![/bold red]",
    "dataset_not_found_message": "[yellow]The {dataset} dataset requires raw data files that are not included in this repository.[/yellow]",
    "dataset_not_found_instructions": "[dim]Please download the required dataset files or try a different challenge.[/dim]",
    "dataset_not_found_arc_instructions": "[dim]For ARC challenges, see the README for dataset preparation instructions.[/dim]",
    "no_models_to_evaluate": "[bold yellow]No models to evaluate in final comparison.[/bold yellow]"
}

# Display parameters for the main demo
DEMO_DISPLAY_PARAMS = {
    "demo_title": "🚀 HRM vs HREM Demonstration: {challenge_name}",
    "demo_subtitle": "Complete end-to-end system showcasing real-time results generation",
    "challenge_details_format": "💻 {hardware} | ⏱️  {duration}",
    "models_comparison_format": "[bold]Models to be compared:[/bold] [cyan]{models}[/cyan]",
    "start_demo_prompt": "\n[bold green]Press Enter to start the demonstration...[/bold green]",
    "intro_title": "\n[bold]Challenge:[/bold] {challenge_name}",
    "intro_challenge_label": "[dim]{description}[/dim]",
    "intro_showcase_title": "\n[bold]This demonstration showcases:[/bold]",
    "intro_showcase_items": [
        "• 🔍 Hyperparameter optimization with real-time feedback",
        "• 🏆 Best results and their parameters"
    ],
    "approach_title": "\n[bold blue]🧠 Approach Explanation[/bold blue]",
    "approach_explanation": "The HRM System uses a novel approach to hyperparameter optimization:",
    "approach_bullets": [
        "• Uses guided search to explore hyperparameter space efficiently",
        "• Generates actionable results after each iteration",
        "• Continuously improves based on real-time feedback"
    ],
    "demo_completed_title": "✅ Demonstration Completed Successfully!",
    "final_comparison_title": "📊 Best Performance Comparison",
    "key_insights_title": "\n[bold]🔑 Key Insights:[/bold]",
    "key_insights_items": [
        "• HRM: Traditional recurrent model with external memory",
        "• HREM: Hierarchical approach with multiple memory layers",
        "• Multi-objective optimization balances accuracy and parameter efficiency",
        "• All models are optimized for fair parameter count comparison",
        "• Each iteration provides actionable insights for improvement"
    ],
    "unique_features_title": "\n[bold green]🎯 What Makes This Approach Unique:[/bold green]",
    "unique_features_items": [
        "• Real-time results generation after each iteration",
        "• Guided search for efficient hyperparameter exploration",
        "• Continuous feedback for actionable insights",
        "• No configuration parameters needed - fully turnkey"
    ],
    "demo_failed_message": "[bold red]❌ Demo failed: {error}[/bold red]",
    "challenge_not_found": "[bold red]❌ Challenge '{challenge_key}' not found![/bold red]",
    "available_challenges": "[dim]Available challenges:[/dim]",
    "challenge_list_item": "[dim]  • {challenge_name}[/dim]",
    "no_valid_models": "[bold red]❌ No valid models specified![/bold red]",
    "invalid_model_configs": "[bold red]❌ No valid models specified or invalid model configurations![/bold red]",
    "difficulty_label": "📈 Difficulty: {difficulty}",
    "optimization_header": "🔍 Step 2: Guided Hyperparameter Optimization",
    "optimization_description": "Optimizing hyperparameters for all models on {dataset} dataset with real-time performance feedback...",
    "optimization_advantage": "[dim]💡 Key advantage: Results are generated after each iteration![/dim]",
    "evaluation_header": "🏆 Step 3: Final Performance Comparison",
    "evaluation_description": "Running final comparison with optimized parameters on {dataset} dataset...",
    "best_results_title": "🏆 Best Results",
    "press_enter_evaluation": "\n[bold blue]Press Enter to see final evaluation...[/bold blue]",
    "iteration_header_format": "[bold blue]{title}[/bold blue]",
    "final_comparison_header": "[bold]{title}[/bold]",
    "continuous_improvement_1": "\n[italic]The system begins generating results immediately,[/italic]",
    "continuous_improvement_2": "[italic]allowing for continuous improvement and insights.[/italic]"
}

# Display colors for model comparisons
DISPLAY_COLORS = ["blue", "green", "yellow", "magenta", "cyan", "red"]

# Metrics information for display
METRICS_INFO = [
    ('all/accuracy', 'Accuracy', '🎯 Higher is better - task solving accuracy'),
    ('all/lm_loss', 'Loss', '📉 Lower is better - language modeling loss'),
    ('all/steps', 'Steps', '⚡ Lower is better - average steps to solve'),
    ('step', 'Training Steps', '📊 Training iterations completed'),
    ('num_params', 'Parameters', '⚙️ Model parameter count')
]

# Key metrics for comparison
KEY_METRICS = [
    ('all/accuracy', 'Accuracy'),
    ('all/lm_loss', 'Loss'),
    ('all/steps', 'Steps'),
    ('num_params', 'Parameters')
]

# Challenge hardware and duration information
CHALLENGE_INFO = {
    "computational_requirements": {
        "minimal": "Minimal - runs on any hardware",
        "moderate": "Moderate - requires GPU for reasonable training time",
        "high": "High - requires significant GPU memory and compute",
        "very_high": "Very High - requires powerful multi-GPU setup"
    },
    "expected_duration": {
        "seconds": "Seconds",
        "minutes": "Minutes",
        "hours": "Hours",
        "days": "Days"
    },
    "recommended_hardware": {
        "any": "Any CPU or GPU",
        "gpu_4gb": "GPU with 4GB+ VRAM",
        "gpu_6gb": "GPU with 6GB+ VRAM",
        "gpu_8gb": "GPU with 8GB+ VRAM",
        "multi_gpu_24gb": "Multi-GPU setup with 24GB+ VRAM total",
        "multi_gpu_48gb": "Multi-GPU setup with 48GB+ VRAM total"
    }
}

# Default storage paths
STORAGE_PATHS = {
    "default_optimization_db": "sqlite:///experiments/optuna_demo_cli.db"
}

# Default search space paths
SEARCH_SPACE_PATHS = {
    "hrm": "config/hrm_search_space.yaml",
    "hrem": "config/hparam_search_space.yaml"
}

# Default model names
DEFAULT_MODEL_NAMES = ["HRM", "HREM"]

# Progress display settings
PROGRESS_SETTINGS = {
    "transient": True
}