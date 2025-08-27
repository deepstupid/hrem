"""Configuration management for the HRM/HREM demo system."""

from demo_config_manager import ConfigManager, PatienceLevel

# Export the classes and functions from the unified config manager
load_ui_config = ConfigManager.load_ui_config
load_challenge_config = ConfigManager.load_challenge_config
get_configs_for_challenge = ConfigManager.get_configs_for_challenge