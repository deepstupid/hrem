"""Dataset manager for abstracting dataset handling and providing more synthetic challenges."""

import os
import subprocess
import sys
from typing import Dict, Any, List, Tuple
from pathlib import Path
from rich.console import Console

console = Console()

class DatasetManager:
    """Manages dataset operations including downloading, caching, and loading."""
    
    def __init__(self):
        self.datasets_dir = Path("data")
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Define available datasets
        self.available_datasets = {
            "synthetic": {
                "type": "synthetic",
                "tasks": ["copy", "reverse", "sort", "parity", "duplicate"],
                "default_task": "copy"
            },
            "arc": {
                "type": "external",
                "requires_download": True,
                "download_instructions": "See README for ARC dataset preparation instructions."
            },
            "sudoku": {
                "type": "generated",
                "requires_download": False
            },
            "maze": {
                "type": "generated",
                "requires_download": False
            }
        }
    
    def get_synthetic_dataset_path(self, task_type: str = "copy", smoke_test: bool = False) -> str:
        """Get path for synthetic dataset, generating it if needed."""
        dataset_name = f"synthetic-{task_type}"
        if smoke_test:
            dataset_name += "-smoke"
            
        dataset_path = self.datasets_dir / dataset_name
        if not dataset_path.exists():
            console.print(f"[yellow]Generating synthetic {task_type} dataset...[/yellow]")
            self._generate_synthetic_dataset(task_type, dataset_path, smoke_test)
            
        return str(dataset_path)
    
    def _generate_synthetic_dataset(self, task_type: str, dataset_path: Path, smoke_test: bool = False):
        """Generate a synthetic dataset."""
        try:
            # Import the synthetic dataset builder
            script_path = Path("dataset/build_synthetic_dataset.py")
            
            # Prepare arguments
            args = [
                sys.executable,
                str(script_path),
                "--output-dir", str(dataset_path),
                "--task-type", task_type
            ]
            
            if smoke_test:
                args.extend(["--num-samples", "10", "--seq-len", "5"])
            else:
                args.extend(["--num-samples", "1000", "--seq-len", "10"])
                
            # Run the dataset generation script
            result = subprocess.run(args, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"[red]Error generating synthetic dataset: {result.stderr}[/red]")
                # Fallback to copy task if the requested task fails
                if task_type != "copy":
                    console.print("[yellow]Falling back to copy task...[/yellow]")
                    self._generate_synthetic_dataset("copy", dataset_path, smoke_test)
                else:
                    raise Exception(f"Failed to generate synthetic dataset: {result.stderr}")
            else:
                console.print(f"[green]Successfully generated synthetic {task_type} dataset at {dataset_path}[/green]")
                
        except Exception as e:
            console.print(f"[red]Error generating synthetic dataset: {str(e)}[/red]")
            raise
    
    def get_dataset_path(self, dataset_name: str, smoke_test: bool = False) -> str:
        """Get path for a dataset, downloading/generating it if needed."""
        try:
            # Handle synthetic datasets with specific tasks
            if dataset_name.startswith("synthetic-"):
                task_type = dataset_name.split("-", 1)[1]
                return self.get_synthetic_dataset_path(task_type, smoke_test)
            
            if dataset_name not in self.available_datasets:
                # Try to treat it as a synthetic task
                if dataset_name in self.available_datasets["synthetic"]["tasks"]:
                    return self.get_synthetic_dataset_path(dataset_name, smoke_test)
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            dataset_info = self.available_datasets[dataset_name]
            
            # Handle synthetic datasets
            if dataset_info["type"] == "synthetic":
                task_type = dataset_info.get("default_task", "copy")
                return self.get_synthetic_dataset_path(task_type, smoke_test)
            
            # Handle generated datasets (sudoku, maze)
            if dataset_info["type"] == "generated":
                dataset_path = self.datasets_dir / (dataset_name + ("-smoke" if smoke_test else ""))
                if not dataset_path.exists():
                    console.print(f"[yellow]Generating {dataset_name} dataset...[/yellow]")
                    self._generate_dataset(dataset_name, dataset_path, smoke_test)
                return str(dataset_path)
            
            # Handle external datasets (ARC)
            if dataset_info["type"] == "external":
                dataset_path = self.datasets_dir / (dataset_name + ("-smoke" if smoke_test else ""))
                if not dataset_path.exists():
                    if dataset_info.get("requires_download", False):
                        console.print(f"[red]{dataset_info['download_instructions']}[/red]")
                        # Try to generate a synthetic fallback
                        console.print("[yellow]Falling back to synthetic dataset...[/yellow]")
                        return self.get_synthetic_dataset_path("copy", smoke_test)
                return str(dataset_path)
            
            raise ValueError(f"Unsupported dataset type: {dataset_info['type']}")
        except Exception as e:
            # Final fallback to synthetic copy dataset
            console.print(f"[red]Error accessing dataset {dataset_name}: {str(e)}[/red]")
            console.print("[yellow]Falling back to synthetic copy dataset...[/yellow]")
            return self.get_synthetic_dataset_path("copy", smoke_test)
    
    def _generate_dataset(self, dataset_name: str, dataset_path: Path, smoke_test: bool = False):
        """Generate a dataset using the appropriate builder."""
        try:
            script_name = f"build_{dataset_name}_dataset.py"
            script_path = Path("dataset") / script_name
            
            if not script_path.exists():
                raise FileNotFoundError(f"Dataset builder script not found: {script_path}")
            
            # Prepare arguments
            args = [
                sys.executable,
                str(script_path),
                "--output-dir", str(dataset_path)
            ]
            
            if smoke_test:
                args.append("--num-aug=0")
                
            # Run the dataset generation script
            result = subprocess.run(args, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"[red]Error generating {dataset_name} dataset: {result.stderr}[/red]")
                raise Exception(f"Failed to generate {dataset_name} dataset: {result.stderr}")
            else:
                console.print(f"[green]Successfully generated {dataset_name} dataset at {dataset_path}[/green]")
                
        except Exception as e:
            console.print(f"[red]Error generating {dataset_name} dataset: {str(e)}[/red]")
            raise
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets."""
        return list(self.available_datasets.keys()) + [
            f"synthetic-{task}" for task in self.available_datasets["synthetic"]["tasks"]
        ]
    
    def is_dataset_available(self, dataset_name: str) -> bool:
        """Check if a dataset is available."""
        try:
            self.get_dataset_path(dataset_name)
            return True
        except:
            return False

# Global instance
dataset_manager = DatasetManager()