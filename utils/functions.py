import importlib
import inspect


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)


def prepare_data_config(dataset: str, smoke_test: bool, num_aug: int = 0):
    """
    Prepares the data configuration for an experiment.
    """
    from hrm_system.config import DataConfig
    from dataset_manager import dataset_manager
    from rich.console import Console

    console = Console()

    try:
        dataset_path = dataset_manager.get_dataset_path(dataset, smoke_test)
        return DataConfig(dataset=dataset, dataset_path=dataset_path, num_aug=num_aug)
    except Exception as e:
        console.print(f"[red]Error accessing dataset '{dataset}': {str(e)}[/red]")
        if dataset != "synthetic":
            console.print("[yellow]Falling back to synthetic dataset...[/yellow]")
            try:
                dataset_path = dataset_manager.get_dataset_path("synthetic", smoke_test)
                return DataConfig(dataset="synthetic", dataset_path=dataset_path, num_aug=num_aug)
            except Exception as e2:
                console.print(f"[red]Failed to access synthetic dataset: {str(e2)}[/red]")
                raise
        else:
            raise

def prepare_run_config(smoke_test: bool, study_name: str):
    """
    Prepares the run configuration for an experiment.
    """
    from hrm_system.config import RunConfig
    from hrm_system import logger_callback

    return RunConfig(smoke_test=smoke_test, study_name=study_name, logger_callback=logger_callback)
