import importlib
import inspect

def load_model_class(identifier: str):
    """Dynamically load a class from a string path."""
    try:
        module_path, class_name = identifier.split('@')
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class '{identifier}'") from e

import collections.abc

def deep_merge(d, u):
    """
    Deep merge two dictionaries.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_merge(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_model_source_path(identifier: str, prefix: str = "models."):
    """This function is not used and will be removed."""
    pass
