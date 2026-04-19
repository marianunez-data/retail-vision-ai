"""
Simple configuration loader.

Why a config file instead of hardcoding values?
    1. Change a threshold WITHOUT editing Python code
    2. ONE source of truth for all hyperparameters
    3. Easy to track changes in git (diff the YAML)
    4. Reproducibility: the YAML documents exactly what settings
       were used for each experiment

"""

from pathlib import Path

import yaml


def load_config(config_path=None):
    """Load project config from YAML file and return as dictionary.

    The function looks for configs/base_config.yaml relative to the
    project root (3 levels up from this file: utils/ -> src/ -> root/).

    Args:
        config_path: Optional custom path. If None, uses the default
            location at configs/base_config.yaml.

    Returns:
        A nested dictionary matching the YAML structure.
        Access values like: config["training"]["epochs"] -> 50

    Raises:
        FileNotFoundError: If the YAML file doesn't exist at the
            expected path. Common cause: running from wrong directory.

    Usage:
        from src.utils.config import load_config

        config = load_config()
        print(config["training"]["epochs"])       # 50
        print(config["dataset"]["class_names"])   # ["product", "gap"]
        print(config["inference"]["confidence_threshold"])  # 0.45
    """
    if config_path is None:
        # Navigate from src/utils/config.py -> src/utils -> src -> project root
        # Path(__file__)         = src/utils/config.py
        # .parent               = src/utils/
        # .parent               = src/
        # .parent               = project root (retailvision-ai/)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "base_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            f"Make sure you run from the project root directory."
        )

    # yaml.safe_load reads the YAML and converts it to a Python dict.
    # "safe" means it won't execute arbitrary code embedded in YAML
    # (a security best practice).
    with open(config_path) as f:
        return yaml.safe_load(f)
