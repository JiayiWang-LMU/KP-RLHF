"""
Helper module to load YAML configuration files.
"""

import yaml
from pathlib import Path

class DictToObject:
    """Convert dictionary to object with attribute access."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return f"<Config {self.__dict__}>"


def load_config(config_path: str) -> DictToObject:
    """
    Load YAML configuration file and convert to object with attribute access.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration object with nested attribute access
        
    Example:
        config = load_config('configs/config_M_pair.yaml')
        print(config.training.learning_rate)  # Access as attributes
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to object for attribute access
    config_obj = DictToObject(config_dict)
    
    return config_obj


if __name__ == '__main__':
    # Test the loader
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'configs/config_M_pair.yaml'
    
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    print("\n=== Configuration ===")
    print(f"Dataset path: {config.dataset.path}")
    print(f"Model name: {config.model.name}")
    print(f"Output dir: {config.training.output_dir}")
    print(f"Learning rate: {config.training.learning_rate} (type: {type(config.training.learning_rate)})")
    print(f"Batch size: {config.training.per_device_batch_size}")
    print(f"Epochs: {config.training.num_train_epochs}")
    print(f"GPU preset: {config.gpu_preset}")
