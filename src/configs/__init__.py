import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration manager for overide"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "base_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """Get nested config value: config.get('data.batch_size')"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def update(self, updates: Dict[str, Any]):
        """Update config dynamically"""
        for key, value in updates.items():
            keys = key.split('.')
            d = self.config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    
    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)
