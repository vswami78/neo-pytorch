import json
from types import SimpleNamespace

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path

    def load_and_validate(self):
        with open(self.config_path, 'r') as config_file:
            config_data = json.load(config_file)
        
        # Convert to SimpleNamespace for dot notation access
        config = SimpleNamespace(**config_data)
        
        # Validate required fields
        required_fields = ['input_file', 'node_types', 'edge_types', 'node_color_map', 'edge_color_map']
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"{field} is missing from the configuration")
        
        return config