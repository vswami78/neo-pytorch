import json
from typing import Dict, List, Any, Tuple
import pandas as pd
from dataclasses import dataclass, field
import os

@dataclass
class Config:
    input_file: str
    node_types: Dict[str, str]
    edge_types: List[Tuple[str, str, str]]
    node_color_map: Dict[str, str]
    edge_color_map: Dict[str, str]
    default_edge_color: str = "#888888"

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None

    def load_and_validate(self) -> Config:
        raw_config = self._load_config()
        self._validate_config(raw_config)
        self._validate_against_data(raw_config)
        
        # Convert edge_types to tuples
        raw_config['edge_types'] = [tuple(edge) for edge in raw_config['edge_types']]
        
        self.config = Config(**raw_config)
        return self.config

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as config_file:
                return json.load(config_file)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file: {self.config_path}")
        except IOError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def _validate_config(self, config: Dict[str, Any]):
        required_keys = ['input_file', 'node_types', 'edge_types', 'node_color_map', 'edge_color_map']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")

        if not isinstance(config['node_types'], dict):
            raise ValueError("node_types must be a dictionary mapping node labels to column names")

        if not isinstance(config['edge_types'], list):
            raise ValueError("edge_types must be a list of [subject, relationship, object] lists")

        for edge in config['edge_types']:
            if not isinstance(edge, list) or len(edge) != 3:
                raise ValueError(f"Invalid edge type: {edge}. Must be a list with exactly 3 elements.")

    def _validate_against_data(self, config: Dict[str, Any]):
        df = pd.read_csv(config['input_file'])
        
        for node_label, column_name in config['node_types'].items():
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' for node type '{node_label}' not found in input data")

        for edge in config['edge_types']:
            subject, _, object_ = edge
            subject_column = config['node_types'].get(subject)
            object_column = config['node_types'].get(object_)
            if subject_column not in df.columns or object_column not in df.columns:
                raise ValueError(f"Invalid edge type: {edge}")

        for node_type in config['node_types'].keys():
            if node_type not in config['node_color_map']:
                raise ValueError(f"Missing color for node type: {node_type}")

        for edge in config['edge_types']:
            subject, _, object_ = edge
            edge_key = f"{subject}_{object_}"
            if edge_key not in config['edge_color_map']:
                config['edge_color_map'][edge_key] = config.get('default_edge_color', "#888888")
                print(f"Warning: Using default color for edge type: {edge_key}")

    def get_config(self) -> Config:
        if self.config is None:
            raise ValueError("Config not loaded. Call load_and_validate() first.")
        return self.config