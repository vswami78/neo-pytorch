import json
from typing import Dict, List, Any, Tuple
import pandas as pd
from dataclasses import dataclass

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
        self.raw_config = None
        self.config = None

    def load_and_validate(self) -> Config:
        """Load, validate, and return the configuration."""
        self.raw_config = self._load_config_file()
        self._validate_config()
        self._process_config()
        return self.config

    def _load_config_file(self) -> Dict[str, Any]:
        """Load the configuration file."""
        try:
            with open(self.config_path, 'r') as config_file:
                return json.load(config_file)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file: {self.config_path}")
        except IOError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def _validate_config(self):
        """Validate the configuration structure and data."""
        self._check_required_keys()
        self._validate_node_types()
        self._validate_edge_types()
        self._validate_against_data()

    def _check_required_keys(self):
        required_keys = ['input_file', 'node_types', 'edge_types', 'node_color_map', 'edge_color_map']
        for key in required_keys:
            if key not in self.raw_config:
                raise ValueError(f"Missing required key in config: {key}")

    def _validate_node_types(self):
        if not isinstance(self.raw_config['node_types'], dict):
            raise ValueError("node_types must be a dictionary mapping node labels to column names")

    def _validate_edge_types(self):
        if not isinstance(self.raw_config['edge_types'], list):
            raise ValueError("edge_types must be a list of [src, rel, dst] lists")
        for edge in self.raw_config['edge_types']:
            if not isinstance(edge, list) or len(edge) != 3:
                raise ValueError(f"Invalid edge type: {edge}. Must be a list with exactly 3 elements [src, rel, dst].")

    def _validate_against_data(self):
        df = pd.read_csv(self.raw_config['input_file'])
        self._validate_node_columns(df)
        self._validate_edge_columns(df)
        self._validate_color_maps()

    def _validate_node_columns(self, df):
        for node_label, column_name in self.raw_config['node_types'].items():
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' for node type '{node_label}' not found in input data")

    def _validate_edge_columns(self, df):
        for edge in self.raw_config['edge_types']:
            src, _, dst = edge
            src_column = self.raw_config['node_types'].get(src)
            dst_column = self.raw_config['node_types'].get(dst)
            if src_column not in df.columns or dst_column not in df.columns:
                raise ValueError(f"Invalid edge type: {edge}. Columns for src or dst not found in input data.")

    def _validate_color_maps(self):
        for node_type in self.raw_config['node_types'].keys():
            if node_type not in self.raw_config['node_color_map']:
                raise ValueError(f"Missing color for node type: {node_type}")

        for edge in self.raw_config['edge_types']:
            src, _, dst = edge
            edge_key = f"{src}_{dst}"
            if edge_key not in self.raw_config['edge_color_map']:
                self.raw_config['edge_color_map'][edge_key] = self.raw_config.get('default_edge_color', "#888888")
                print(f"Warning: Using default color for edge type: {edge_key}")

    def _process_config(self):
        """Process and create the final Config object."""
        self.raw_config['edge_types'] = [tuple(edge) for edge in self.raw_config['edge_types']]
        self.config = Config(**self.raw_config)

    def get_config(self) -> Config:
        if self.config is None:
            raise ValueError("Config not loaded. Call load_and_validate() first.")
        return self.config