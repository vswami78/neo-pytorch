import json
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger("graph_explorer")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse the configuration file.
    
    :param config_path: Path to the JSON configuration file
    :return: Dictionary containing the parsed configuration
    """
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        required_keys = ['input_file', 'node_types', 'edge_types', 'node_color_map', 'edge_color_map']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required key in config: {key}")
        
        # Convert edge_color_map keys to tuples
        config['edge_color_map'] = {tuple(k.split('_')): v for k, v in config['edge_color_map'].items()}
        
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file: {e}")
        raise
    except KeyError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        raise

def validate_config(config: Dict[str, Any], df: pd.DataFrame) -> None:
    """
    Validate the configuration against the loaded data.
    
    :param config: Parsed configuration dictionary
    :param df: Loaded DataFrame
    """
    # Collect all column names from node_types
    all_columns = set()
    for columns in config['node_types'].values():
        if isinstance(columns, str):
            all_columns.add(columns)
        elif isinstance(columns, list):
            all_columns.update(columns)
    
    # Check if all specified columns exist in the DataFrame
    missing_columns = all_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns specified in config not found in data: {missing_columns}")
    
    # Validate edge types
    node_types = set(config['node_types'].keys())
    for edge in config['edge_types']:
        if len(edge) != 3:
            raise ValueError(f"Invalid edge specification: {edge}. Should be [source, edge_type, target]")
        if edge[0] not in node_types or edge[2] not in node_types:
            raise ValueError(f"Invalid node type in edge specification: {edge}")