import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import logging

def create_heterogeneous_graph(config, logger):
    """
    Create a heterogeneous graph from tabular data.
    
    :param config: Configuration object containing input_file, node_types, and edge_types
    :param logger: Logger object for logging information and errors
    :return: PyTorch Geometric HeteroData object
    """    
    df = _load_data(config.input_file, logger)
    data = HeteroData()
    _create_nodes(df, config.node_types, data, logger)
    _create_edges(df, config.edge_types, config.node_types, data, logger)
    return data

def _load_data(input_file, logger):
    df = pd.read_csv(input_file)
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    logger.info(f"DataFrame columns: {df.columns}")
    return df

def _create_nodes(df, node_types, data, logger):
    for node_type, id_col in node_types.items():
        unique_ids = df[id_col].unique()
        data[node_type].x = torch.arange(len(unique_ids))
        data[node_type].mapping = {id: i for i, id in enumerate(unique_ids)}
        data[node_type].original_ids = unique_ids
        hashed_ids = np.array([hash(id) for id in unique_ids], dtype=np.int64)
        data[node_type][id_col] = torch.tensor(hashed_ids, dtype=torch.long)
        logger.info(f"Created {len(unique_ids)} {node_type} nodes")
        logger.debug(f"Sample {node_type} IDs: {unique_ids[:5]}")

def _create_edges(df, edge_types, node_types, data, logger):
    for src_type, edge_type, dst_type in edge_types:
        _create_edge(df, src_type, edge_type, dst_type, node_types, data, logger)

def _create_edge(df, src_type, edge_type, dst_type, node_types, data, logger):
    src_col, dst_col = node_types[src_type], node_types[dst_type]
    src_indices = df[src_col].map(data[src_type].mapping)
    dst_indices = df[dst_col].map(data[dst_type].mapping)
    mask = src_indices.notna() & dst_indices.notna()
    edge_index = torch.stack([
        torch.tensor(src_indices[mask].values, dtype=torch.long),
        torch.tensor(dst_indices[mask].values, dtype=torch.long)
    ], dim=0)
    data[src_type, edge_type, dst_type].edge_index = edge_index
    logger.info(f"Added {edge_index.shape[1]} edges for {src_type}-{edge_type}-{dst_type}")

# # The create_example_graph function can be updated or removed if it's not needed anymore

# if __name__ == "__main__":
#     # This section might need to be updated or removed depending on how you want to test the function
#     pass