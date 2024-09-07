import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import logging

# logger = logging.getLogger(__name__)

def create_heterogeneous_graph(data_path, node_types, edge_types, logger):
    """
    Create a heterogeneous graph from tabular data.
    
    :param data_path: Path to the CSV file containing the tabular data
    :param node_types: Dictionary mapping node types to their identifying columns
    :param edge_types: List of tuples defining edge types (source, edge_type, target)
    :return: PyTorch Geometric HeteroData object
    """
    if isinstance(data_path, HeteroData):
        return data_path  # Return the HeteroData object directly
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    logger.info(f"DataFrame columns: {df.columns}")
    data = HeteroData()
    
    # Create nodes
    for node_type, id_col in node_types.items():
        unique_ids = df[id_col].unique()
        data[node_type].x = torch.arange(len(unique_ids))
        data[node_type].mapping = {id: i for i, id in enumerate(unique_ids)}
        # Store original IDs for display
        data[node_type].original_ids = unique_ids
        # Store hashed IDs for internal use
        hashed_ids = np.array([hash(id) for id in unique_ids], dtype=np.int64)
        data[node_type][f'{node_type}_id'] = torch.tensor(hashed_ids, dtype=torch.long)
        logger.info(f"Created {len(unique_ids)} {node_type} nodes")
        logger.debug(f"Sample {node_type} IDs: {unique_ids[:5]}")
    
    # Create edges
    logger.info(f"Edge types to create: {edge_types}")
    for src_type, edge_type, dst_type in edge_types:
        logger.info(f"Creating edges for {src_type}-{edge_type}-{dst_type}")
        src_col = node_types[src_type]
        dst_col = node_types[dst_type]
        logger.debug(f"Source column: {src_col}, Destination column: {dst_col}")
        
        src_ids = df[src_col].map(lambda x: data[src_type].mapping.get(x, -1))
        dst_ids = df[dst_col].map(lambda x: data[dst_type].mapping.get(x, -1))
        
        mask = (src_ids != -1) & (dst_ids != -1)
        src_ids = src_ids[mask]
        dst_ids = dst_ids[mask]
        
        logger.info(f"Number of valid edges: {len(src_ids)}")
        
        if len(src_ids) > 0:
            edge_index = torch.stack([torch.tensor(src_ids.values), torch.tensor(dst_ids.values)], dim=0)
            data[src_type, edge_type, dst_type].edge_index = edge_index
            logger.info(f"Added {edge_index.shape[1]} edges for {src_type}-{edge_type}-{dst_type}")
        else:
            logger.warning(f"No valid edges found for {src_type}-{edge_type}-{dst_type}")
    
    logger.info(f"Final graph node types: {data.node_types}")
    logger.info(f"Final graph edge types: {data.edge_types}")
    return data

def create_example_graph():
    """
    Creates an example heterogeneous graph using sample data.
    
    :return: PyTorch Geometric HeteroData object
    """
    data_path = "./data/user-prod-categ-team.csv"
    node_types = {
        "user": "user_id",
        "product": "product_id",
        "category": "category_id",
        "team": "team_id"
    }
    edge_types = [
        ("user", "purchases", "product"),
        ("product", "belongs_to", "category"),
        ("category", "belongs_to", "team")
    ]

    return create_heterogeneous_graph(data_path, node_types, edge_types)

if __name__ == "__main__":
    graph = create_example_graph()
    print(graph)